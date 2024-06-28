"""
MAPLE Workflow
Main Script that runs the inference workflow pipeline.
Pre Process
1. Create water mask
2. Image Tiling
Classification / Inference
Post processing
3. Stich back to the original image dims from the tiles (2.)


Project: Permafrost Discovery Gateway: Mapping Application for Arctic Permafrost Land Environment(MAPLE)
PI      : Chandi Witharana
"""

import argparse
import os
from typing import Any, Dict

import tensorflow as tf
import ray

from mpl_config import MPL_Config
import ray_image_preprocessing
import ray_infer_tiles as ray_inference
import ray_write_shapefiles
import ray_tile_and_stitch_util


def create_geotiff_images_dataset(config: MPL_Config) -> ray.data.Dataset:
    if config.GCP_FILESYSTEM is not None:
        return ray.data.read_binary_files(config.INPUT_IMAGE_DIR + "/", filesystem=config.GCP_FILESYSTEM, include_paths=True)
    return ray.data.read_binary_files(config.INPUT_IMAGE_DIR, include_paths=True)


def add_image_name(row: Dict[str, Any]) -> Dict[str, Any]:
    row["image_name"] = os.path.basename(row["path"]).split(".tif")[0]
    return row


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    parser = argparse.ArgumentParser(
        description="Extract IWPs from satellite image scenes using MAPLE."
    )

    # Optional Arguments
    parser.add_argument(
        "--image",
        required=False,
        default="test_image_01.tif",
        metavar="<command>",
        help="Image name",
    )

    parser.add_argument(
        "--root_dir",
        required=False,
        default="",
        help="The directory path from where the workflow is running. If none is "
        "provided, the current working directory will be used by the workflow. "
        "If the root directory starts with gcs:// or gs:// then the workflow will "
        "read and write to the google cloud storage buckets.",
    )

    parser.add_argument(
        "--adc_dir",
        required=False,
        default="",
        help="The directory path for application default credentials (adc). This path must be set if "
        "you want to give ray access to your gcs buckets when you are running this workflow on your "
        "*local computer*. It is necessary for service account impersonation, which is used to give "
        "this code access to your storage bucket when running the code locally."
    )

    parser.add_argument(
        "--weight_file",
        required=False,
        default="hyp_best_train_weights_final.h5",
        help="The file path to where the model weights can be found. Should be "
        "relative to the root directory.",
    )

    parser.add_argument(
        "--gpus_per_core",
        required=False,
        default=1,
        help="Number of GPUs available per core. Used to determine how many "
        "inference processes to spin up. Set this to 0 if you want to run the "
        "workflow on a CPU.",
        type=int
    )

    args = parser.parse_args()

    image_name = args.image
    config = MPL_Config(
        args.root_dir, args.adc_dir, args.weight_file, num_gpus_per_core=args.gpus_per_core
    )

    print("0. Load geotiffs into ray dataset")
    dataset = create_geotiff_images_dataset(config).map(add_image_name, num_cpus=3)
    print("1. Start calculating watermask")
    dataset_with_water_mask = dataset.map(fn=ray_image_preprocessing.cal_water_mask,
                                          fn_kwargs={"config": config}, num_cpus=3)
    print("2. Start tiling image")
    image_tiles_dataset = dataset_with_water_mask.flat_map(
        fn=ray_tile_and_stitch_util.tile_image, fn_kwargs={"config": config}, num_cpus=3)
    image_tiles_dataset = image_tiles_dataset.drop_columns(["mask"])
    print("3. Start inferencing")
    inferenced_dataset = image_tiles_dataset.map(
        fn=ray_inference.MaskRCNNPredictor, fn_constructor_kwargs={"config": config}, num_gpus=args.gpus_per_core, concurrency=2)
    print("4. Start stitching")
    data_per_image = inferenced_dataset.groupby(
        "image_name").map_groups(ray_tile_and_stitch_util.stitch_shapefile, num_cpus=3)
    print("5. Write shapefiles")
    shapefiles_dataset = data_per_image.map(
        fn=ray_write_shapefiles.WriteShapefiles, fn_constructor_kwargs={"config": config}, concurrency=1, num_cpus=4)
    shapefiles_dataset.materialize()    
# shp, shx, dbf as layers.
# You can also look at compare_shapefile_features.py for how to compare the features in two shapefiles.
# You can also use 'ogrinfo -so -al <path to shapefile>' on the command line to examine a shapefile.
