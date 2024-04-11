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
import mpl_preprocessing as preprocessing
import mpl_infer_tiles_ray as ray_inference
import write_shapefiles_ray as write_shapefiles
import mpl_tile_and_stitch_ray_util as tile_and_stitch_util


def create_geotiff_images_dataset(input_image_dir: str) -> ray.data.Dataset:
    return ray.data.read_binary_files(input_image_dir, include_paths=True)


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
        "provided, the current working directory will be used by the workflow.",
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
        args.root_dir, args.weight_file, num_gpus_per_core=args.gpus_per_core
    )

    print("0. Load geotiffs into ray dataset")
    dataset = create_geotiff_images_dataset(
        config.INPUT_IMAGE_DIR).map(add_image_name)
    print("Ray dataset schema:", dataset.schema())
    print("1. Start calculating watermask")
    dataset_with_water_mask = dataset.map(fn=preprocessing.cal_water_mask,
                                          fn_kwargs={"config": config})
    print("Dataset schema with water mask: ", dataset_with_water_mask.schema())
    print("2. Start tiling image")
    image_tiles_dataset = dataset_with_water_mask.flat_map(
        fn=tile_and_stitch_util.tile_image, fn_kwargs={"config": config})
    image_tiles_dataset = image_tiles_dataset.drop_columns(["mask"])
    print("Dataset schema with image tiles: ", image_tiles_dataset.schema())
    print("3. Start inferencing")
    inferenced_dataset = image_tiles_dataset.map(
        fn=ray_inference.MaskRCNNPredictor, fn_constructor_kwargs={"config": config}, concurrency=2)
    print("Dataset schema with inferenced tiles: ", inferenced_dataset.schema())
    print("4. Start stitching")
    data_per_image = inferenced_dataset.groupby(
        "image_name").map_groups(tile_and_stitch_util.stitch_shapefile)
    print("Dataset schema where each row is an image (result of a group by tile): ",
          data_per_image.schema())
    print("5. Write shapefiles")
    shapefiles_dataset = data_per_image.map(
        fn=write_shapefiles.WriteShapefiles, fn_constructor_kwargs={"shpfile_output_dir": config.TEST_SHAPEFILE}, concurrency=2)
    print("Done writing shapefiles", shapefiles_dataset.schema())

# Once you are done you can check the output on ArcGIS (win) or else you can check in QGIS (nx) Add the image and the
# shp, shx, dbf as layers.
