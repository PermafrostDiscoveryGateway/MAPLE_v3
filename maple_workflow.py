"""
MAPLE Workflow
Main Script that runs the inference workflow pipeline.
Pre Process
1. Create water mask
2. Image Tiling
Classification / Inference
3. Infer "ground truth" from images based on the trained model
Post processing
4. Stich back to the original image dims from the tiles (2.)
5. Clean the output based on known ground truth

Project: Permafrost Discovery Gateway: Mapping Application for Arctic Permafrost Land Environment(MAPLE)
PI      : Chandi Witharana
Author  : Rajitha Udwalpola
"""

import argparse
import mpl_clean_inference as inf_clean
import mpl_divideimg_234_water_new as divide
import mpl_infer_tiles_GPU_new as inference
import mpl_process_shapefile as process
import mpl_stitchshpfile_new as stich
import numpy as np
import os
import ray
import sys
import shutil
from pathlib import Path

from mpl_config import MPL_Config
from osgeo import gdal
from skimage import color, filters, io
from skimage.morphology import disk
import tensorflow as tf
from typing import Any, Dict

# work tag
WORKTAG = 1
DIETAG = 0


def create_geotiff_images_dataset(input_image_dir: str) -> ray.data.Dataset:
    return ray.data.read_binary_files(input_image_dir, include_paths=True)


def add_virtual_GDAL_file_path(row: Dict[str, Any]) -> Dict[str, Any]:
    vfs_dir_path = '/vsimem/vsidir/'
    image_file_name = os.path.basename(row["path"])
    vfs_filename = os.path.join(vfs_dir_path, image_file_name)
    dst = gdal.VSIFOpenL(vfs_filename, 'wb+')
    gdal.VSIFWriteL(row["bytes"], 1, len(row["bytes"]), dst)
    gdal.VSIFCloseL(dst)
    row["image_file_name"] = image_file_name
    row["vfs_image_path"] = vfs_filename
    return row


def test_gdal_operation(row: Dict[str, Any]) -> Dict[str, Any]:
    input_image_gtif = gdal.Open(row["path"])
    vfs_image_gtif = gdal.Open(row["vfs_image_path"])
    row["gdal_test"] = np.array_equal(input_image_gtif.GetRasterBand(
        1).ReadAsArray(), vfs_image_gtif.GetRasterBand(1).ReadAsArray())
    return row


def tile_image(config: MPL_Config, input_img_name: str):
    """
    Tile the image into multiple pre-deifined sized parts so that the processing can be done on smaller parts due to
    processing limitations

    Parameters
    ----------
    config : Contains static configuration information regarding the workflow.
    input_img_name : Name of the input image
    """
    sys.path.append(config.ROOT_DIR)

    crop_size = config.CROP_SIZE

    # worker roots
    worker_img_root = config.INPUT_IMAGE_DIR
    worker_divided_img_root = config.DIVIDED_IMAGE_DIR
    # input image path
    input_img_path = os.path.join(worker_img_root, input_img_name)

    # Create subfolder for each image
    new_file_name = input_img_name.split(".tif")[0]
    worker_divided_img_subroot = os.path.join(
        worker_divided_img_root, new_file_name)

    print(worker_divided_img_subroot)

    try:
        shutil.rmtree(worker_divided_img_subroot)
    except:
        print("directory deletion failed")
        pass
    os.mkdir(worker_divided_img_subroot)

    file1 = os.path.join(worker_divided_img_subroot, "image_data.h5")
    file2 = os.path.join(worker_divided_img_subroot, "image_param.h5")
    # ----------------------------------------------------------------------------------------------------
    # Call divide image <mpl_divided_img_water> to put the water mask and also to tile and store the data
    # Multiple image overlaps are NOT taken into account in called code.
    #
    divide.divide_image(config, input_img_path, crop_size, file1, file2)

    print("finished tiling")


def cal_water_mask(config: MPL_Config, input_img_name: str):
    """
    This will calculate the water mask to avoid (inference) processing of the masked areas with water
    Uses gdal to transform the image into the required format.
    Parameters
    ----------
    config : Contains static configuration information regarding the workflow.
    input_img_name : Name of the input image
    """
    image_file_name = (input_img_name).split(".tif")[0]

    # os.path.join(worker_root, "water_shp")
    worker_water_root = config.WATER_MASK_DIR
    temp_water_root = (
        config.TEMP_W_IMG_DIR
    )  # os.path.join(worker_root, "temp_8bitmask")

    worker_water_subroot = os.path.join(worker_water_root, image_file_name)
    temp_water_subroot = os.path.join(temp_water_root, image_file_name)
    # Prepare to make directories to create the files
    try:
        shutil.rmtree(worker_water_subroot)
    except:
        print("directory deletion failed")
        pass

    try:
        shutil.rmtree(temp_water_subroot)
    except:
        print("directory deletion failed")
        pass

    # check local storage for temporary storage
    Path(worker_water_subroot).mkdir(parents=True, exist_ok=True)
    Path(temp_water_subroot).mkdir(parents=True, exist_ok=True)

    output_watermask = os.path.join(
        worker_water_subroot, r"%s_watermask.tif" % image_file_name
    )
    output_tif_8b_file = os.path.join(
        temp_water_subroot, r"%s_8bit.tif" % image_file_name
    )
    nir_band = 3  # set number of NIR band

    input_image = os.path.join(config.INPUT_IMAGE_DIR, input_img_name)

    # %% Median and Otsu
    value = 5

    # UPDATED CODE - amal 01/11/2023
    # cmd line execution thrown exceptions unable to capture
    # Using gdal to execute the gdal_Translate
    # output file checked against the cmd line gdal_translate
    gdal.UseExceptions()  # Enable errors
    try:
        gdal.Translate(
            destName=output_tif_8b_file,
            srcDS=input_image,
            format="GTiff",
            outputType=gdal.GDT_Byte,
        )
    except RuntimeError:
        print("gdal Translate failed with", gdal.GetLastErrorMsg())
        pass

    image = io.imread(
        output_tif_8b_file
    )  # image[rows, columns, dimensions]-> image[:,:,3] is near Infrared
    nir = image[:, :, nir_band]

    bilat_img = filters.rank.median(nir, disk(value))

    gtif = gdal.Open(input_image)
    geotransform = gtif.GetGeoTransform()
    sourceSR = gtif.GetProjection()

    x = np.shape(image)[1]
    y = np.shape(image)[0]

    # Normalize and blur before thresholding. Usually instead of normalizing
    # a rgb to greyscale transformation is applied. In this case, we are already
    # dealing with a single channel so we divide all the pixels in the image by
    # 255 to get a value between [0, 1].
    normalized_bilat_img = bilat_img / 255
    normalized_blurred_bilat_img = filters.gaussian(
        normalized_bilat_img, sigma=2.0)

    # find the threshold to filter if all values are same otsu cannot find value
    # hence t is made to 0.0
    try:
        t = filters.threshold_otsu(normalized_blurred_bilat_img)
    except:
        t = 0.0
    # perform inverse binary thresholding
    mask = normalized_blurred_bilat_img > t

    # output np array as GeoTiff
    dst_ds = gdal.GetDriverByName("GTiff").Create(
        output_watermask, x, y, 1, gdal.GDT_Byte, ["NBITS=1"]
    )
    dst_ds.GetRasterBand(1).WriteArray(mask)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(sourceSR)
    dst_ds.FlushCache()
    dst_ds = None


def infer_image(config: MPL_Config, input_img_name: str):
    """
    Inference based on the trained model reperesented by the saved weights

    Parameters
    ----------
    config : Contains static configuration information regarding the workflow.
    input_img_name : Name of the input image file
    """
    sys.path.append(config.ROOT_DIR)

    # worker roots
    worker_divided_img_root = config.DIVIDED_IMAGE_DIR

    # Create subfolder for each image
    new_file_name = input_img_name.split(".tif")[0]
    worker_divided_img_subroot = os.path.join(
        worker_divided_img_root, new_file_name)

    print(worker_divided_img_subroot)

    file1 = os.path.join(worker_divided_img_subroot, "image_data.h5")
    file2 = os.path.join(worker_divided_img_subroot, "image_param.h5")

    worker_output_shp_root = config.OUTPUT_SHP_DIR
    worker_output_shp_subroot = os.path.join(
        worker_output_shp_root, new_file_name)
    try:
        shutil.rmtree(worker_output_shp_subroot)
    except:
        print("directory deletion failed")
        pass

    inference.inference_image(
        config,
        worker_output_shp_subroot,
        file1,
        file2,
        new_file_name,
    )

    print("done")


def stich_shapefile(config: MPL_Config, input_img_name: str):
    """
    Put (stich) the image tiles back to the original

    Parameters
    ----------
    config : Contains static configuration information regarding the workflow.
    input_img_name : Name of the input image file

    Returns
    -------

    """
    sys.path.append(config.ROOT_DIR)

    worker_finaloutput_root = config.FINAL_SHP_DIR
    worker_output_shp_root = config.OUTPUT_SHP_DIR

    # Create subfolder for each image within the worker img root
    new_file_name = input_img_name.split(".tif")[0]

    worker_finaloutput_subroot = os.path.join(
        worker_finaloutput_root, new_file_name)
    worker_output_shp_subroot = os.path.join(
        worker_output_shp_root, new_file_name)

    try:
        shutil.rmtree(worker_finaloutput_subroot)
    except:
        print("directory deletion failed")
        pass
    os.mkdir(worker_finaloutput_subroot)

    stich.stitch_shapefile(
        config,
        worker_output_shp_subroot,
        worker_finaloutput_subroot,
        new_file_name,
        new_file_name,
    )

    return "done Divide"


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

    print("0. load geotiffs into ray dataset")
    dataset = create_geotiff_images_dataset(
        config.INPUT_IMAGE_DIR).map(add_virtual_GDAL_file_path)
    print("ray dataset:", dataset.schema())
    print("1.start caculating wartermask")
    cal_water_mask(config, image_name)
    print("2. start tiling image")
    tile_image(config, image_name)
    print("3. start inferencing")
    infer_image(config, image_name)
    print("4. start stiching")
    stich_shapefile(config, image_name)
    process.process_shapefile(config, image_name)
    print("5. start cleaning")
    inf_clean.clean_inference_shapes(
        config.CLEAN_DATA_DIR,
        config.PROJECTED_SHP_DIR,
        "./data/input_bound/sample2_out_boundry.shp",
    )

# Once you are done you can check the output on ArcGIS (win) or else you can check in QGIS (nx) Add the image and the
# shp, shx, dbf as layers.
