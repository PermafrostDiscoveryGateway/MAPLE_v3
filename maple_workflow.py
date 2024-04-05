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
import copy
import cv2
import mpl_clean_inference as inf_clean
import mpl_infer_tiles_ray as ray_inference
import write_shapefiles_ray as write_shapefiles
import mpl_process_shapefile as process
import mpl_stitchshpfile_new as stich
import mpl_stitchshpfile_ray as ray_stitch
import numpy as np
import os
import ray
import sys
import shutil
from pathlib import Path

from dataclasses import dataclass
from mpl_config import MPL_Config
from osgeo import gdal
from skimage import color, filters, io
from skimage.morphology import disk
import tensorflow as tf
from typing import Any, Dict, List

# work tag
WORKTAG = 1
DIETAG = 0


@dataclass
class ImageMetadata:
    len_x_list: int
    len_y_list: int
    x_resolution: float
    y_resolution: float


@dataclass
class ImageTileMetadata:
    upper_left_row: float
    upper_left_col: float
    id_i: int
    id_j: int
    tile_num: int


@dataclass
class ImageTile:
    tile_values: np.array
    tile_metadata: ImageTileMetadata


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


def cal_water_mask(row: Dict[str, Any], config: MPL_Config) -> Dict[str, Any]:
    """
    This will calculate the water mask to avoid (inference) processing of the masked areas with water
    Uses gdal to transform the image into the required format.
    Parameters
    ----------
    row : Row in Ray Dataset, there is one row per image.
    config : Contains static configuration information regarding the workflow.
    """
    image_name = row["image_file_name"].split(".tif")[0]

    # os.path.join(worker_root, "water_shp")
    worker_water_root = config.WATER_MASK_DIR
    temp_water_root = (
        config.TEMP_W_IMG_DIR
    )  # os.path.join(worker_root, "temp_8bitmask")

    worker_water_subroot = os.path.join(worker_water_root, image_name)
    temp_water_subroot = os.path.join(temp_water_root, image_name)
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
        worker_water_subroot, r"%s_watermask.tif" % image_name
    )
    output_tif_8b_file = os.path.join(
        temp_water_subroot, r"%s_8bit.tif" % image_name
    )
    nir_band = 3  # set number of NIR band

    # %% Median and Otsu
    value = 5

    input_image_file_path = row["vfs_image_path"]

    # UPDATED CODE - amal 01/11/2023
    # cmd line execution thrown exceptions unable to capture
    # Using gdal to execute the gdal_Translate
    # output file checked against the cmd line gdal_translate
    gdal.UseExceptions()  # Enable errors
    try:
        gdal.Translate(
            destName=output_tif_8b_file,
            srcDS=input_image_file_path,
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

    gtif = gdal.Open(input_image_file_path)
    geotransform = gtif.GetGeoTransform()
    sourceSR = gtif.GetProjection()
    # Close the file.
    gtif = None

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
    row["mask"] = mask
    return row


def tile_image(row: Dict[str, Any], config: MPL_Config) -> List[Dict[str, Any]]:
    """
    Tile the image into multiple pre-deifined sized parts so that the processing can be done on smaller parts due to
    processing limitations

    Parameters
    ----------
    config : Contains static configuration information regarding the workflow.
    input_img_name : Name of the input image
    """

    input_image_gtif = gdal.Open(row["vfs_image_path"])
    mask = row["mask"]

    # convert the original image into the geo cordinates for further processing using gdal
    # https://gdal.org/tutorials/geotransforms_tut.html
    # GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
    # GT(1) w-e pixel resolution / pixel width.
    # GT(2) row rotation (typically zero).
    # GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
    # GT(4) column rotation (typically zero).
    # GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).

    # ---------------------- crop image from the water mask----------------------
    # dot product of the mask and the orignal data before breaking it for processing
    # Also band 2 3 and 4 are taken because the 4 bands cannot be processed by the NN learingin algo
    # Need to make sure that the training bands are the same as the bands used for inferencing.
    #
    final_array_2 = input_image_gtif.GetRasterBand(2).ReadAsArray()
    final_array_3 = input_image_gtif.GetRasterBand(3).ReadAsArray()
    final_array_4 = input_image_gtif.GetRasterBand(4).ReadAsArray()

    final_array_2 = np.multiply(final_array_2, mask)
    final_array_3 = np.multiply(final_array_3, mask)
    final_array_4 = np.multiply(final_array_4, mask)

    # ulx, uly is the upper left corner
    ulx, x_resolution, _, uly, _, y_resolution = input_image_gtif.GetGeoTransform()

    # ---------------------- Divide image (tile) ----------------------
    overlap_rate = 0.2
    block_size = config.CROP_SIZE
    ysize = input_image_gtif.RasterYSize
    xsize = input_image_gtif.RasterXSize

    # Close the file.
    input_image_gtif = None

    tile_count = 0

    y_list = range(0, ysize, int(block_size * (1 - overlap_rate)))
    x_list = range(0, xsize, int(block_size * (1 - overlap_rate)))

    # ---------------------- Find each Upper left (x,y) for each divided images ----------------------
    #  ***-----------------
    #  ***
    #  ***
    #  |
    #  |
    #
    tiles = []
    for id_i, i in enumerate(y_list):
        # don't want moving window to be larger than row size of input raster
        if i + block_size < ysize:
            rows = block_size
        else:
            rows = ysize - i

        # read col
        for id_j, j in enumerate(x_list):
            if j + block_size < xsize:
                cols = block_size
            else:
                cols = xsize - j
            # get block out of the whole raster
            # todo check the array values is similar as ReadAsArray()
            band_1_array = final_array_4[i: i + rows, j: j + cols]
            band_2_array = final_array_2[i: i + rows, j: j + cols]
            band_3_array = final_array_3[i: i + rows, j: j + cols]

            # filter out black image
            if (
                band_3_array[0, 0] == 0
                and band_3_array[0, -1] == 0
                and band_3_array[-1, 0] == 0
                and band_3_array[-1, -1] == 0
            ):
                continue

            # stack three bands into one array
            img = np.stack((band_1_array, band_2_array, band_3_array), axis=2)
            cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)
            B, G, R = cv2.split(img)
            out_B = cv2.equalizeHist(B)
            out_R = cv2.equalizeHist(R)
            out_G = cv2.equalizeHist(G)
            final_image = np.array(cv2.merge((out_B, out_G, out_R)))        

            # Upper left (x,y) for each images
            ul_row_divided_img = uly + i * y_resolution
            ul_col_divided_img = ulx + j * x_resolution

            tile_metadata = ImageTileMetadata(
                upper_left_row=ul_row_divided_img, upper_left_col=ul_col_divided_img, tile_num=tile_count, id_i=id_i, id_j=id_j)
            image_tile = ImageTile(
                tile_values=final_image, tile_metadata=tile_metadata)
            tiles.append(image_tile)
            tile_count += 1

    # --------------- Store all the title as an object file
    image_metadata = ImageMetadata(
        len_x_list=len(x_list), len_y_list=len(y_list), x_resolution=x_resolution, y_resolution=y_resolution)
    row["image_metadata"] = image_metadata
    new_rows = []
    for image_tile in tiles:
        new_row = copy.deepcopy(row)
        new_row["image_tile"] = image_tile
        new_rows.append(new_row)
    return new_rows


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
    print("1. start calculating watermask")
    dataset_with_water_mask = dataset.map(fn=cal_water_mask,
                                          fn_kwargs={"config": config})
    print("dataset with water mask: ", dataset_with_water_mask.schema())
    print("2. start tiling image")
    image_tiles_dataset = dataset_with_water_mask.flat_map(
        fn=tile_image, fn_kwargs={"config": config})
    #image_tiles_dataset = image_tiles_dataset.drop_columns(["bytes"])
    image_tiles_dataset = image_tiles_dataset.drop_columns(["mask"])
    print("dataset with image tiles: ", image_tiles_dataset.schema())
    print("3. start inferencing")
    inferenced_dataset = image_tiles_dataset.map(
        fn=ray_inference.MaskRCNNPredictor, fn_constructor_kwargs={"config": config}, concurrency=2)
    print("inferenced:", inferenced_dataset.schema())
    print("4. start stiching")
    data_per_image = inferenced_dataset.groupby("image_file_name").map_groups(ray_stitch.stitch_shapefile)
    print("grouped by file: ", data_per_image.schema())
    print("5. experimenting with shapefiles")
    shapefiles_dataset = data_per_image.map(
        fn=write_shapefiles.WriteShapefiles, fn_constructor_kwargs={"shpfile_output_dir": config.TEST_SHAPEFILE}, concurrency=2)
    print("done writing to shapefiles", shapefiles_dataset.schema())
    """
    process.process_shapefile(config, image_name)
    print("5. start cleaning")
    inf_clean.clean_inference_shapes(
        config.CLEAN_DATA_DIR,
        config.PROJECTED_SHP_DIR,
        "./data/input_bound/sample2_out_boundry.shp",
    )
    """

# Once you are done you can check the output on ArcGIS (win) or else you can check in QGIS (nx) Add the image and the
# shp, shx, dbf as layers.
