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

import functools
import mpl_clean_inference as inf_clean
import mpl_divideimg_234_water_new as divide
import mpl_infer_tiles_GPU_new as inference
import mpl_process_shapefile as process
import mpl_stitchshpfile_new as stich
import numpy as np
import os
import sys
import shutil
import tensorflow as tf

from flytekit import task, workflow, map_task
from mpl_config import MPL_Config
from osgeo import gdal
from pathlib import Path
from skimage import filters, io
from skimage.morphology import disk
from typing import List

# work tag
WORKTAG = 1
DIETAG = 0


@task
def tile_image(config: MPL_Config, input_img_name: str):
    """
    Tile the image into multiple pre-deifined sized parts so that the processing can be done on smaller parts due to
    processing limitations

    Parameters
    ----------
    config : Contains static configuration information regarding the workflow.
    input_img_name : Name of the input image
    """
    print("2. start tiling")
    sys.path.append(config.ROOT_DIR)

    crop_size = config.CROP_SIZE

    # worker roots
    worker_img_root = config.INPUT_IMAGE_DIR
    worker_divided_img_root = config.DIVIDED_IMAGE_DIR
    # input image path
    input_img_path = os.path.join(worker_img_root, input_img_name)

    # Create subfolder for each image
    new_file_name = input_img_name.split(".tif")[0]
    worker_divided_img_subroot = os.path.join(worker_divided_img_root, new_file_name)

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


@task
def cal_water_mask(config: MPL_Config, input_img_name: str):
    """
    This will calculate the water mask to avoid (inference) processing of the masked areas with water
    Uses gdal to transform the image into the required format.
    Parameters
    ----------
    config : Contains static configuration information regarding the workflow.
    input_img_name : Name of the input image
    """
    print("1.start caculating wartermask")
    image_file_name = (input_img_name).split(".tif")[0]

    worker_water_root = config.WATER_MASK_DIR  # os.path.join(worker_root, "water_shp")
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

    ### UPDATED CODE - amal 01/11/2023
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
    normalized_blurred_bilat_img = filters.gaussian(normalized_bilat_img, sigma=2.0)

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


@task
def infer_image(config: MPL_Config, input_img_name: str):
    """
    Inference based on the trained model reperesented by the saved weights

    Parameters
    ----------
    config : Contains static configuration information regarding the workflow.
    input_img_name : Name of the input image file
    """
    print("3. start inferencing")
    sys.path.append(config.ROOT_DIR)

    # worker roots
    worker_divided_img_root = config.DIVIDED_IMAGE_DIR

    # Create subfolder for each image
    new_file_name = input_img_name.split(".tif")[0]
    worker_divided_img_subroot = os.path.join(worker_divided_img_root, new_file_name)

    print(worker_divided_img_subroot)

    file1 = os.path.join(worker_divided_img_subroot, "image_data.h5")
    file2 = os.path.join(worker_divided_img_subroot, "image_param.h5")

    worker_output_shp_root = config.OUTPUT_SHP_DIR
    worker_output_shp_subroot = os.path.join(worker_output_shp_root, new_file_name)
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


@task
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
    print("4. start stiching")
    sys.path.append(config.ROOT_DIR)

    worker_finaloutput_root = config.FINAL_SHP_DIR
    worker_output_shp_root = config.OUTPUT_SHP_DIR

    # Create subfolder for each image within the worker img root
    new_file_name = input_img_name.split(".tif")[0]

    worker_finaloutput_subroot = os.path.join(worker_finaloutput_root, new_file_name)
    worker_output_shp_subroot = os.path.join(worker_output_shp_root, new_file_name)

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
    process.process_shapefile(config, input_img_name)

    print("done stitching")


@task
def clean_inference(config: MPL_Config, data_boundray_file: str):
    print("5. start cleaning")
    inf_clean.clean_inference_shapes(
        config.CLEAN_DATA_DIR,
        config.PROJECTED_SHP_DIR,
        data_boundray_file,
    )


@task
def setup(root_dir: str, weight_file: str, gpus_per_core: int) -> MPL_Config:
    # disable TF eager execution
    tf.compat.v1.disable_eager_execution()
    return MPL_Config(root_dir, weight_file, num_gpus_per_core=gpus_per_core)


@task
def get_image_files(config: MPL_Config) -> List[str]:
    # TODO: we could clean up some duplicate code by removing the .tif extension
    # here instead of in each individual task but that will take some refactoring
    # leaving it for a later time.
    return os.listdir(config.INPUT_IMAGE_DIR)


# Once flyte is installed on the environment you can run the workflow with the
# following command:
# pyflyte run maple_workflow.py maple_workflow --gpus_per_core 0
@workflow
def maple_workflow(
    root_dir: str = "",
    weight_file: str = "hyp_best_train_weights_final.h5",
    gpus_per_core: int = 1,
):
    config = setup(
        root_dir=root_dir, weight_file=weight_file, gpus_per_core=gpus_per_core
    )
    image_files = get_image_files(config=config)
    partial_t0 = functools.partial(cal_water_mask, config=config)
    t0 = map_task(partial_t0)(input_img_name=image_files)
    partial_t1 = functools.partial(tile_image, config=config)
    t1 = map_task(partial_t1)(input_img_name=image_files)
    partial_t2 = functools.partial(infer_image, config=config)
    t2 = map_task(partial_t2)(input_img_name=image_files)
    partial_t3 = functools.partial(stich_shapefile, config=config)
    t3 = map_task(partial_t3)(input_img_name=image_files)
    # clean inference is set up as a bulk operator on the whole directory. Should
    # probably refactor it to clean the inference of individual images to follow
    # the same patterns as the tasks above.
    t4 = clean_inference(
        config=config, data_boundray_file="./data/input_bound/sample2_out_boundry.shp"
    )
    t0 >> t1
    t1 >> t2
    t2 >> t3
    t3 >> t4

# Once you are done you can check the output on ArcGIS (win) or else you can check in QGIS (nx) Add the image and the
# shp, shx, dbf as layers.
