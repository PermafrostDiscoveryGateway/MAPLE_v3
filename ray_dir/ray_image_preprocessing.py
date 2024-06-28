"""
Preprocessing helper functions for the MAPLE pipeline.

Project: Permafrost Discovery Gateway: Mapping Application for Arctic Permafrost Land Environment(MAPLE)
PI      : Chandi Witharana
"""


import os
from pathlib import Path
import shutil
from typing import Dict, Any

import numpy as np
from osgeo import gdal
from skimage import filters, io
from skimage.morphology import disk

from mpl_config import MPL_Config
import gdal_virtual_file_system as gdal_vfs



def cal_water_mask(row: Dict[str, Any], config: MPL_Config) -> Dict[str, Any]:
    """
    This will calculate the water mask to avoid (inference) processing of the masked areas with water
    Uses gdal to transform the image into the required format.
    The result of this function is that it will add a "mask" column with the water mask to each row.
    It also writes the water mask to the config.WATER_MASK_DIR.
    Parameters
    ----------
    row : Row in Ray Dataset, there is one row per image.
    config : Contains static configuration information regarding the workflow.
    """
    worker_water_root = config.WATER_MASK_DIR
    temp_water_root = (
        config.TEMP_W_IMG_DIR
    ) 

    image_name = row["image_name"]
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

    # Create virtual file system file for image to use GDAL's file apis.
    vfs = gdal_vfs.GDALVirtualFileSystem(
        file_path=row["path"], file_bytes=row["bytes"])
    virtual_file_path = vfs.create_virtual_file()

    # UPDATED CODE - amal 01/11/2023
    # cmd line execution thrown exceptions unable to capture
    # Using gdal to execute the gdal_Translate
    # output file checked against the cmd line gdal_translate
    gdal.UseExceptions()  # Enable errors
    try:
        gdal.Translate(
            destName=output_tif_8b_file,
            srcDS=virtual_file_path,
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

    gtif = gdal.Open(virtual_file_path)
    geotransform = gtif.GetGeoTransform()
    sourceSR = gtif.GetProjection()
    # Close the file.
    gtif = None
    vfs.close_virtual_file()

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
