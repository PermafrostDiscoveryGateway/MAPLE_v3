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
import gdal_virtual_file_path as gdal_vfp



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
    # Prepare to make directories to create the files
    try:
        shutil.rmtree(worker_water_subroot)
    except:
        print("directory deletion failed")
        pass

    # check local storage for temporary storage
    Path(worker_water_subroot).mkdir(parents=True, exist_ok=True)

    output_watermask = os.path.join(
        worker_water_subroot, r"%s_watermask.tif" % image_name
    )
    
    # Create a gdal virtual file path so we don't have to create
    # and store a temp file to disk to then read it right after
    # storing it.
    output_tif_8b_virtual_file = os.path.join(
        "/vsimem/vsidir", r"%s_8bit.tif" % image_name
    )
    nir_band = 4  # set number of NIR band

    # %% Median and Otsu
    value = 5

    # Create virtual file path for image to use GDAL's file apis.
    with gdal_vfp.GDALVirtualFilePath(
        file_path=row["path"], file_bytes=row["bytes"]) as virtual_file_path:
        # UPDATED CODE - amal 01/11/2023
        # cmd line execution thrown exceptions unable to capture
        # Using gdal to execute the gdal_Translate
        # output file checked against the cmd line gdal_translate
        gdal.UseExceptions()  # Enable errors
        try:
            gdal.Translate(
                destName=output_tif_8b_virtual_file,
                srcDS=virtual_file_path,
                format="GTiff",
                outputType=gdal.GDT_Byte,
            )
        except RuntimeError:
            print("gdal Translate failed with", gdal.GetLastErrorMsg())
            pass

        with gdal.Open(output_tif_8b_virtual_file) as output_tif_8b:
            nir = np.array(output_tif_8b.GetRasterBand(nir_band).ReadAsArray())
            bilat_img = filters.rank.median(nir, disk(value))
            gdal.Unlink(output_tif_8b_virtual_file)

        with gdal.Open(virtual_file_path) as gtif:
            geotransform = gtif.GetGeoTransform()
            sourceSR = gtif.GetProjection()

    x = np.shape(nir)[1]
    y = np.shape(nir)[0]

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
    del dst_ds
    row["mask"] = mask
    return row
