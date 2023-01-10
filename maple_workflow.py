"""
Main Script that runs the inference workflow pipeline.
Pre Process
1. Create watermask
2. Image Tiling
Classification
3. Infer image based on the trained model
Post processing
4. Stich image from the tiles (2.)



"""


import shutil
import argparse


import os.path
import shutil
import datetime
import os
from mpl_config import MPL_Config
import mpl_divideimg_234_water_new as divide
import mpl_infer_tiles_GPU_new as inference
import sys
import mpl_stitchshpfile_new as stich
import mpl_process_shapefile as process

# work tag
WORKTAG = 1
DIETAG = 0

def tile_image(input_img_name):
    """
    Tile the image into multiple pre-deifined sized parts so that the processing can be done on smaller parts due to processing limitations

    Parameters
    ----------
    input_img_name : Name of the input image
    """
    sys.path.append(MPL_Config.ROOT_DIR)

    crop_size = MPL_Config.CROP_SIZE

    # worker roots
    worker_root = MPL_Config.WORKER_ROOT
    worker_img_root = MPL_Config.INPUT_IMAGE_DIR
    worker_divided_img_root = MPL_Config.DIVIDED_IMAGE_DIR

    #input image path
    input_img_path = os.path.join(worker_img_root, input_img_name)

    # Create subfolder for each image
    new_file_name = input_img_name.split('.tif')[0]
    worker_divided_img_subroot = os.path.join(worker_divided_img_root, new_file_name)

    print(worker_divided_img_subroot)

    try:
        shutil.rmtree(worker_divided_img_subroot)
    except:
        print("director deletion failed")
        pass
    os.mkdir(worker_divided_img_subroot)


    file1 = (os.path.join(worker_divided_img_subroot, 'image_data.h5'))
    file2 = (os.path.join(worker_divided_img_subroot, 'image_param.h5'))

    divide.divide_image(input_img_path, crop_size,
                        file1, file2)

    print("finished tiling")

def cal_water_mask(input_img_name):
    """
    This will calculate the water mask so that we can avoid (inference) processing of the masked water areas

    Parameters
    ----------
    input_img_name : input file name
    """
    from mpl_config import MPL_Config
    import os
    from osgeo import gdal, ogr
    import numpy as np
    import skimage.color
    import skimage.filters
    import skimage.io
    import skimage.viewer
    import shutil
    from skimage.morphology import disk
    import cv2

    image_file_name = (input_img_name).split('.tif')[0]

    worker_root = MPL_Config.WORKER_ROOT
    worker_water_root = MPL_Config.WATER_MASK_DIR #  os.path.join(worker_root, "water_shp")
    temp_water_root =  MPL_Config.TEMP_W_IMG_DIR#os.path.join(worker_root, "temp_8bitmask")

    ouput_image = os.path.join(MPL_Config.OUTPUT_IMAGE_DIR,"%s.tif"%image_file_name)

    worker_water_subroot = os.path.join(worker_water_root, image_file_name)
    temp_water_subroot = os.path.join(temp_water_root, image_file_name)

    try:
        shutil.rmtree(worker_water_subroot)

    except:
  #      print("director deletion failed")
        pass

    try:
       shutil.rmtree(temp_water_subroot)
    except:
   #     print("director deletion failed")
        pass

        # check local storage for temporary storage
    os.mkdir(worker_water_subroot)
    os.mkdir(temp_water_subroot)

    output_watermask = os.path.join(worker_water_subroot, r"%s_watermask.tif" % image_file_name)
    output_tif_8b_file = os.path.join(temp_water_subroot, r"%s_8bit.tif" % image_file_name)
    nir_band = 3  # set number of NIR band

    input_image = os.path.join(MPL_Config.INPUT_IMAGE_DIR, input_img_name)

    # %% Median and Otsu
    value = 5
    clips = []

    cmd = "gdal_translate -ot Byte -of GTiff %s %s" % (input_image, output_tif_8b_file)
    os.system(cmd)

    image = skimage.io.imread(output_tif_8b_file)  # image[rows, columns, dimensions]-> image[:,:,3] is near Infrared
    nir = image[:, :, nir_band]

    bilat_img = skimage.filters.rank.median(nir, disk(value))

    gtif = gdal.Open(input_image)
    geotransform = gtif.GetGeoTransform()
    sourceSR = gtif.GetProjection()

    x = np.shape(image)[1]
    y = np.shape(image)[0]
    bands = np.shape(image)[2]

    # blur and grayscale before thresholding
    blur = skimage.color.rgb2gray(bilat_img)
    blur = skimage.filters.gaussian(blur, sigma=2.0)

    t = skimage.filters.threshold_otsu(blur)
 #   print(t)

    # perform inverse binary thresholding
    mask = blur > t
 #   print(mask.dtype)

    # output np array as GeoTiff

    dst_ds = gdal.GetDriverByName('GTiff').Create(output_watermask, x, y, 1, gdal.GDT_Byte, ['NBITS=1'])
    dst_ds.GetRasterBand(1).WriteArray(mask)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(sourceSR)
    dst_ds.FlushCache()
    dst_ds = None


def infer_image(input_img_name):
    """
    Inference based on the trained model reperesented by the saved weights

    Parameters
    ----------
    input_img_name : Name of the image file
    """
    sys.path.append(MPL_Config.ROOT_DIR)

    crop_size = MPL_Config.CROP_SIZE

    # worker roots
    worker_root = MPL_Config.WORKER_ROOT
    worker_img_root = MPL_Config.INPUT_IMAGE_DIR
    worker_divided_img_root = MPL_Config.DIVIDED_IMAGE_DIR

    #input image path
    input_img_path = os.path.join(worker_img_root, input_img_name)

    # Create subfolder for each image
    new_file_name = input_img_name.split('.tif')[0]
    worker_divided_img_subroot = os.path.join(worker_divided_img_root, new_file_name)

    print(worker_divided_img_subroot)

    file1 = (os.path.join(worker_divided_img_subroot, 'image_data.h5'))
    file2 = (os.path.join(worker_divided_img_subroot, 'image_param.h5'))

    worker_output_shp_root = MPL_Config.OUTPUT_SHP_DIR
    worker_output_shp_subroot = os.path.join(worker_output_shp_root, new_file_name)
    try:
        shutil.rmtree(worker_output_shp_subroot)

    except:
        print("director deletion failed")
        pass

    POLYGON_DIR = worker_root
    weights_path = MPL_Config.WEIGHT_PATH

    inference.inference_image(POLYGON_DIR,
                              weights_path,
                              worker_output_shp_subroot, file1, file2,new_file_name)

    print("done")




def stich_shapefile(input_img_name):
    """
    Put (stich) the image tiles back to the original

    Parameters
    ----------
    input_img_name : input file name

    Returns
    -------

    """
    sys.path.append(MPL_Config.ROOT_DIR)

    crop_size = MPL_Config.CROP_SIZE

    # worker roots - location to put the tiled files
    worker_img_root = MPL_Config.INPUT_IMAGE_DIR

    worker_finaloutput_root =  MPL_Config.FINAL_SHP_DIR
    worker_output_shp_root = MPL_Config.OUTPUT_SHP_DIR

    # Create subfolder for each image within the worker img root
    new_file_name = input_img_name.split('.tif')[0]

    worker_finaloutput_subroot = os.path.join(worker_finaloutput_root, new_file_name)
    worker_output_shp_subroot = os.path.join(worker_output_shp_root, new_file_name)


    try:
        shutil.rmtree(worker_finaloutput_subroot)
    except:
        print("director deletion failed")
        pass
    os.mkdir(worker_finaloutput_subroot)



    stich.stitch_shapefile(worker_output_shp_subroot,
                            worker_finaloutput_subroot, new_file_name,new_file_name)

    return "done Divide"

#############################################################
parser = argparse.ArgumentParser(
    description='Train Mask R-CNN to detect balloons.')

parser.add_argument("--image", required=False,
                    default='test_image_01.tif',
                    metavar="<command>",
                    help="Image name")

args = parser.parse_args()

image_name = args.image

print("start caculating wartermask")
cal_water_mask(image_name)
print("start tiling image")
tile_image(image_name)
print("start inferencing")
infer_image(image_name)
print("start stiching")
stich_shapefile(image_name)
process.process_shapefile(image_name)
