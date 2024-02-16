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
import h5py
import mpl_clean_inference as inf_clean
import mpl_divideimg_234_water_new as divide
import mpl_infer_tiles_GPU_new as inference
import mpl_process_shapefile as process
import mpl_stitchshpfile_new as stich
import numpy as np
import os
import pickle
import shapefile
import sys
import shutil
import tensorflow as tf

from collections import defaultdict
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from flytekit import task, workflow, map_task, ImageSpec
from model import MaskRCNN
from mpl_config import MPL_Config, PolygonConfig
from osgeo import gdal
from pathlib import Path
from skimage import filters, io
from skimage.measure import find_contours
from skimage.morphology import disk
from typing import List, Tuple

# work tag
WORKTAG = 1
DIETAG = 0

MAPLE_IMAGE_SPEC = ImageSpec(
    name="maple",
    base_image="ghcr.io/permafrostdiscoverygateway/maple_v3:latest",
    packages=["flytekit"],
    registry="localhost:30000",
)


@dataclass_json
@dataclass
class TiledImage:
    image_path: str
    image_tiles: List[Tuple]
    x_resolution: float
    y_resolution: float


@task(container_image=MAPLE_IMAGE_SPEC)
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


@task(container_image=MAPLE_IMAGE_SPEC)
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


@task(container_image=MAPLE_IMAGE_SPEC)
def fetch_tiled_image(config: MPL_Config, input_img_name: str) -> TiledImage:
    # Output represents a list of image tiles where each tile is represented by
    # the image data and the image stack where the image data has information
    # about where the tile belongs in the image and the image stack are the pixel
    # values.
    print("3. fetching tiled images")
    sys.path.append(config.ROOT_DIR)

    # worker roots
    worker_divided_img_root = config.DIVIDED_IMAGE_DIR

    # Create subfolder for each image
    new_file_name = input_img_name.split(".tif")[0]
    worker_divided_img_subroot = os.path.join(worker_divided_img_root, new_file_name)

    print(worker_divided_img_subroot)

    file1 = os.path.join(worker_divided_img_subroot, "image_data.h5")
    file2 = os.path.join(worker_divided_img_subroot, "image_param.h5")
    f1 = h5py.File(file1, "r")
    f2 = h5py.File(file2, "r")

    values = f2.get("values")
    n1 = np.array(values)
    x_resolution = n1[0]
    y_resolution = n1[1]
    len_imgs = n1[2]

    image_tiles = []
    for img in range(int(len_imgs)):
        image = f1.get(f"image_{img+1}")
        params = f2.get(f"param_{img+1}")
        img_stack = np.array(image)
        img_data = np.array(params)
        image_tiles.append((img_stack.tolist(), img_data.tolist()))

    return TiledImage(input_img_name, image_tiles, x_resolution, y_resolution)


@task(container_image=MAPLE_IMAGE_SPEC)
def infer_tiled_image(config: MPL_Config, tiled_image: TiledImage):
    # This function will serve the same purpose of the Predictor run method. But
    # instead of working on individual tiles it will do all the computation for
    # all the tiles of an image.
    print("4. Inferencing tiled images")
    sys.path.append(config.ROOT_DIR)

    # Configure output directory.
    image_name = tiled_image.image_path.split(".tif")[0]
    output_shp_path = os.path.join(config.OUTPUT_SHP_DIR, image_name)
    try:
        shutil.rmtree(output_shp_path)
    except:
        print("directory deletion failed")
        pass

    output_shape_file_name = "%s.shp" % image_name
    output_shp_path = os.path.join(output_shp_path, output_shape_file_name)
    w_final = shapefile.Writer(output_shp_path)
    w_final.field("Class", "C", size=5)

    # Figure out how to determine device.
    model_config = PolygonConfig()
    device = "/cpu:0"
    with tf.device(device):
        model = MaskRCNN(
            mode="inference", model_dir="", config=model_config
        )

    # Load weights.
    print("Loading weights ", config.WEIGHT_PATH)
    model.keras_model.load_weights(config.WEIGHT_PATH, by_name=True)

    # Run inference on the image tiles.
    dict_polygons = defaultdict(dict)
    for count, tile in enumerate(tiled_image.image_tiles):
        image_data = np.array(tile[1])
        image_stack = np.array(tile[0])

        # get the upper left x y of the image
        ul_row_divided_img = image_data[2]
        ul_col_divided_img = image_data[3]
        tile_no = image_data[4]

        results = model.detect([image_stack], verbose=False)

        r = results[0]

        if len(r["class_ids"]):
            count_p = 0

            for id_masks in range(r["masks"].shape[2]):
                # read the mask
                mask = r["masks"][:, :, id_masks]
                padded_mask = np.zeros(
                    (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8
                )
                padded_mask[1:-1, 1:-1] = mask
                class_id = r["class_ids"][id_masks]

                try:
                    contours = find_contours(padded_mask, 0.5, "high")[
                        0
                    ] * np.array([[tiled_image.y_resolution, tiled_image.x_resolution]])
                    contours = contours + np.array(
                        [[float(ul_row_divided_img), float(ul_col_divided_img)]]
                    )
                    # swap two cols
                    contours.T[[0, 1]] = contours.T[[1, 0]]
                    # write shp file
                    w_final.poly([contours.tolist()])
                    w_final.record(Class=class_id)

                except:
                    contours = []
                    pass

                count_p += 1

        dict_polygons[int(tile_no)] = [r["masks"].shape[2]]

        if config.LOGGING:
            print(
                f"## {count} of {len(tiled_image.image_tiles)} ::: {len(r['class_ids'])}  $$$$ {r['class_ids']}"
            )
            sys.stdout.flush()

    worker_root = config.WORKER_ROOT
    db_file_path = os.path.join(
        worker_root,
        "neighbors/%s_polydict.pkl" % (image_name),
    )
    dbfile = open(db_file_path, "wb")
    pickle.dump(dict_polygons, dbfile)
    dbfile.close()
    w_final.close()


@task(container_image=MAPLE_IMAGE_SPEC)
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


@task(container_image=MAPLE_IMAGE_SPEC)
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
    print("4. start stitching")
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


@task(container_image=MAPLE_IMAGE_SPEC)
def clean_inference(config: MPL_Config, data_boundray_file: str):
    print("5. start cleaning")
    inf_clean.clean_inference_shapes(
        config.CLEAN_DATA_DIR,
        config.PROJECTED_SHP_DIR,
        data_boundray_file,
    )


@task(container_image=MAPLE_IMAGE_SPEC)
def setup(root_dir: str, weight_file: str, gpus_per_core: int) -> MPL_Config:
    # disable TF eager execution
    tf.compat.v1.disable_eager_execution()
    return MPL_Config(root_dir, weight_file, num_gpus_per_core=gpus_per_core)


@task(container_image=MAPLE_IMAGE_SPEC)
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

@workflow
def updated_maple_workflow(
    root_dir: str = "",
    weight_file: str = "hyp_best_train_weights_final.h5",
    gpus_per_core: int = 0,
):
    config = setup(
        root_dir=root_dir, weight_file=weight_file, gpus_per_core=gpus_per_core
    )

    image_files = get_image_files(config=config)

    partial_t0 = functools.partial(cal_water_mask, config=config)
    calculate_water_masks = map_task(partial_t0)(input_img_name=image_files)

    partial_t1 = functools.partial(tile_image, config=config)
    tile_images = map_task(partial_t1)(input_img_name=image_files)

    partial_t2 = functools.partial(fetch_tiled_image, config=config)
    tiled_images = map_task(partial_t2)(input_img_name=image_files)

    partial_t3 = functools.partial(infer_tiled_image, config=config)
    infer_images = map_task(partial_t3)(tiled_image=tiled_images)

    partial_t4 = functools.partial(stich_shapefile, config=config)
    stitch_shapefiles = map_task(partial_t4)(input_img_name=image_files)

    # clean inference is set up as a bulk operator on the whole directory. Should
    # probably refactor it to clean the inference of individual images to follow
    # the same patterns as the tasks above.
    clean_shapefiles = clean_inference(
        config=config, data_boundray_file="./data/input_bound/sample2_out_boundry.shp"
    )

    # Chain the tasks that don't have direct data dependencies so that their order
    # of operations is clear in the execution task.
    # TODO: Ideally we would pass around the files of intereset between tasks
    # so that they do have a data dependency between them to better follow
    # the flyte paradigm.
    calculate_water_masks >> tile_images
    tile_images >> tiled_images
    infer_images >> stitch_shapefiles
    stitch_shapefiles >> clean_shapefiles
