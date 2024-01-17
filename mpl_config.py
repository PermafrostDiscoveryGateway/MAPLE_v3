"""
MAPLE Workflow
This is the configuration file for maple workflow
(1) modifies required Mask R-CNN configurations
(2) indicates the local environment to execute the workflow i.e where files are located

Project: Permafrost Discovery Gateway: Mapping Application for Arctic Permafrost Land Environment(MAPLE)
PI      : Chandi Witharana
Author  : Rajitha Udwalpola
"""

from config import Config
class MPL_Config(object):
    # ROOT_DIR where the code will look for all the input/output and generated files
    # Can change this to the location where you want to run the code
    ROOT_DIR = r'/usr/local/google/home/djfernandez/Desktop/MAPLE_v3'

    ## Do not change this section
    # Code depends on the relative locations indicated so should not change
    # Code expects some of the locations to be available when executing.
    #-----------------------------------------------------------------
    INPUT_IMAGE_DIR = ROOT_DIR + r'/data/input_img_local'
    DIVIDED_IMAGE_DIR = ROOT_DIR + r'/data/divided_img'
    OUTPUT_SHP_DIR = ROOT_DIR + r'/data/output_shp'
    FINAL_SHP_DIR = ROOT_DIR + r'/data/final_shp'
    PROJECTED_SHP_DIR = ROOT_DIR + r'/data/projected_shp'
    WATER_MASK_DIR = ROOT_DIR + r'/data/water_mask'
    TEMP_W_IMG_DIR = ROOT_DIR + r'/data/water_mask/temp'
    OUTPUT_IMAGE_DIR = ROOT_DIR + r'/data/output_img'
    WORKER_ROOT =  ROOT_DIR + r'/data'

    # ADDED to include inference cleaning post-processing
    CLEAN_DATA_DIR = ROOT_DIR + r'/data/cln_data'
    INPUT_DATA_BOUNDARY_FILE_PATH = ROOT_DIR + r'/data/input_bound'

    #-------------------------------------------------------------------
    # Name of the weight file used for the inference
    weight_name = r'hyp_best_train_weights_final.h5'

    #-----------------------------------------------------------------
    # Location of the weight file used for the inference
    WEIGHT_PATH = ROOT_DIR + r"/" + weight_name
    #-----------------------------------------------------------------
    CROP_SIZE = 200

    LOGGING = True
    NUM_GPUS_PER_CORE = 1
#


class PolygonConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ice_wedge_polygon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1  # Background + highcenter + lowcenter

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 340

    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.3

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 200
    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3
    RPN_NMS_THRESHOLD = 0.8
    IMAGE_MIN_DIM = 200
    IMAGE_MAX_DIM = 256

