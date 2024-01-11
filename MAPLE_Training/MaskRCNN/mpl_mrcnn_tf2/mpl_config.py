from config import Config
import os
class MPL_Config(object):
    ROOT_DIR = r'/mnt/data/Rajitha/MAPLE/Training_03'

    ## Do not change this section
    #-----------------------------------------------------------------
    INPUT_IMAGE_DIR = ROOT_DIR + r'/data/input_img'
    WORKER_ROOT =  ROOT_DIR + r'/data'
    TRAIN_ROOT = ROOT_DIR + r'/files/Training'
    #WEIGHT_PATH = ROOT_DIR + r'/mask_rcnn_ice_wedge_polygon_rajitha_001_0019.h5'
    WEIGHT_PATH = ROOT_DIR + r'/trained_weights_Dataset_179_9_33.h5'

    DEFAULT_DATASET_DIR = os.path.join(TRAIN_ROOT, "dataset_00_to_06")

    DEFAULT_LOGS_DIR = os.path.join(DEFAULT_DATASET_DIR, "logs")
    TRAINED_WEIGHT_PATH = DEFAULT_LOGS_DIR + r"/trained_weights_dataset_0.020000_100epoch_102_16_41_20220412T1641"


    #-----------------------------------------------------------------
    CROP_SIZE = 200

    LOGGING = True
    NUM_GPUS_PER_CORE = 4



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
    DETECTION_MIN_CONFIDENCE = 0.1

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100
    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.5
    RPN_NMS_THRESHOLD = 0.8
    IMAGE_MIN_DIM = 400
    IMAGE_MAX_DIM = 512


