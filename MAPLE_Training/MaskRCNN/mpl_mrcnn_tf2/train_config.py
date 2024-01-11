import os
import  datetime
import numpy as np

class Train_Config(object):
    #ROOT_DIR = r'/mnt/data/Rajitha/MAPLE/Training_05'
    ROOT_DIR = r'/media/maple_backup2/newlogs'

    TRAIN_ROOT = ROOT_DIR + r'/files/Training'
    ## Do not change this section
    #-----------------------------------------------------------------
    DEFAULT_LOGS_DIR = os.path.join(TRAIN_ROOT, "dataset_00_to_06/logs")

    DEFAULT_DATASET_DIR = os.path.join(TRAIN_ROOT, "dataset_00_to_06")

    now = datetime.datetime.now()
    NO_OF_EPOCH = 300
    DEFAULT_OUTPUT_NAME = 'trained_weights_Dataset_%s_%s_%s_' % (now.strftime('%j'), now.hour, now.minute)
    DEFAULT_OUTPUT_WEIGHT = '%s.h5'%(DEFAULT_OUTPUT_NAME)

from config import Config

class MPL_Config(object):
    ROOT_DIR = r'/media/maple_backup2/newlogs'
    #ROOT_DIR = r'/mnt/data/Rajitha/MAPLE/Training_03'

    ## Do not change this section
    #-----------------------------------------------------------------
    INPUT_IMAGE_DIR = ROOT_DIR + r'/data/input_img'
    WORKER_ROOT =  ROOT_DIR + r'/data'
    #WEIGHT_PATH = ROOT_DIR + r'/mask_rcnn_ice_wedge_polygon_rajitha_001_0019.h5'
    WEIGHT_PATH = ROOT_DIR + r'/rajitha_comb_poly_0.h5'
    #-----------------------------------------------------------------
    CROP_SIZE = 200

    LOGGING = True
    NUM_GPUS_PER_CORE = 1


class PolygonConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """

    # Give the configuration a recognizable name


    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1  # Background + highcenter + lowcenter

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # VALIDATION_STEPS = 50
    WEIGHT_DECAY = 0.0001
    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.50

    # Learning Rate
    #LEARNING_RATE = 0.02 # amal:orignal LR
    LEARNING_RATE = 0.02

    ###############################################################

    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    IMAGE_MIN_DIM = 400
    IMAGE_MAX_DIM = 512
    now = datetime.datetime.now()
    NAME = DEFAULT_OUTPUT_NAME = 'trained_weights_Dataset_%f_%depoch_%s_%s_%s_' % (LEARNING_RATE,Train_Config.NO_OF_EPOCH, now.strftime('%j'), now.hour, now.minute)

    #IMAGE_CHANNEL_COUNT = 4
    IMAGE_CHANNEL_COUNT = 3

    MEAN_PIXEL = np.array([123.7, 123.7,116.8, 103.9])

    #Frontera Execution
    GPU_COUNT = 4
