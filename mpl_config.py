"""
MAPLE Workflow
This is the configuration file for maple workflow
(1) modifies required Mask R-CNN configurations
(2) indicates the local environment to execute the workflow i.e where files are located

Project: Permafrost Discovery Gateway: Mapping Application for Arctic Permafrost Land Environment(MAPLE)
PI      : Chandi Witharana
Author  : Rajitha Udwalpola
"""

import os

import gcsfs
import google.auth

from config import Config


class MPL_Config(object):
    """Initializes MPL_Config object.

    Arguments:
    root_dir -- Path to where the workflow will be ran from.
    weight_file -- Path to the Mask-RCNN model weights. Should be relative to the
    root directory.
    logging -- Whether to enable logging messages when running the workflow.
    crop_size -- Used to determine tile size when splitting the image up.
    num_spus_per_core -- Number of GPUs available per node.
    """

    def __init__(
        self,
        root_dir="",
        weight_file="hyp_best_train_weights_final.h5",
        logging=True,
        crop_size=200,
        num_gpus_per_core=1,
    ):
        # Do not change this section
        # Code depends on the relative locations indicated so should not change
        # Code expects some of the locations to be available when executing.
        # -----------------------------------------------------------------
        self.ROOT_DIR = root_dir if root_dir else os.getcwd()
        self.INPUT_IMAGE_DIR = self.ROOT_DIR + r"/data/input_img"
        self.DIVIDED_IMAGE_DIR = self.ROOT_DIR + r"/data/divided_img"
        self.OUTPUT_SHP_DIR = self.ROOT_DIR + r"/data/output_shp"
        self.FINAL_SHP_DIR = self.ROOT_DIR + r"/data/final_shp"
        self.PROJECTED_SHP_DIR = self.ROOT_DIR + r"/data/projected_shp"
        self.WATER_MASK_DIR = self.ROOT_DIR + r"/data/water_mask"
        self.TEMP_W_IMG_DIR = self.ROOT_DIR + r"/data/water_mask/temp"
        self.OUTPUT_IMAGE_DIR = self.ROOT_DIR + r"/data/output_img"
        self.WORKER_ROOT = self.ROOT_DIR + r"/data"
        self.MODEL_DIR = self.ROOT_DIR + r"/local_dir/datasets/logs"
        self.RAY_OUTPUT_SHAPEFILES_DIR = self.ROOT_DIR + r"/data/ray_output_shapefiles"

        self.GCP_FILESYSTEM = None
        if (self.ROOT_DIR.startswith(
            "gcs://") or self.ROOT_DIR.startswith("gs://")):
            creds, _ = google.auth.load_credentials_from_file(
                "/usr/local/google/home/kaylahardie/.config/gcloud/application_default_credentials.json", scopes=["https://www.googleapis.com/auth/cloud-platform"])
            self.GCP_FILESYSTEM = gcsfs.GCSFileSystem(
                project="pdg-project-406720", token=creds)

        # ADDED to include inference cleaning post-processing
        self.CLEAN_DATA_DIR = self.ROOT_DIR + r"/data/cln_data"
        self.INPUT_DATA_BOUNDARY_FILE_PATH = self.ROOT_DIR + r"/data/input_bound"

        # -----------------------------------------------------------------
        # Location of the weight file used for the inference
        self.WEIGHT_PATH = self.ROOT_DIR + r"/" + weight_file
        # -----------------------------------------------------------------
        self.CROP_SIZE = crop_size

        self.LOGGING = logging
        self.NUM_GPUS_PER_CORE = num_gpus_per_core


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
