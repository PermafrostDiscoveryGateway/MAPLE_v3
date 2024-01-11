"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>

    tensorboard --logdir newlogs

    runfile('/mnt/data/Model_Training/Maple_Train.py','train --dataset=/mnt/data/Model_Training/Datasets_Satellite/Combined_v1_v2_v3 --weights=/mnt/data/Model_Training/Weixing_mask_rcnn_ice_wedge_polygon_0008.h5 --output=/mnt/data/Model_Training/logs')
"""


import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
#from imgaug import augmenters as iaa
#import tensorflow-gpu as tf
#import matplotlib.pyplot as plt
# Root directory of the project
#ROOT_DIR = r"/media/maple_backup2/newlogs"
ROOT_DIR = r"/scratch1/09208/asperera/maple_run/MAPLE_train/Training_03_tf2/"
#ROOT_DIR = r"/mnt/data/Rajitha/MAPLE/Training_05"

if ROOT_DIR.endswith("samples/polygon"):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

#Original mask RCNN config.py
   
# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "newlogs")

############################################################
#  Configurations
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################
#DEVICE = "/gpu:0"

class PolygonConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ice_wedge_polygon_amal_12_28"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    
    GPU_COUNT = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1  # Background + highcenter + lowcenter

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 3
    #STEPS_PER_EPOCH = 10 # For testing

    #VALIDATION_STEPS = 50
    WEIGHT_DECAY = 0.0001
    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.20
    
    # Learning Rate
    #LEARNING_RATE = 0.005
    LEARNING_RATE = 0.02
    
    ###############################################################
    
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    IMAGE_MIN_DIM = 400
    IMAGE_MAX_DIM = 512

############################################################
#  Dataset
############################################################

class PolygonDataset(utils.Dataset):

    def load_polygon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("polygon", 1, "lowcenter")
        self.add_class("polygon", 2, "highcenter")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)


        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            #print(a['regions'])
            #polygons = [(r['shape_attributes'],r['region_attributes']['object_name']) for r in a['regions']]

            if type(a['regions']) is dict:
                polygons = [(r['shape_attributes'],r['region_attributes']['object_name'])for r in a['regions'].values()]
            else:
                polygons = [(r['shape_attributes'],r['region_attributes']['object_name']) for r in a['regions']]


                #polygons = [r['shape_attributes'] for r in a['regions']]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "polygon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        polygons = image_info["polygons"]
        if image_info["source"] != "polygon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, (p, _) in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        class_ids = np.array([self.class_names.index(s) for (_,s) in polygons])
        return mask, class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "polygon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model,output_model_path):
    """Train the model."""
    
    # Training dataset.
    dataset_train = PolygonDataset()
    dataset_train.load_polygon(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PolygonDataset()
    dataset_val.load_polygon(args.dataset, "val")
    dataset_val.prepare()


    print("Training network heads")
    model.set_log_dir(DEFAULT_LOGS_DIR)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs= 2,
                layers='heads'
                )

    # save the best model
    print("Output weigts path:",output_model_path)
    try:
        model.keras_model.save_weights(output_model_path+'/final_weights.h5')
    except:
        print("unable to write final model weights")


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--output', required=True,
                        metavar="polygon_model.h5",
                        help='model for output')
    args = parser.parse_args()

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PolygonConfig()
    else:
        class InferenceConfig(PolygonConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        #with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
            model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
    print("COCO Weight Path:",COCO_WEIGHTS_PATH)
    # Load weights
    print("Loading weights ",weights_path)
    print("##################################################################")
    #print(tf.test.gpu_device_name())
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
    
    # Train or evaluate
    output_path = args.output
    if args.command == "train":
        print("$$$$$$$$$$$$$$$$$$")
        print("finel ouptut file path: ",output_path)
        train(model,output_path)
    #elif args.command == "splash":
    #    detect_and_color_splash(model, image_path=args.image,
    #                            video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
