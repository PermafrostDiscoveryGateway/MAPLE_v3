#!/usr/bin/python3
"""
MAPLE Workflow
(3) Inference using the trained Mask RCNN
Will load the tiled images and do the inference.

Project: Permafrost Discovery Gateway: Mapping Application for Arctic Permafrost Land Environment(MAPLE)
PI      : Chandi Witharana
Author  : Rajitha Udwalpola
"""

import h5py
import model as modellib
import multiprocessing
import numpy as np
import os
import pickle
import sys
import shapefile
import tensorflow as tf

from collections import defaultdict
from mpl_config import MPL_Config, PolygonConfig
from skimage.measure import find_contours


class Predictor(multiprocessing.Process):
    def __init__(
        self,
        config: MPL_Config,
        input_queue: multiprocessing.JoinableQueue,
        use_gpu: bool,
        process_counter: int,
        POLYGON_DIR,
        weights_path: str,
        output_shp_root: str,
        x_resolution: int,
        y_resolution: int,
        len_imgs: int,
        image_name: str,
    ):
        multiprocessing.Process.__init__(self)
        self.config = config
        self.input_queue = input_queue
        self.process_counter = process_counter
        self.use_gpu = use_gpu
        self.device = "/gpu:%d" % self.process_counter if self.use_gpu else "/cpu:0"
        self.len_imgs = len_imgs

        self.POLYGON_DIR = POLYGON_DIR
        self.weights_path = weights_path
        self.output_shp_root = output_shp_root

        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.image_name = image_name

    def run(self):
        # --------------------------- Preseting ---------------------------
        # Root directory of the project
        ROOT_DIR = self.config.ROOT_DIR
        MY_WEIGHT_FILE = self.config.WEIGHT_PATH

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "local_dir/datasets/logs")

        # --------------------------- Configurations ---------------------------
        # Set config
        model_config = PolygonConfig()

        output_shp_root = self.output_shp_root

        # --------------------------- Preferences ---------------------------
        # Device to load the neural network on.
        # Useful if you're training a model on the same
        # machine, in which case use CPU and leave the
        # GPU for training.
        if self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.process_counter)

        # Inspect the model in training or inference modes
        # values: 'inference' or 'training'
        # TODO: code for 'training' test mode not ready yet
        # Create model in inference mode
        with tf.device(self.device):
            model = modellib.MaskRCNN(
                mode="inference", model_dir=MODEL_DIR, config=model_config
            )

        # Load weights
        print("Loading weights ", MODEL_DIR)
        model.load_weights(MY_WEIGHT_FILE, by_name=True)
        output_shp_name_1 = output_shp_root.split("/")[-1]

        temp_name = "%s_%d.shp" % (output_shp_name_1, self.process_counter)

        output_path_1 = os.path.join(output_shp_root, temp_name)
        w_final = shapefile.Writer(output_path_1)
        w_final.field("Class", "C", size=5)
        count = 0
        total = self.len_imgs
        # --------------------------- Workers ---------------------------

        dict_polygons = defaultdict(dict)
        while not self.input_queue.empty():
            job_data = self.input_queue.get()
            count += 1

            # get the upper left x y of the image
            ul_row_divided_img = job_data[0][2]
            ul_col_divided_img = job_data[0][3]
            tile_no = job_data[0][4]
            image = job_data[1]

            results = model.detect([image], verbose=False)

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
                        ] * np.array([[self.y_resolution, self.x_resolution]])
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

            if self.config.LOGGING:
                print(
                    f"## {count} of {total} ::: {len(r['class_ids'])}  $$$$ {r['class_ids']}"
                )
                sys.stdout.flush()

        worker_root = self.config.WORKER_ROOT
        db_file_path = os.path.join(
            worker_root, "neighbors/%s_polydict_%d.pkl" % (self.image_name, self.process_counter)
        )
        dbfile = open(db_file_path, "wb")
        pickle.dump(dict_polygons, dbfile)
        dbfile.close()
        w_final.close()
        self.input_queue.task_done()
        print("Exiting Process %d" % self.process_counter)


def inference_image(
    config: MPL_Config,
    POLYGON_DIR: str,
    weights_path: str,
    output_shp_root: str,
    file1: str,
    file2: str,
    image_name: str,
):
    f1 = h5py.File(file1, "r")
    f2 = h5py.File(file2, "r")

    values = f2.get("values")
    n1 = np.array(values)
    x_resolution = n1[0]
    y_resolution = n1[1]
    len_imgs = n1[2]

    # The number of GPU you want to use
    num_gpus = config.NUM_GPUS_PER_CORE

    input_queue = multiprocessing.JoinableQueue()

    p_list = []

    # If the number of GPUs pero core is 0 create a single predictor that will
    # run on the CPU.
    if config.NUM_GPUS_PER_CORE == 0:
        p = Predictor(
            config,
            input_queue,
            False,
            0,
            POLYGON_DIR,
            weights_path,
            output_shp_root,
            x_resolution,
            y_resolution,
            len_imgs,
            image_name,
        )
        p_list.append(p)
    else:
        # If there are GPUs available, create a Predictor for each one to run
        # multiple inferences in parallel.
        for i in range(num_gpus):
            # set the i as the GPU device you want to use
            p = Predictor(
                config,
                input_queue,
                True,
                i,
                POLYGON_DIR,
                weights_path,
                output_shp_root,
                x_resolution,
                y_resolution,
                len_imgs,
                image_name,
            )
            p_list.append(p)

    # populate input queue with tasks for processes to consume.
    for img in range(int(len_imgs)):
        image = f1.get(f"image_{img+1}")
        params = f2.get(f"param_{img+1}")
        img_stack = np.array(image)
        img_data = np.array(params)

        job = [img_data, img_stack]

        input_queue.put(job)
    f1.close()
    f2.close()

    # start processes to start consuming jobs from the queue.
    for p in p_list:
        p.start()

    # join all processes to ensure proper clean up when all jobs are done.
    for p in p_list:
        p.join()
