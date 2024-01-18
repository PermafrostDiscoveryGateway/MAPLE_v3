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
import multiprocessing
import numpy as np
import os
import sys
import shapefile
import tensorflow as tf

from collections import defaultdict
from mpl_config import MPL_Config, PolygonConfig
from skimage.measure import find_contours

class Predictor(multiprocessing.Process):
    def __init__(self, input_queue, gpu_id,
                          POLYGON_DIR,
                          weights_path,
                          output_shp_root,
                          x_resolution,
                          y_resolution,len_imgs,image_name):

        multiprocessing.Process.__init__(self)
        self.input_queue = input_queue
        self.gpu_id = gpu_id
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
        ROOT_DIR = MPL_Config.ROOT_DIR
        MY_WEIGHT_FILE = MPL_Config.WEIGHT_PATH

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "local_dir/datasets/logs")


        import model as modellib

        # --------------------------- Configurations ---------------------------
        # Set config
        config = PolygonConfig()

        output_shp_root = self.output_shp_root

        # --------------------------- Preferences ---------------------------
        # Device to load the neural network on.
        # Useful if you're training a model on the same
        # machine, in which case use CPU and leave the
        # GPU for training.
        # DEVICE = "/gpu:%s"%(self.gpu_id)  # /cpu:0 or /gpu:0
        DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
        os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(self.gpu_id)

        # Inspect the model in training or inference modes
        # values: 'inference' or 'training'
        # TODO: code for 'training' test mode not ready yet
        # Create model in inference mode
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                      config=config)

        # Load weights
        print("Loading weights ", MODEL_DIR)
        model.load_weights(MY_WEIGHT_FILE, by_name=True)
        output_shp_name_1 = output_shp_root.split('/')[-1]

        temp_name = "%s_%d.shp"%(output_shp_name_1, self.gpu_id)

        output_path_1 = os.path.join(output_shp_root, temp_name)
        w_final = shapefile.Writer(output_path_1)
        w_final.field('Class', 'C', size=5)
        count =0
        total = self.len_imgs
        # --------------------------- Workers ---------------------------

        dict_polygons = defaultdict(dict)
        while True:
            job_data = self.input_queue.get()
            count += 1

            if job_data is None:
                self.input_queue.task_done()
                print("Exiting Process %d" % self.gpu_id)
                break

            else:
                # get the upper left x y of the image
                ul_row_divided_img = job_data[0][2]
                ul_col_divided_img = job_data[0][3]
                tile_no = job_data[0][4]
                image = job_data[1]

                results = model.detect([image], verbose=False)

                r = results[0]

                if len(r['class_ids']):
                    count_p = 0

                    for id_masks in range(r['masks'].shape[2]):

                        # read the mask
                        mask = r['masks'][:, :, id_masks]
                        padded_mask = np.zeros(
                            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                        padded_mask[1:-1, 1:-1] = mask
                        class_id = r['class_ids'][id_masks]

                        try:
                            contours = find_contours(padded_mask, 0.5, 'high')[0] * np.array(
                                [[self.y_resolution, self.x_resolution]])
                            contours = contours + np.array([[float(ul_row_divided_img), float(ul_col_divided_img)]])
                            # swap two cols
                            contours.T[[0, 1]] = contours.T[[1, 0]]
                            # write shp file
                            w_final.poly([contours.tolist()])
                            w_final.record(Class=class_id)

                        except:
                            contours = []
                            pass

                        count_p += 1


                dict_polygons[int(tile_no)] = [r['masks'].shape[2]]

                if (MPL_Config.LOGGING):
                    print(f"## {count} of {total} ::: {len(r['class_ids'])}  $$$$ {r['class_ids']}")
                    sys.stdout.flush()

        import pickle
        worker_root = MPL_Config.WORKER_ROOT
        db_file_path = os.path.join(worker_root, "neighbors/%s_polydict_%d.pkl" % (self.image_name,self.gpu_id))
        dbfile = open(db_file_path, 'wb')
        pickle.dump(dict_polygons, dbfile)
        dbfile.close()
        w_final.close()

def inference_image(POLYGON_DIR,
                    weights_path,
                    output_shp_root,
                    file1,file2,image_name):

    f1 = h5py.File(file1, 'r')
    f2 = h5py.File(file2, 'r')

    values = f2.get('values')
    n1 = np.array(values)
    x_resolution = n1[0]
    y_resolution = n1[1]
    len_imgs = n1[2]

    # The number of GPU you want to use
    num_gpus = MPL_Config.NUM_GPUS_PER_CORE

    input_queue = multiprocessing.JoinableQueue()

    p_list = []

    for i in range(0, num_gpus):
        # set the i as the GPU device you want to use

        p = Predictor(input_queue, i,
                      POLYGON_DIR,
                      weights_path,
                      output_shp_root,
                      x_resolution,
                      y_resolution,
                      len_imgs,image_name)
        p_list.append(p)

    for p in p_list:
        p.start()


    for img in range(int(len_imgs)):
        image = f1.get(f"image_{img+1}")
        params = f2.get(f"param_{img+1}")
        img_stack = np.array(image)
        img_data = (np.array(params))

        job = [img_data,img_stack]

        input_queue.put(job)
    f1.close()
    f2.close()

    for i in range(num_gpus):
        input_queue.put(None)

    for p in p_list:
        p.join()
