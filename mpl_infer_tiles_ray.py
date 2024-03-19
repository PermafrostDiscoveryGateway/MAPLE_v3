#!/usr/bin/python3
"""
MAPLE Workflow
(3) Inference using the trained Mask RCNN
Will load the tiled images and do the inference.

Project: Permafrost Discovery Gateway: Mapping Application for Arctic Permafrost Land Environment(MAPLE)
PI      : Chandi Witharana
Author  : Rajitha Udwalpola
"""

import model as modellib
import numpy as np
import ray
import tensorflow as tf

from dataclasses import dataclass
from mpl_config import MPL_Config, PolygonConfig
from skimage.measure import find_contours
from typing import Any, Dict


@dataclass
class ShapefileResult:
    polygons: np.array
    class_id: str


class MaskRCNNPredictor:
    def __init__(
        self,
        config: MPL_Config
    ):
        self.config = config
        # Used to identify a specific predictor when mulitple predictors are
        # created to run inference in parallel. The counter is also used to
        # know which GPU to use when multiple are available.
        self.process_counter = 1  # TODO need to fix this process_counter
        self.use_gpu = config.NUM_GPUS_PER_CORE > 0
        self.device = "/gpu:%d" % self.process_counter if self.use_gpu else "/cpu:0"

        with tf.device(self.device):
            self.model = modellib.MaskRCNN(
                mode="inference", model_dir=self.config.MODEL_DIR, config=PolygonConfig()
            )
        self.model.keras_model.load_weights(
            self.config.WEIGHT_PATH, by_name=True)

    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:

        # get the upper left x y of the image
        image_tile = row["image_tile"]
        ul_row_divided_img = image_tile.tile_metadata.upper_left_row
        ul_col_divided_img = image_tile.tile_metadata.upper_left_col
        image_tile_values = image_tile.tile_values
        image_metadata = row["image_metadata"]
        x_resolution = image_metadata.x_resolution
        y_resolution = image_metadata.y_resolution

        results = self.model.detect([image_tile_values], verbose=False)

        r = results[0]
        shapefile_results = []
        if len(r["class_ids"]):
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
                    ] * np.array([[y_resolution, x_resolution]])
                    contours = contours + np.array(
                        [[float(ul_row_divided_img), float(ul_col_divided_img)]]
                    )
                    # swap two cols
                    contours.T[[0, 1]] = contours.T[[1, 0]]
                    shapefile_results.append(ShapefileResult(
                        polygons=contours, class_id=class_id))
                except:
                    contours = []
                    pass
        row["shapefile_results"] = shapefile_results
        return row
