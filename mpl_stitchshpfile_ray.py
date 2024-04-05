#!/usr/bin/env python3
"""
MAPLE Workflow
(4) Stich back to the original image dims from the tiles created by the inference process
Project: Permafrost Discovery Gateway: Mapping Application for Arctic Permafrost Land Environment(MAPLE)
PI      : Chandi Witharana
Author  : Rajitha Udwalpola
"""

import glob
import numpy as np
import os
import pandas as pd
import pickle
import random
import ray
import shapefile

from collections import defaultdict
from dataclasses import dataclass
from mpl_config import MPL_Config
from osgeo import ogr
from scipy.spatial import distance
from shapely.geometry import Polygon
from typing import Any, Dict, List

@dataclass
class ShapefileResult:
    polygons: np.array
    class_id: int

@dataclass
class ShapefileResults:
    shapefile_results: List[ShapefileResult]

@dataclass
class ImageMetadata:
    len_x_list: int
    len_y_list: int
    x_resolution: float
    y_resolution: float

def stitch_shapefile(group: pd.DataFrame):
    image_shapefile_results = []
    temp_polygon_dict = defaultdict(dict)
    dict_ij = defaultdict(dict)
    for index, row in group.iterrows():
        image_shapefile_results.extend(row["tile_shapefile_results"].shapefile_results)
        image_tile = row["image_tile"]
        tile_num = image_tile.tile_metadata.tile_num
        temp_polygon_dict[tile_num] = row["num_polygons_in_tile"]
        id_i = image_tile.tile_metadata.id_i
        id_j = image_tile.tile_metadata.id_j
        dict_ij[id_i][id_j] = tile_num

    polygon_dict = defaultdict(dict)
    poly_count = 0
    for k, v in temp_polygon_dict.items():
        polygon_dict[k] = [poly_count, poly_count + v]
        poly_count += v

    first_row = group.head(1)
    image_data_from_arbitrary_row = first_row["image_metadata"][0]
    size_i, size_j = image_data_from_arbitrary_row.len_y_list, image_data_from_arbitrary_row.len_x_list

    # create a list to store those centroid point
    centroid_list = list()
    # create a count number for final checking
    for shapefile_result in image_shapefile_results:
        # create a polygon in shapely
        ref_polygon = Polygon(shapefile_result.polygons)
        # parse wkt return
        geom = ogr.CreateGeometryFromWkt(ref_polygon.centroid.wkt)
        centroid_x, centroid_y = geom.GetPoint(0)[0], geom.GetPoint(0)[1]
        centroid_list.append([centroid_x, centroid_y])

    close_list = list()
    print("Total number of polygons: ", len(centroid_list))
    tile_blocksize = 4

    for id_i in range(0, size_i, 3):
        if id_i + tile_blocksize < size_i:
            n_i = tile_blocksize
        else:
            n_i = size_i - id_i

        for id_j in range(0, size_j, 3):
            if id_j + tile_blocksize < size_j:
                n_j = tile_blocksize
            else:
                n_j = size_j - id_j

            # add to the neighbor list.
            centroid_neighbors = []
            poly_neighbors = []

            for ii in range(n_i):
                for jj in range(n_j):
                    if (ii + id_i) in dict_ij.keys():
                        if (jj + id_j) in dict_ij[(ii + id_i)].keys():
                            n = dict_ij[ii + id_i][jj + id_j]
                            poly_range = polygon_dict[n]
                            poly_list = [*range(poly_range[0], poly_range[1])]
                            poly_neighbors.extend(poly_list)
                            centroid_neighbors.extend(
                                centroid_list[poly_range[0]: poly_range[1]]
                            )

            if len(centroid_neighbors) == 0:
                continue
            dst_array = distance.cdist(
                centroid_neighbors, centroid_neighbors, "euclidean"
            )

            # filter out close objects
            filter_object_array = np.argwhere(
                (dst_array < 10) & (dst_array > 0))

            filter_object_array[:, 0] = [
                poly_neighbors[i] for i in filter_object_array[:, 0]
            ]
            filter_object_array[:, 1] = [
                poly_neighbors[i] for i in filter_object_array[:, 1]
            ]

            if filter_object_array.shape[0] != 0:
                for i in filter_object_array:
                    close_list.append(i.tolist())
            else:
                continue

    # remove duplicated index
    close_list = set(frozenset(sublist) for sublist in close_list)
    close_list = [list(x) for x in close_list]

    # --------------- looking for connected components in a graph ---------------
    def connected_components(lists):
        neighbors = defaultdict(set)
        seen = set()
        for each in lists:
            for item in each:
                neighbors[item].update(each)

        def component(node, neighbors=neighbors, seen=seen, see=seen.add):
            nodes = set([node])
            next_node = nodes.pop
            while nodes:
                node = next_node()
                see(node)
                nodes |= neighbors[node] - seen
                yield node

        for node in neighbors:
            if node not in seen:
                yield sorted(component(node))

    close_list = list(connected_components(close_list))

    # --------------- create a new shp file to store ---------------
    # randomly pick one of many duplications
    chosen_polygon_indexes = []
    for close_possible in close_list:
        chosen_polygon_indexes.append(random.choice(close_possible))

    image_shapefile_results_without_dups = [
        image_shapefile_results[index] for index in chosen_polygon_indexes]

    first_row["image_shapefile_results"] = ShapefileResults(image_shapefile_results_without_dups)
    return first_row
