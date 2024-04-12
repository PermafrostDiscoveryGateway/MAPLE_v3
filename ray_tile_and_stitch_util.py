#!/usr/bin/env python3
"""
MAPLE Workflow
Functions for image tiling and stitching.
Project: Permafrost Discovery Gateway: Mapping Application for Arctic Permafrost Land Environment(MAPLE)
PI      : Chandi Witharana
"""
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List
import random

import cv2
from osgeo import gdal, ogr
from scipy.spatial import distance
from shapely.geometry import Polygon
import numpy as np
import pandas as pd

import gdal_virtual_file_system as gdal_vfs
from mpl_config import MPL_Config


@dataclass
class ImageMetadata:
    len_x_list: int
    len_y_list: int
    x_resolution: float
    y_resolution: float


@dataclass
class ImageTileMetadata:
    upper_left_row: float
    upper_left_col: float
    # Indexes that are used to reconstruct image after tiling.
    id_i: int
    id_j: int
    tile_num: int


@dataclass
class ImageTile:
    tile_values: np.array
    tile_metadata: ImageTileMetadata


@dataclass
class ShapefileResult:
    polygons: np.array
    class_id: int


@dataclass
class ShapefileResults:
    shapefile_results: List[ShapefileResult]


def tile_image(row: Dict[str, Any], config: MPL_Config) -> List[Dict[str, Any]]:
    """
    Tile the image into multiple pre-deifined sized parts so that the processing can be done on smaller parts due to
    processing limitations

    Parameters
    ----------
    row : Row in ray dataset corresponding to the information for a single image.
    config : Contains static configuration information regarding the workflow.
    """

    # Create virtual file system file for image to use GDAL's file apis.
    vfs = gdal_vfs.GDALVirtualFileSystem(
        file_path=row["path"], file_bytes=row["bytes"])
    virtual_image_file_path = vfs.create_virtual_file()
    input_image_gtif = gdal.Open(virtual_image_file_path)
    mask = row["mask"]

    # convert the original image into the geo cordinates for further processing using gdal
    # https://gdal.org/tutorials/geotransforms_tut.html
    # GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
    # GT(1) w-e pixel resolution / pixel width.
    # GT(2) row rotation (typically zero).
    # GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
    # GT(4) column rotation (typically zero).
    # GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).

    # ---------------------- crop image from the water mask----------------------
    # dot product of the mask and the orignal data before breaking it for processing
    # Also band 2 3 and 4 are taken because the 4 bands cannot be processed by the NN learingin algo
    # Need to make sure that the training bands are the same as the bands used for inferencing.
    #
    final_array_2 = input_image_gtif.GetRasterBand(2).ReadAsArray()
    final_array_3 = input_image_gtif.GetRasterBand(3).ReadAsArray()
    final_array_4 = input_image_gtif.GetRasterBand(4).ReadAsArray()

    final_array_2 = np.multiply(final_array_2, mask)
    final_array_3 = np.multiply(final_array_3, mask)
    final_array_4 = np.multiply(final_array_4, mask)

    # ulx, uly is the upper left corner
    ulx, x_resolution, _, uly, _, y_resolution = input_image_gtif.GetGeoTransform()

    # ---------------------- Divide image (tile) ----------------------
    overlap_rate = 0.2
    block_size = config.CROP_SIZE
    ysize = input_image_gtif.RasterYSize
    xsize = input_image_gtif.RasterXSize

    # Close the file.
    input_image_gtif = None
    vfs.close_virtual_file()

    tile_count = 0

    y_list = range(0, ysize, int(block_size * (1 - overlap_rate)))
    x_list = range(0, xsize, int(block_size * (1 - overlap_rate)))

    # ---------------------- Find each Upper left (x,y) for each divided images ----------------------
    #  ***-----------------
    #  ***
    #  ***
    #  |
    #  |
    #
    tiles = []
    for id_i, i in enumerate(y_list):
        # don't want moving window to be larger than row size of input raster
        if i + block_size < ysize:
            rows = block_size
        else:
            rows = ysize - i

        # read col
        for id_j, j in enumerate(x_list):
            if j + block_size < xsize:
                cols = block_size
            else:
                cols = xsize - j
            # get block out of the whole raster
            # todo check the array values is similar as ReadAsArray()
            band_1_array = final_array_4[i: i + rows, j: j + cols]
            band_2_array = final_array_2[i: i + rows, j: j + cols]
            band_3_array = final_array_3[i: i + rows, j: j + cols]

            # filter out black image
            if (
                band_3_array[0, 0] == 0
                and band_3_array[0, -1] == 0
                and band_3_array[-1, 0] == 0
                and band_3_array[-1, -1] == 0
            ):
                continue

            # stack three bands into one array
            img = np.stack((band_1_array, band_2_array, band_3_array), axis=2)
            cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)
            B, G, R = cv2.split(img)
            out_B = cv2.equalizeHist(B)
            out_R = cv2.equalizeHist(R)
            out_G = cv2.equalizeHist(G)
            final_image = np.array(cv2.merge((out_B, out_G, out_R)))

            # Upper left (x,y) for each images
            ul_row_divided_img = uly + i * y_resolution
            ul_col_divided_img = ulx + j * x_resolution

            tile_metadata = ImageTileMetadata(
                upper_left_row=ul_row_divided_img, upper_left_col=ul_col_divided_img, tile_num=tile_count, id_i=id_i, id_j=id_j)
            image_tile = ImageTile(
                tile_values=final_image, tile_metadata=tile_metadata)
            tiles.append(image_tile)
            tile_count += 1

    # --------------- Store all the title as an object file
    image_metadata = ImageMetadata(
        len_x_list=len(x_list), len_y_list=len(y_list), x_resolution=x_resolution, y_resolution=y_resolution)
    row["image_metadata"] = image_metadata
    new_rows = []
    tile_count = 0
    for image_tile in tiles:
        new_row = copy.deepcopy(row)
        new_row["image_tile"] = image_tile
        new_row["tile_num"] = tile_count
        new_rows.append(new_row)
        tile_count += 1
    return new_rows


def stitch_shapefile(group: pd.DataFrame):
    """
    Create a shapefile for each image.
    Note that normally ray rows are dictionaries but this is a group because we've called
    grouped by. When ray does a groupby and the rows can't be represented in numpy arrays
    it uses pandas dataframe.

    Parameters
    ----------
    group: a pandas dataframe that has all of the shapefile information for each tile in the
            image.
    """
    image_shapefile_results = []
    temp_polygon_dict = defaultdict(dict)
    dict_ij = defaultdict(dict)
    sorted_group = group.sort_values(by="tile_num")
    for index, row in sorted_group.iterrows():
        image_shapefile_results.extend(
            row["tile_shapefile_results"].shapefile_results)
        image_tile = row["image_tile"]
        tile_num = row["tile_num"]
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
    del_index_list = list()
    for close_possible in close_list:
        close_possible.sort()
        random_id = close_possible[0]
        # random_id = random.choice(close_possible)
        close_possible.remove(random_id)
        del_index_list.extend(close_possible)

    print("Features before: ", len(image_shapefile_results))

    # Note that we sort the indices in reversed order to ensure that the shift of indices
    # induced by the deletion of elements at lower indices won’t invalidate the index
    # specifications of elements at larger indices
    for index in sorted(del_index_list, reverse=True):
        del image_shapefile_results[index]

    print("Features after: ", len(image_shapefile_results))

    first_row["image_shapefile_results"] = ShapefileResults(
        image_shapefile_results)
    return first_row
