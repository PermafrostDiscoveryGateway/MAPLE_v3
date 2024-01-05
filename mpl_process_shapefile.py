#!/usr/bin/env python3
"""
MAPLE Workflow
(4) Post process the inferences into a standard shape file that can be processed further (Final Product)
Will create .shp / .dbf / .shx and .prj files in the data/projected_shp directory. Code reads the final_shp
created by the inferencing step

Project: Permafrost Discovery Gateway: Mapping Application for Arctic Permafrost Land Environment(MAPLE)
PI      : Chandi Witharana
Author  : Rajitha Udwalpola / Amal Perera
"""

import os
import shutil
import shapefile
from shapely.geometry import Polygon, Point
from mpl_config import MPL_Config
from osgeo import gdal, osr


import rasterio
from rasterio.plot import show

def get_coordinate_system_info_rio(filepath):
#raster_path = "/home/jcohen/lake_change_time_series/geotiff/WGS1984Quad/11/3479/187.tif"
with rasterio.open(filepath) as data:
    crs = data.crs
    print(crs)


def get_coordinate_system_info(filepath):
    try:
        # Open the dataset
        dataset = gdal.Open(filepath)

        # Check if the dataset is valid
        if dataset is None:
            raise Exception("Error: Unable to open the dataset.")

        # Get the spatial reference
        spatial_ref = osr.SpatialReference()

        # Try to import the coordinate system from WKT
        if spatial_ref.ImportFromWkt(dataset.GetProjection()) != gdal.CE_None:
            raise Exception("Error: Unable to import coordinate system from WKT.")

        # Check if the spatial reference is valid
        if spatial_ref.Validate() != gdal.CE_None:
            raise Exception("Error: Invalid spatial reference.")

        # Export the spatial reference to WKT
        return spatial_ref.ExportToWkt()

    except Exception as e:
        print(f"Error: {e}")
        return None

def write_prj_file(geotiff_path, prj_file_path):
    """
        Will create the prj file by getting the geo cordinate system from the input tiff file

        Parameters
        ----------
        geotiff_path : Path to the geo tiff used for processing
        prj_file_path : Path to the location to create the prj files
        """
    try:
        # Get the coordinate system information
        wkt = get_coordinate_system_info(geotiff_path)

        get_coordinate_system_info_rio(geotiff_path)

        print(wkt)
        if wkt is not None:
            # Write the WKT to a .prj file
            with open(prj_file_path, 'w') as prj_file:
                prj_file.write(wkt)

            print(f"Coordinate system information written to {prj_file_path}")
        else:
            print("Failed to get coordinate system information.")

    except Exception as e:
        print(f"Error: {e}")

def copy_to_project_dir(source_path, new_name):
    try:
        # Ensure the source directory exists
        if not os.path.exists(source_path):
            print(f"Source directory '{source_path}' does not exist.")
            return

        # Get the parent directory of the source directory
        parent_directory = os.path.dirname(source_path)

        # Create the new directory path by combining the parent directory and the new name
        new_directory_path = os.path.join(parent_directory, new_name)

        # Copy the source directory to the same location and rename it
        shutil.copytree(source_path, new_directory_path)

        print(f"Directory copied to '{new_directory_path}' and renamed to '{new_name}'.")

    except Exception as e:
        print(f"Error: {e}")

def process_shapefile(image_name):
    data_dir = MPL_Config.WORKER_ROOT
    image_file_name = (image_name).split('.tif')[0]

    shp_dir = os.path.join(data_dir, 'final_shp', image_file_name)
    #projected_dir = os.path.join(data_dir, 'projected_shp', image_file_name)
    projected_dir=os.path.join(MPL_Config.PROJECTED_SHP_DIR,image_file_name)
    temp_dir = os.path.join(data_dir, 'temp_shp', image_file_name)

    shape_file = os.path.join(shp_dir, f"{image_file_name}.shp")
    output_shape_file = os.path.join(temp_dir, f"{image_file_name}.shp")
    projected_shape_file = os.path.join(projected_dir, f"{image_file_name}.shp")

    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Error deleting directory {temp_dir}: {e}")

    try:
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Created directory {temp_dir}")
    except Exception as e:
        print(f"Error creating directory {temp_dir}: {e}")

    w = shapefile.Writer(output_shape_file)

    try:
        r = shapefile.Reader(shape_file)
        print(f"Reading shaper file {r.fields[1:]}")
    except Exception as e:
        print(f"Error reading shapefile {shape_file}: {e}")
        return

    try:
        shutil.rmtree(projected_dir)
        print(f"removed directory {projected_dir}")
    except Exception as e:
        print(f"Error deleting directory {projected_dir}: {e}")

    try:
        os.makedirs(projected_dir, exist_ok=True)
        print(f"Created directory {projected_dir}")
    except Exception as e:
        print(f"Error creating directory {projected_dir}: {e}")

    w.fields = r.fields[1:]

    w.field("Sensor", "C", "10")
    w.field("Date", "C", "10")
    w.field("Time", "C", "10")
    w.field("CatalogID", "C", "20")
    w.field("Area", "N", decimal=3)
    w.field("CentroidX", "N", decimal=3)
    w.field("CentroidY", "N", decimal=3)
    w.field("Perimeter", "N", decimal=3)
    w.field("Length", "N", decimal=3)
    w.field("Width", "N", decimal=3)

    for shaperec in r.iterShapeRecords():
        rec = shaperec.record
        rec.extend([
            image_file_name[0:4],
            image_file_name[5:13],
            image_file_name[13:19],
            image_file_name[20:36]
        ])

        poly_vtx = shaperec.shape.points
        poly = Polygon(poly_vtx)
        area = poly.area
        perimeter = poly.length
        box = poly.minimum_rotated_rectangle
        x, y = box.exterior.coords.xy
        centroid = poly.centroid

        p0 = Point(x[0], y[0])
        p1 = Point(x[1], y[1])
        p2 = Point(x[2], y[2])
        edge_length = (p0.distance(p1), p1.distance(p2))
        length = max(edge_length)
        width = min(edge_length)

        rec.extend([area, centroid.x, centroid.y, perimeter, length, width])
        #print(rec)
        w.record(*rec)
        w.shape(shaperec.shape)
        #print(f"{area} : {perimeter} : {length} : {width}")

    #try:
    w.close()
    #except Exception as e:
    #print(f"Error closing shapefile {output_shape_file}: {e}")

    try:
        shutil.rmtree(projected_dir)
    except Exception as e:
        print(f"Error deleting directory {projected_dir}: {e}")

    # try:
    #     os.makedirs(projected_dir, exist_ok=True)
    # except Exception as e:
    #     print(f"Error creating directory {projected_dir}: {e}")

    os.chmod(temp_dir, 0o777)

    #temp_dir_dbf = os.path.join(temp_dir, f"{image_file_name}.dbf")
    #temp_dir_shx = os.path.join(temp_dir, f"{image_file_name}.shx")

    temp_dir_prj = os.path.join(temp_dir, f"{image_file_name}.prj")
    input_image = os.path.join(MPL_Config.INPUT_IMAGE_DIR, image_name)
    #    data_dir = MPL_Config.WORKER_ROOT
    #    temp_dir = os.path.join(data_dir, 'temp_shp', image_name)
    #    output_project_file = "/home/bizon/amal/code/git_maple_workflow/data3/temp_shp/FID_329_Polygon_3/FID_329_Polygon_3.prj"
    write_prj_file(input_image, temp_dir_prj)

    try:
        shutil.copytree(temp_dir,projected_dir)
    except Exception as e:
        print(f"Error creating projected directory {projected_dir}: {e}")

# Example usage:

def get_tif_file_names(directory_path):
    tif_file_names = []
    try:
        # List all files in the specified directory
        files = os.listdir(directory_path)

        # Add each *.tif file name to the list
        for file in files:
            if file.endswith('.tif'):
                tif_file_names.append(file)

    except Exception as e:
        print(f"Error: {e}")

    return tif_file_names

# Unit TEST CODE
#
files_to_process = get_tif_file_names(MPL_Config.INPUT_IMAGE_DIR)
for image_name in files_to_process[0:3]:
     print("##################################### PROCESSING:", image_name, "###########################")
     process_shapefile(image_name)
     get_coordinate_system_info_rio(image_name)
