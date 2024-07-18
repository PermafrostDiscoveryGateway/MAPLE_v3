from datetime import datetime
from io import BytesIO 
import os
import shutil
from typing import Any, Dict

from osgeo import gdal, osr
from shapely.geometry import Polygon, Point
import shapefile

import gdal_virtual_file_path as gdal_vfp
from mpl_config import MPL_Config


class WriteShapefiles:
    def __init__(
        self,
        config: MPL_Config
    ):
        self.config = config
        self.current_timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def __get_coordinate_system_info(self, file_path: str, file_bytes: bytes):
        try:
            # Open the dataset
            # Create virtual file path for image to use GDAL's file apis.
            spatial_ref = osr.SpatialReference()
            with gdal_vfp.GDALVirtualFilePath(
                file_path, file_bytes) as virtual_file_path:
                with gdal.Open(virtual_file_path) as dataset:
                    # Check if the dataset is valid
                    if dataset is None:
                        print("gdal error: ", gdal.GetLastErrorMsg())
                        raise Exception("Error: Unable to open the dataset")

                    # Try to import the coordinate system from WKT
                    if spatial_ref.ImportFromWkt(dataset.GetProjection()) != gdal.CE_None:
                        raise Exception(
                            "Error: Unable to import coordinate system from WKT.")

            # Check if the spatial reference is valid
            if spatial_ref.Validate() != gdal.CE_None:
                raise Exception("Error: Invalid spatial reference.")

            # Export the spatial reference to WKT
            return spatial_ref.ExportToWkt()

        except Exception as e:
            print(f"Error when getting the coordinate system info: {e}")
            return None

    def write_prj_file(self, geotiff_path: str, geotiff_bytes: bytes, prj_file_path: str):
        """
        Will create the prj file by getting the geo cordinate system from the input tiff file

        Parameters
        ----------
        geotiff_path : geotiff_path : Path to the geo tiff used for processing
        geotiff_bytes: Bytes of the input image
        prj_file_path : Path to the location to create the prj files
        """
        try:
            # Get the coordinate system information
            wkt = self.__get_coordinate_system_info(geotiff_path, geotiff_bytes)
            if wkt is not None:
                # Write the WKT to a .prj file
                if self.config.GCP_FILESYSTEM is not None:
                    with self.config.GCP_FILESYSTEM.open(prj_file_path, "w") as prj_file:
                        prj_file.write(wkt)
                else:
                    with open(prj_file_path, "w") as prj_file:
                        prj_file.write(wkt)
                print(
                    f"Coordinate system information written to {prj_file_path}")
            else:
                print("Failed to get coordinate system information.")

        except Exception as e:
            print(f"Error when trying to write the prj file: {e}")

    def __write_shapefile_helper(self, row: Dict[str, Any], writer: shapefile.Writer):
        writer.field("Class", "C", size=5)
        writer.field("Sensor", "C", "10")
        writer.field("Date", "C", "10")
        writer.field("Time", "C", "10")
        writer.field("CatalogID", "C", "20")
        writer.field("Area", "N", decimal=3)
        writer.field("CentroidX", "N", decimal=3)
        writer.field("CentroidY", "N", decimal=3)
        writer.field("Perimeter", "N", decimal=3)
        writer.field("Length", "N", decimal=3)
        writer.field("Width", "N", decimal=3)
        image_name = row["image_name"]
        for shapefile_result in row["image_shapefile_results"].shapefile_results:
            polygons = shapefile_result.polygons
            writer.poly([polygons.tolist()])

            poly = Polygon(polygons)
            centroid = poly.centroid
            box = poly.minimum_rotated_rectangle
            x, y = box.exterior.coords.xy
            p0 = Point(x[0], y[0])
            p1 = Point(x[1], y[1])
            p2 = Point(x[2], y[2])
            edge_length = (p0.distance(p1), p1.distance(p2))
            length = max(edge_length)
            width = min(edge_length)

            writer.record(Class=shapefile_result.class_id, Sensor=image_name[0:4], Date=image_name[5:13],
                     Time=image_name[13:19], CatalogID=image_name[20:36], Area=poly.area,
                     CentroidX=centroid.x, CentroidY=centroid.y, Perimeter=poly.length, Length=length, Width=width)

    def write_shapefile(self, row: Dict[str, Any], shapefile_output_dir_for_image):
        if self.config.GCP_FILESYSTEM is not None:
            with BytesIO() as shp_mem, BytesIO() as shx_mem, BytesIO() as dbf_mem:
                with shapefile.Writer(shp=shp_mem, shx=shx_mem, dbf=dbf_mem) as writer:
                    self.__write_shapefile_helper(row, writer)
                    writer.close()

                with self.config.GCP_FILESYSTEM.open(f"{shapefile_output_dir_for_image}.shp", 'wb') as shp_file:
                    shp_file.write(shp_mem.getvalue())
                with self.config.GCP_FILESYSTEM.open(f"{shapefile_output_dir_for_image}.shx", 'wb') as shx_file:
                    shx_file.write(shx_mem.getvalue())
                with self.config.GCP_FILESYSTEM.open(f"{shapefile_output_dir_for_image}.dbf", 'wb') as dbf_file:
                    dbf_file.write(dbf_mem.getvalue())
        else:
            writer = shapefile.Writer(f"{shapefile_output_dir_for_image}.shp")
            self.__write_shapefile_helper(row, writer)
            writer.close()

    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
        image_name = row["image_name"]
        print("Writing shapefiles for:", image_name)
        shapefile_output_dir_for_image = os.path.join(
            self.config.RAY_OUTPUT_SHAPEFILES_DIR, self.current_timestamp_str, image_name)
        self.write_shapefile(row, shapefile_output_dir_for_image)
        self.write_prj_file(
            geotiff_path=row["path"], geotiff_bytes=row["bytes"], prj_file_path=f"{shapefile_output_dir_for_image}.prj")
        row["shapefile_output_dir"] = shapefile_output_dir_for_image
        return row
