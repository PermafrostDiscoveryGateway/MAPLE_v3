import os
import shutil
import shapefile

from mpl_config import MPL_Config
from osgeo import gdal, osr
from shapely.geometry import Polygon, Point
from typing import Any, Dict

def delete_and_create_dir(dir: str):
    try:
        shutil.rmtree(dir)
    except Exception as e:
        print(f"Error deleting directory {dir}: {e}")

    try:
        os.makedirs(dir, exist_ok=True)
        print(f"Created directory {dir}")
    except Exception as e:
        print(f"Error creating directory {dir}: {e}")

class WriteShapefiles:
    def __init__(
        self,
        shpfile_output_dir: str
    ):
        self.shpfile_output_dir = shpfile_output_dir
        delete_and_create_dir(self.shpfile_output_dir)

    def get_coordinate_system_info(self, filepath: str):
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
            dataset = None
            return spatial_ref.ExportToWkt()

        except Exception as e:
            print(f"Error when getting the coordinate system info: {e}")
            dataset = None
            return None

    def write_prj_file(self, geotiff_path: str, prj_file_path: str):
        """
        Will create the prj file by getting the geo cordinate system from the input tiff file

        Parameters
        ----------
        geotiff_path : Path to the geo tiff used for processing
        prj_file_path : Path to the location to create the prj files
        """
        try:
            # Get the coordinate system information
            wkt = self.get_coordinate_system_info(geotiff_path)

            print(wkt)
            if wkt is not None:
                # Write the WKT to a .prj file
                with open(prj_file_path, "w") as prj_file:
                    prj_file.write(wkt)

                print(f"Coordinate system information written to {prj_file_path}")
            else:
                print("Failed to get coordinate system information.")

        except Exception as e:
            print(f"Error when trying to write the prj file: {e}")

    def write_shapefile(self, row: Dict[str, Any], shapefile_output_dir_for_image: str):
        w = shapefile.Writer(shapefile_output_dir_for_image)
        w.field("Class", "C", size=5)
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
        image_file_name = row["image_file_name"].split(".tif")[0]
        for shapefile_result in row["image_shapefile_results"].shapefile_results:
            polygons = shapefile_result.polygons
            w.poly([polygons.tolist()])

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

            w.record(Class=shapefile_result.class_id, Sensor=image_file_name[0:1], Date=image_file_name[1:2],
                     Time=image_file_name[2:3], CatalogID=image_file_name[3:4], Area=poly.area,
                     CentroidX=centroid.x, CentroidY=centroid.y, Perimeter=poly.length, Length=length, Width=width)
        w.close()

    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
        image_file_name = row["image_file_name"].split(".tif")[0]
        shapefile_output_dir_for_image = os.path.join(self.shpfile_output_dir, f"{image_file_name}.shp")
        self.write_shapefile(row, shapefile_output_dir_for_image)

        prj_output_dir_for_image = os.path.join(self.shpfile_output_dir, f"{image_file_name}.prj")
        print("here is the virtual file path: ", row["vfs_image_path"])
        self.write_prj_file(geotiff_path=row["vfs_image_path"], prj_file_path=prj_output_dir_for_image)
        row["shapefile_output_dir"] = shapefile_output_dir_for_image
        return row

