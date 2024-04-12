import os
import shutil
import shapefile

from osgeo import gdal, osr
import gdal_virtual_file_system as gdal_vfs
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

    def get_coordinate_system_info(self, file_path: str, file_bytes: bytes):
        try:
            # Open the dataset
            # Create virtual file system file for image to use GDAL's file apis.
            vfs = gdal_vfs.GDALVirtualFileSystem(
                file_path, file_bytes)
            virtual_file_path = vfs.create_virtual_file()
            print("here is the virtual file path: ", virtual_file_path)
            dataset = gdal.Open(virtual_file_path)

            # Check if the dataset is valid
            if dataset is None:
                print("gdal error: ", gdal.GetLastErrorMsg())
                raise Exception("Error: Unable to open the dataset")

            # Get the spatial reference
            spatial_ref = osr.SpatialReference()

            # Try to import the coordinate system from WKT
            if spatial_ref.ImportFromWkt(dataset.GetProjection()) != gdal.CE_None:
                raise Exception(
                    "Error: Unable to import coordinate system from WKT.")

            # Check if the spatial reference is valid
            if spatial_ref.Validate() != gdal.CE_None:
                raise Exception("Error: Invalid spatial reference.")

            # Export the spatial reference to WKT
            dataset = None
            vfs.close_virtual_file()
            return spatial_ref.ExportToWkt()

        except Exception as e:
            print(f"Error when getting the coordinate system info: {e}")
            dataset = None
            vfs.close_virtual_file()
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
            wkt = self.get_coordinate_system_info(geotiff_path, geotiff_bytes)

            print(wkt)
            if wkt is not None:
                # Write the WKT to a .prj file
                with open(prj_file_path, "w") as prj_file:
                    prj_file.write(wkt)

                print(
                    f"Coordinate system information written to {prj_file_path}")
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
        image_name = row["image_name"]
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

            w.record(Class=shapefile_result.class_id, Sensor=image_name[0:4], Date=image_name[5:13],
                     Time=image_name[13:19], CatalogID=image_name[20:36], Area=poly.area,
                     CentroidX=centroid.x, CentroidY=centroid.y, Perimeter=poly.length, Length=length, Width=width)
        w.close()

    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
        image_name = row["image_name"]
        print("Writing shapefiles for:", image_name)
        shapefile_output_dir_for_image = os.path.join(
            self.shpfile_output_dir, f"{image_name}.shp")
        self.write_shapefile(row, shapefile_output_dir_for_image)

        prj_output_dir_for_image = os.path.join(
            self.shpfile_output_dir, f"{image_name}.prj")
        self.write_prj_file(
            geotiff_path=row["path"], geotiff_bytes=row["bytes"], prj_file_path=prj_output_dir_for_image)
        row["shapefile_output_dir"] = shapefile_output_dir_for_image
        return row
