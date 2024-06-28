#!/usr/bin/env python3
# env maple
"""
MAPLE Workflow
(5) Code base to clean the inferences from known artifacts.

This code will take as input shape files with known artifacts and do an intersect with infered shape file and remove if
they fall into a known artifact.

The input shp file will be edited from this code. If you want to keep the original file make a copy and point it to this
Code base.

Expects the cleaning artifacts to be in a single folder and will go through all the shp files. These shape files should
have only one feature (dissolved) if multi feature it will take the first feature and continue with a warning.

Since combining the cleaning data (Union) take time, it advisable to run this for a batch of files.

Project: Permafrost Discovery Gateway: Mapping Application for Arctic Permafrost Land Environment(MAPLE)
PI      : Chandi Witharana
Author  : Amal Shehan Perera
"""

from mpl_config import MPL_Config
from osgeo import ogr, osr
import os
import time
import glob


def listdirpaths(folder):
    """List the file paths in a folder/directory so that they can be read
    Returns:
        list of file paths
    Arguments:
        folder: folder in which file paths to be listed.
        Output
    """
    return [
        d
        for d in (os.path.join(folder, d1) for d1 in os.listdir(folder))
        if os.path.isdir(d)
    ]


######################## GLOBAL settings ################################################
# Reporting verbosity to find out deletion of polygons for testing
VERBOSE_granularity = 0  # output print frequency for deleting loops 0 will have no outputs 1=all 10=every 10


#########################################################################################
def clean_inference_shapes(
    clean_data_dir="./data/cln_data",
    input_data_dir="./data/projected_shp",
    input_data_boundary_file_path="./data/input_bound/sample2_out_boundry.shp",
):
    """Clean Known false positives (mis-classified) Polygons, based on known ground truth (rivers,lakes,buildings,etc
    Parameters:
        clean_data_dir: Location where you have the cleaning data ground truth to be used for cleaning
        input_data_dir: Location of shape files to be cleaned
        input_data_boundary_file_path: Boundary/Envelope of the input data files if available to reduce compute.
        ToDo:Refactor this function to show the higher level abstract functionality
    """
    st = time.time()
    # LOAD the cleaning data files
    ogr.UseExceptions()  # Enable gdal:ogr errors to be captured/shown so that we know when things fail!
    # Note: No error handling in gdal
    try:  # LOAD cleaning data
        drv = ogr.GetDriverByName("ESRI Shapefile")
        try:  # GET the source/input file projected coordinate system to transfer the cleaning data coordinates
            input_data_fps = listdirpaths(input_data_dir)
            inDataFile = drv.Open(input_data_fps[0])
            inDataLyr = inDataFile.GetLayer()
            inSpatialRef = inDataLyr.GetSpatialRef()
            # loading projection
            sr = osr.SpatialReference(str(inSpatialRef))
            srid = sr.GetAuthorityCode(None)
            print("Input Data SRID:", srid)

            print("Geo Ref for input data for transformation of cleaning data obtained")
        except:
            print(
                "Unable to get Geo Ref for input data for transfomation of cleaning data"
            )

        clean_data_fps = [
            (clean_data_dir + "/" + f_nm)
            for f_nm in os.listdir(clean_data_dir)
            if f_nm.endswith(".shp")
        ]

        print("*********************************************")
        print("List of cleaning data files selected to LOAD:")
        print(*clean_data_fps, sep="\n")
        try:
            all_geoms_union = ogr.Geometry(ogr.wkbMultiPolygon)
            for fp in clean_data_fps:
                # Open cleaning data files:layer
                print("Loading Cleaning Data FILE:\n", fp)
                try:
                    dc = drv.Open(fp, True)  # True allowed to edit if there are errors
                except:
                    print("Unable to open cleaning Data:", fp)
                try:
                    c_lyr = dc.GetLayer()
                except:
                    print("Unable to get Layer in cleaning Data:", fp)
                if c_lyr.GetFeatureCount() > 1:
                    print(
                        "Warning: More than 1 feature in cleaning shapefile only first taken, dissolve data and use"
                    )
                try:  # Try to fix geom errors in cleaning data before reaging for processing if required
                    feature = c_lyr.GetFeature(0)
                    geom = feature.GetGeometryRef()
                    if not geom.IsValid():
                        feature.SetGeometry(geom.Buffer(0))  # <====== SetGeometry
                        c_lyr.SetFeature(feature)  # <====== SetFeature
                        assert feature.GetGeometryRef().IsValid()  # Doesn't fail
                        c_lyr.ResetReading()
                        print("Invalid Geometry in cleaning data Fixed")
                except:
                    print("unable to fix geometry")
                try:
                    c_feature = c_lyr.GetFeature(0)
                except:
                    print("Unable to get feature")
                print(
                    "Cleaning Data Added:",
                    fp,
                    "\nFields in File",
                    [field.name for field in c_lyr.schema],
                )
                try:
                    c_geometry = c_feature.geometry().Clone()
                except:
                    print("unable to clone geom")
                try:  # Transfer cleaning data geo coordinates to the input file coordinate system
                    clnSpatialRef = c_lyr.GetSpatialRef()
                    sr = osr.SpatialReference(str(clnSpatialRef))
                    srid = sr.GetAuthorityCode(None)
                    print("Cleaning SRID:", srid)
                    coordTrans = osr.CoordinateTransformation(
                        clnSpatialRef, inSpatialRef
                    )
                    c_geometry.Transform(coordTrans)
                    print("Cleaning Data Geometry geo coordinate transformed")
                except:
                    print("Unable to do Cleaning Data geo coordinate transform")
                try:
                    all_geoms_union = all_geoms_union.Union(c_geometry).Clone()
                    print("Cleaining Geometry Unioined")
                    print("All geom union geometry Area:", all_geoms_union.Area())
                except:
                    print("Unable to union cleaning data geom")

                try:  # Fix if any geometry errors
                    c_geometry.CloseRings()
                    # print("Geometry Closed Rings")
                except:
                    print("Unable Close Ring geom")

            # TIME taken for testing/Optimizing code can/should be commented
            nd = time.time()
            print("****************************************************")
            print(
                "Wall Time for LOAD/UNION all cleaning data %5.4f min"
                % ((nd - st) / 60)
            )
            st = time.time()
            all_union = all_geoms_union.Clone()
            print("****************************************************")
            # </TIME>#######################################################
        except:
            print("Unable to process load/process shape files via gdal:ogr")
    except:
        print("Unable to READ/LOAD Cleaning Data main loop")

    # LOAD : AND INTERSECT the boundary aka feature envelope of the input files to reduce the processing if availble
    if all_union.Area() == 0:
        print("All Clean Geom union geometry Area=0 nothing will be cleaned:")
    print("input data boundry file path:", input_data_boundary_file_path)
    if os.path.isfile(input_data_boundary_file_path):
        dboundry = drv.Open(
            input_data_boundary_file_path, False
        )  # False as we are NOT editing the file
        dboundry_lyr = dboundry.GetLayer()
        b_feature = dboundry_lyr.GetNextFeature()
        b_geometry = b_feature.geometry().Clone()
        all_union = all_union.Intersection(b_geometry).Clone()
        print("Boundary Data Loaded from {}".format("sample2_out_boundry.shp"))
    else:
        print("No Boundary/Feature Envelope Data Loaded to reduce computing")
    print("all geom union geometry Area:", all_union.Area())
    if all_union.Area() == 0:
        print(
            "All Clean Geom union geometry Area=0 after boundry intersection nothing will be cleaned:"
        )
    # LOAD all Input files given for cleaning
    input_data_fps = sorted(listdirpaths(input_data_dir))
    print("input data directory", input_data_dir)
    print("List of input data directories selected to PROCESS:CLEAN:")
    print(*input_data_fps, sep="\n")
    # FOR testing if you want to take only few files
    if all_union.Area() > 0:
        for fp in input_data_fps:
            # LOAD : READ each input file given
            input_data_file = sorted(glob.glob(fp + "/*.shp"))
            print("File selected to PROCESS:CLEAN:", input_data_file[0])
            ds = drv.Open(input_data_file[0], True)  # True as we are editing the file
            s_lyr = ds.GetLayer()
            print(
                "Total Features before Cleaning Filter {}".format(
                    s_lyr.GetFeatureCount()
                )
            )
            print("input data file : fields", [field.name for field in s_lyr.schema])
            st = time.time()
            st_cpu = time.process_time()

            for fid in range(
                s_lyr.GetFeatureCount()
            ):  # READ all features in the input file to check and remove
                s_feature = s_lyr.GetFeature(fid)

                # OPTION:1 Creating a point shape Using the centroid from the input file to check
                # Instead of checking the polygon of the shape to reduce computing
                shp_centroid = ogr.Geometry(ogr.wkbPoint)
                shp_centroid.AddPoint(
                    s_feature.GetField("CentroidX"), s_feature.GetField("CentroidY")
                )

                # OPTION:2 Using the polygon in the input feature to create a geometry
                # s_geometry=s_feature.geometry().Clone()
                # if (all_union.Intersection(s_geometry).IsEmpty()): #Using the polygon OPTION 2
                keep = True
                try:  # to see if intersection can  be done with errors in the cleaning shp file.
                    keep = all_union.Intersection(
                        shp_centroid
                    ).IsEmpty()  # Using the centroid point of shape OPTION 1
                    # if (all_union.Intersection(shp_centroid).IsEmpty()): # Using the centroid point of shape OPTION 1
                except:  # ignore that shape
                    pass
                    print("Unable to intersect", fid)
                if keep:
                    pass
                    # if VERBOSE_granularity and (fid%VERBOSE_granularity == 0):
                    #     print("NOT Deleted:",fid)
                else:
                    s_lyr.DeleteFeature(fid)
                    # if VERBOSE_granularity and (fid%VERBOSE_granularity == 0):
                    #     print("Deleted:",fid)

            ds.ExecuteSQL("REPACK " + s_lyr.GetName())
            ds.ExecuteSQL("RECOMPUTE EXTENT ON " + s_lyr.GetName())
            # del ds   #not working need to check if required. Since when this goes out of scope it will be deleted
            print("Sample Features after deleting {}".format(s_lyr.GetFeatureCount()))

        # TIME taken for testing/Optimizing code can/should be commented
        nd = time.time()
        nd_cpu = time.process_time()
        print("*****************************************************")
        print("Wall Time for check/remove %6.4f min" % ((nd - st) / 60))
        print("CPU  Time for check/remove %6.4f min" % ((nd_cpu - st_cpu) / 60))
        # </TIME>#######################################################
        print("*******************END - CLEANING *******************")
    else:
        print("No cleaining done : Cleaning artifacts are outside the input data area")


############################## To be customized to eval environment if required or testing ##########################
# Not required since the clean function is called from the workflow
def main():
    config = MPL_Config()
    clean_inference_shapes(
        config.CLEAN_DATA_DIR,
        config.FINAL_SHP_DIR,
        "./data/input_bound/sample3n4_out_bound.shp",
    )


if __name__ == "__main__":
    main()
