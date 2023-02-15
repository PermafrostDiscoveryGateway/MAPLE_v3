#!/usr/bin/env python3
#env maple

"""
Code base to clean the inferences from known artifacts

This code will take as input shape files with known artifacts and do an intersect with infered shape file and remove if
they fall into a known artifact.

The input shp file will be edited from this code. If you want to keep the original file make a copy and point it to this
Code base.

Expects the cleaning artifacts to be in a single folder and will go through all the shp files. These shape files shoul
have only one feature (dissolved) if multi feature it will take the first feature and continue with a warning

Since combining the cleaning data (Union) take time, it advisable to run this for a batch of files.

Author amal.perera@uconn.edu

"""
import sys
from osgeo import ogr
import os
import time
import glob

#from mpl_config import MPL_Config
#sys.path.append(MPL_Config.ROOT_DIR)
# GLOBAL settings
#Reporting verbosity to find out deletion of polygons for testing
VERBOSE_granularity=10000 # output print frequency for deleting loops 0 will have no outputs 1=all 10=every 10
# sys.path.append(MPL_Config.ROOT_DIR)

def clean_inference_shapes(clean_data_dir="./data/cln_data",
                           input_data_dir="./data/final_shp",
                           input_data_boundary_file_path="./data/cvg_data/sample2_out_boundry.shp",
                           ):
    """
        Clean Known false positives (mis-classified) Polygons from shape files. Based on know ground truth
        Parameters
        ----------
        clean_data_dir: Location where you have the cleaning data ground truth to be used for cleaning
        input_data_dir: Location of shape files to be cleaned
        input_data_boundary_file_path: Boundary/Envelope of the input data files if available to reduce compute
    """
    st=time.time()
    # LOAD the cleaning data files
    clean_data_path = clean_data_dir + "/*.shp" # look for all shp files in the cln_data dir given
    clean_data_fps = sorted(glob.glob(clean_data_path))
    print("List of cleaning data files selected to LOAD:",clean_data_fps)
    drv = ogr.GetDriverByName('ESRI Shapefile')
    all_geoms=ogr.Geometry(ogr.wkbMultiPolygon)
    for fp in clean_data_fps:
        # Open cleaning data files:layer
        print("Loading Cleaning Data:",fp)
        dc = drv.Open(fp, False)  # False not allowed to edit the files
        c_lyr = dc.GetLayer()
        if (c_lyr.GetFeatureCount()>1):
            print("Warning: More than 1 feature in cleaning shapefile only first taken, dissolve data and use")
        c_feature=c_lyr.GetFeature(0)
        print("Cleaning Data Added:",fp,"\nFields in File",[field.name for field in c_lyr.schema])
        c_geometry=c_feature.geometry().Clone()
        all_geoms.AddGeometry(c_geometry)
    # TIME taken for testing/Optimizing code can/should be commented
    nd=time.time()
    print("Wall Time for LOADING cleaning data {}".format(nd-st))
    st=time.time()
    all_union=all_geoms.UnionCascaded()
    nd=time.time()
    print("Wall Time for UNION cleaning data {}".format(nd-st))

    # </TIME>#######################################################
    # Open COVERAGE area filter to remove areas that are not covered by the study
    # NOT used as it is better to remove unwanted areas in the input cleaning files and reduce input dependencies
    #The cleaning data will be interected with the coverage data to reduce the processing overhead.
    #Coverae file is located in the cvg_data folder
    # dcoverage = drv.Open("./data/cvg_data/sample2_out_boundry.shp",False)  # False as we are NOT editing the file
    # coverage_lyr = dcoverage.GetLayer()
    # coverage_feature=coverage_lyr.GetNextFeature()
    # coverage_geometry=coverage_feature.geometry().Clone()
    # all_union=all_union.Intersection(coverage_geometry)
    # print("Coverage Data Loaded from {}".format("./cvg_data/CAVM_coverage.shp"))

    # LOAD : AND INTERSECT the boundary aka feature envelope of the input files to reduce the processing
    #drv = ogr.GetDriverByName('ESRI Shapefile')
    if (os.path.isfile(input_data_boundary_file_path)):
        dboundry = drv.Open(input_data_boundary_file_path,False)  #False as we are NOT editing the file
        dboundry_lyr = dboundry.GetLayer()
        b_feature=dboundry_lyr.GetNextFeature()
        b_geometry=b_feature.geometry().Clone()
        all_union=all_union.Intersection(b_geometry)
        print("Boundary Data Loaded from {}".format("sample2_out_boundry.shp"))
    else:
        print("No Boundary/Feature Envelope Data Loaded to reduce computing")

    #LOAD all Input files given for cleaning
    input_data_paths = input_data_dir + "/*" # look for all directories pointed for input processing
    input_data_fps = sorted(glob.glob(input_data_paths))
    print("input data directory",input_data_dir)
    print("List of input data directories selected to PROCESS:CLEAN:\n")
    print(*input_data_fps,sep="\n")

    # for testing take only few files
    input_data_fps=input_data_fps[0:1]

    for fp in input_data_fps:
        # LOAD : READ each input file given
        input_data_file = sorted(glob.glob(fp + "/*.shp") )
        print("File selected to PROCESS:CLEAN:", input_data_file[0])
        ds = drv.Open(input_data_file[0],True)  # True as we are editing the file
        s_lyr = ds.GetLayer()
        print("Total Features before Cleaning Filter {}".format(s_lyr.GetFeatureCount()))
        print("input data file : fields",[field.name for field in s_lyr.schema])
        st=time.time()
        st_cpu = time.process_time()
        # Go through all features and remove them if it falls within the filter_geom
        for fid in range(s_lyr.GetFeatureCount()): # READ all features in the input file to check and remove
        #for fid in [6064,6065,6066,6067,6068,14270,14271,14272,14273,14274,14275]: # For testing on sample2_org.shp
        # 50% split
            s_feature=s_lyr.GetFeature(fid)

            # OPTION:1 Creating a point shape Using the centroid from the input file to check
            # Instead of checking the polygon of the shape to reduce computing
            shp_centroid = ogr.Geometry(ogr.wkbPoint)
            shp_centroid.AddPoint(s_feature.GetField('CentroidX'),s_feature.GetField('CentroidY'))

            # OPTION:2 Using the polygon in the input feature to create a geometry
            #s_geometry=s_feature.geometry().Clone()
            #if (all_union.Intersection(s_geometry).IsEmpty()): #Using the polygon OPTION 2
            if (all_union.Intersection(shp_centroid).IsEmpty()): # Using the centroid point of shape OPTION 1
                pass
                if VERBOSE_granularity and (fid%VERBOSE_granularity == 0):
                    print("NOT Deleted:",fid)
            else:
                s_lyr.DeleteFeature(fid)
                if VERBOSE_granularity and (fid%VERBOSE_granularity == 0):
                    print("Deleted:",fid)

        ds.ExecuteSQL('REPACK ' + s_lyr.GetName())
        ds.ExecuteSQL('RECOMPUTE EXTENT ON ' + s_lyr.GetName())
        #del ds   #not working need to check if required. Since when this goes out of scope it will be deleted
        print("Sample Features after deleting {}".format(s_lyr.GetFeatureCount()))

        # TIME taken for testing/Optimizing code can/should be commented
        nd = time.time()
        nd_cpu = time.process_time()
        print("Wall Time for deleting {}".format(nd-st))
        print("CPU Time for deleting {}".format(nd_cpu-st_cpu))
        # </TIME>#######################################################

############################## MAIN CODE block to be customized to envirenemnt ##########################
from mpl_config import MPL_Config
clean_inference_shapes(MPL_Config.CLEAN_DATA_DIR,
                       MPL_Config.FINAL_SHP_DIR,
                       "./data/cvg_data/sample2_out_boundry.shp")
