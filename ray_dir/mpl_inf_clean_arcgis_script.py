"""
MAPLE Workflow
(5) (alternative) script to clean the inferences from known landscape features.

This code will take as input shape files with known landscape and do an intersect with infered shape file and remove if
they fall into a known artifact.

The input shp file will be edited from this code. If you want to keep the original file make a copy and point it to this
Code base.

This script provides function for cleaning polygon features from ice wedge polygon (IWP) shapefiles that overlap with
other landscape features that might produce confusion (such as glaciers, ocean, lakes, infrastructure). The function
is based purely on the ArcPy package and uses os library to loop through all the IWP shapefiles. You only have to
provide path to the root directory that holds all the shapefiles.

The 4 landscape features,files,references used for cleaning
1. Infrastructure: SACHI.shp: Sentinel-1/2 derived Arctic Coastal Human Impact dataset (SACHI) (zenodo.org)
2. Glaciers : glims_polygons.shp: GLIMS Glacier Database https://www.glims.org/download/ doi:10.1016/j.gloplacha.2006.07.018
3. Water: arctic_water.shp : National Snow and Ice Data Center: http://nsidc.org/data/
4. Lakes : Lakes lakes.shp : Ingmar Nitze @ Alfred-Wegener-Institut : https://www.awi.de/en/ 

Project: Permafrost Discovery Gateway: Mapping Application for Arctic Permafrost Land Environment(MAPLE)
PI      : Chandi Witharana
Author  : Elias Manos
"""

import arcpy
import os

arcpy.env.overwriteOutput = True

def clean_iwp(root_dir, intersect_lyr1, intersect_lyr2, intersect_lyr3, intersect_lyr4):
    processed_list = []
​
    """
     This function deletes individual polygons in ice wedge polygon prediction shapefiles that intersect with features
     in other data layers representing landscape features that could cause CNN model to generate false positives.
​
     root_dir: string = Location of the directory that holds IWP prediction shapefiles
​
     intersect_lyr1: string = Location of shapefile 1 (infrastructure) which will be used to determine which polygons 
     will be removed as false positives
     
     intersect_lyr2: string = Location of shapefile 2 (glaciers) which will be used to determine which polygons will be 
     removed as false positives
     
     intersect_lyr3: string = Location of shapefile 3 (lakes) which will be used to determine which polygons will be 
     removed as false positives
     
     intersect_lyr4: string = Location of shapefile 4 (coastal water) which will be used to determine which polygons 
     will be removed as false positives
​
    """
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            if d.endswith('iwp'):
                process_folder = os.path.join(root, d)
                print(process_folder)
                for root2, dirs2, files2 in os.walk(process_folder):
                    for file in files2:
                        full_name = os.path.join(root2, file)
                        # Select only files with extension '.shp', since this is the input to ArcPy functions
                        if file.endswith('.shp'):
                            # Create temporary layer from IWP prediction shapefile to perform selection
                            arcpy.MakeFeatureLayer_management(full_name, 'temp_IWP_lyr')
                            # Select IWP features that intersect with other landscape features
                            arcpy.SelectLayerByLocation_management('temp_IWP_lyr', 'INTERSECT', intersect_lyr1,
                                                                   selection_type="NEW_SELECTION")
                            # Select IWP features that intersect with another landscape feature and add this to the
                            # previous selection
                            arcpy.SelectLayerByLocation_management('temp_IWP_lyr', 'INTERSECT', intersect_lyr2,
                                                                   selection_type="ADD_TO_SELECTION")
                            # Same as above
                            arcpy.SelectLayerByLocation_management('temp_IWP_lyr', 'INTERSECT', intersect_lyr3,
                                                                   selection_type="ADD_TO_SELECTION")
                            # Same as above
                            arcpy.SelectLayerByLocation_management('temp_IWP_lyr', 'INTERSECT', intersect_lyr4,
                                                                   selection_type="ADD_TO_SELECTION")
                            # If IWP features have been selected, delete those features
                            if int(arcpy.GetCount_management('temp_IWP_lyr').getOutput(0)) > 0:
                                arcpy.DeleteFeatures_management('temp_IWP_lyr')
                                processed_list.append(full_name)

# SET YOUR DATA PATHS AND RUN SCRIPT
# Change the follwoing paths to your own paths.
# Location of root IWP directory
#iwp_dir_path = "F:/pan_arctic_master_copy/iwp_files/low"
print("starting..........")
#lyr_path1 = "D:/MAPLE/bartsch_data/SACHI.shp"
#lyr_path2 = "D:/MAPLE/glacier/glims_download_13173/glims_polygons.shp"
#lyr_path3 = "D:/MAPLE/ingmar_lakes/lakes.shp"
#lyr_path4 = "D:/MAPLE/arctic_water/arctic_water.shp"
# Run script
clean_iwp(iwp_dir_path, lyr_path1, lyr_path2, lyr_path3, lyr_path4)
