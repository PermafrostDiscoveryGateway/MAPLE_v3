"""
MAPLE Workflow
Post processing script used to assign an individual footprint to each ice wedge polygon shapefile produced by MAPLE. 
The footprints are required to follow a specific file hierarchy that coresponsds to the ice wedge polygon data to support 
the visualization step.

This script is based purely on the ArcPy package and uses os library. 

Project: Permafrost Discovery Gateway: Mapping Application for Arctic Permafrost Land Environment(MAPLE)
PI      : Chandi Witharana
Author  : Elias Manos
"""

import arcpy
import os
​
# Create list to store footprint IDs extracted from the footprint polygon shapefiles.
scene_IDs = []
​
# Dissolve features within footprint shapefiles that have the same name
for fp_root, fp_dirs, fp_files in os.walk(r'F:\pan_arctic_master_copy\FootPrints_usedto_clipimages'):
    for fp_file in fp_files:
        full_name = os.path.join(fp_root, fp_file)
        if fp_file.endswith('.shp'):
            split = fp_root.split("\\")
            area = split[3] + '/'
            if area != 'low/':
                country = split[4] + '/'
                out_path = 'F:/pan_arctic_master_copy/footprints_dissolved/' + area + country + fp_file
            else:
                out_path = 'F:/pan_arctic_master_copy/footprints_dissolved/' + area + fp_file
            arcpy.management.Dissolve(full_name, out_path, ['Name'])
​
# Make subfolders for each grid cell selection in the individual footprint folder
for fp_root, fp_dirs, fp_files in os.walk(r'F:\pan_arctic_master_copy\footprints_dissolved'):
    for fp_file in fp_files:
        full_name = os.path.join(fp_root, fp_file)
        if fp_file.endswith('.shp'):
            split_root = fp_root.split("\\")
            split_fname = fp_file.split(".")
            grid_cells = split_fname[0]
            grid_cells = grid_cells[10:] + '_iwp'
            area = split_root[3] + '/'
            if area != 'low/':
                country = split_root[4] + '/'
                out_folder = 'F:/pan_arctic_master_copy/footprints_individual/' + area + country + grid_cells
            else:
                out_folder = "F:/pan_arctic_master_copy/footprints_individual/" + area + grid_cells
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
​
# Extract scene ID from each feature in each footprint shapefile
for fp_root, fp_dirs, fp_files in os.walk(r'F:\pan_arctic_master_copy\footprints_dissolved'):
    for fp_file in fp_files:
        full_name = os.path.join(fp_root, fp_file)
        if fp_file.endswith('.shp'):
            with arcpy.da.SearchCursor(full_name, ['SHAPE@', 'Name']) as sc:
                for row in sc:
                    # scene_IDs.append(row[1])
                    split_root = fp_root.split("\\")
                    split_fname = fp_file.split(".")
                    grid_cells = split_fname[0]
                    grid_cells = grid_cells[10:] + '_iwp/'
                    area = split_root[3] + '/'
                    name = row[1] + "_u16rf3413_pansh"
                    if area != 'low/':
                        country = split_root[4] + '/'
                        out_folder = 'F:/pan_arctic_master_copy/footprints_individual/' + area + country + grid_cells + name
                    else:
                        out_folder = "F:/pan_arctic_master_copy/footprints_individual/" + area + grid_cells + name
                    if not os.path.exists(out_folder):
                        os.mkdir(out_folder)
                    outFC = name + '.shp'
                    arcpy.FeatureClassToFeatureClass_conversion(row[0], out_folder, outFC)
