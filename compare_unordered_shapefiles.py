"""
Compares the features in two shapefiles. The geometry and properties for each feature are compared. 
The features don't have to be in the same order in the two files for them to be equal. This approach
assumes that each feature in the file is unique (ie. it doesn't guarantee equality if there are 
duplicate features in one of the files.)
"""
import argparse
import sys

import fiona
from shapely.geometry import shape


def feature_to_hashable(feature):
    """Converts a feature into a hashable representation."""
    geometry_repr = shape(feature['geometry']).wkt
    attributes = tuple(sorted(feature['properties'].items()))
    return (geometry_repr, attributes)


def compare_unordered_shapefiles(file1, file2):
    """Compares shapefiles using sets. This approach doesn't work if there are duplicate features in one of the files."""
    with fiona.open(file1) as src1, fiona.open(file2) as src2:
        set1 = {feature_to_hashable(feature) for feature in src1}
        set2 = {feature_to_hashable(feature) for feature in src2}

        if len(set1) != len(set2):
            print(
                f"{file1} doesn't have the same number of elements: {len(set1)}, as {file2}: {len(set2)}")
            return

        if set1 == set2:
            print("Shapefiles seem to have identical features")
            return

        diff1 = set1 - set2  # Features only in shapefile1
        diff2 = set2 - set1  # Features only in shapefile2

        print("Shapefiles have differences:")
        print(f"{len(diff1)} features unique to {file1}")
        print(f"{len(diff2)} features unique to {file2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the features in two shapefiles.")
    parser.add_argument("file1", help="Path to the first shapefile")
    parser.add_argument("file2", help="Path to the second shapefile")

    args = parser.parse_args()

    # Check if both file paths were provided
    if not args.file1 or not args.file2:
        print("Error: Please provide both shapefile paths.")
        sys.exit(1)

    compare_unordered_shapefiles(args.file1, args.file2)

# Usage:
# python compare_unordered_shapefiles.py <path to file1> <path to file2>
