import fiona
import argparse
import sys
from shapely.geometry import shape

def feature_to_hashable(feature):
    """Converts a feature into a hashable representation."""
    geometry_repr = shape(feature['geometry']).wkt  # Represent geometry as WKT
    attributes = tuple(sorted(feature['properties'].items()))  # Sort attributes
    return (geometry_repr, attributes)

def compare_unordered_shapefiles(file1, file2):
    """Compares shapefiles using the set approach."""
    with fiona.open(file1) as src1, fiona.open(file2) as src2:
        set1 = {feature_to_hashable(feature) for feature in src1}
        set2 = {feature_to_hashable(feature) for feature in src2}

        if len(set1) != len(set2):
            print("{} doesn't have the same number of elements: {}, as {}: {}".format(
                file1, len(set1), file2, len(set2)
            ))
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
    parser = argparse.ArgumentParser(description="Compare the features in two shapefiles.")
    parser.add_argument("file1", help="Path to the first shapefile")
    parser.add_argument("file2", help="Path to the second shapefile")

    args = parser.parse_args()

    # Check if both file paths were provided
    if not args.file1 or not args.file2:
        print("Error: Please provide both shapefile paths.")
        sys.exit(1)

    compare_unordered_shapefiles(args.file1, args.file2)

# Usage:
#compare_unordered_shapefiles("/usr/local/google/home/kaylahardie/MAPLE_v3/data/ray_shapefiles/test_image_01.shp", "/usr/local/google/home/kaylahardie/MAPLE_v3/data/projected_shp/test_image_01/test_image_01.shp")
#compare_unordered_shapefiles("/usr/local/google/home/kaylahardie/MAPLE_v3/data/projected_shp copy/test_image_01/test_image_01.shp", "/usr/local/google/home/kaylahardie/MAPLE_v3/data/projected_shp/test_image_01/test_image_01.shp")  