"""
    Code to set up the required dir/file structure to run the MAPLE work flow.
    This script to be run once prior to the executing the workflow for the first time.
    This was created to make sure the all the required files and directories are in place.
    This code was tested only on local execution.
    MAPLE Workflow

Project: Permafrost Discovery Gateway: Mapping Application for Arctic Permafrost Land Environment(MAPLE)
PI      : Chandi Witharana
Author  : Amal Shehan Perera
"""

import os
import shutil
from mpl_config import MPL_Config


def copy_path(*, src, dst, dir_mode=0o777, follow_symlinks: bool = True):
    """
    Copy a source filesystem path to a destination path, creating parent
    directories if they don't exist.
    This tries to use shutil.copy2 until it has required paths created by os.mkdir to copy the files
    Args:
        src: The source filesystem path to copy. This must exist on the
            filesystem.

        dst: The destination to copy to. If the parent directories for this
            path do not exist, we will create them.

        dir_mode: The Unix permissions to set for any newly created
            directories.

        follow_symlinks: Whether to follow symlinks during the copy.

    Returns:
        Returns the destination path.
    """
    try:
        return shutil.copy2(src=src, dst=dst, follow_symlinks=follow_symlinks)
    except FileNotFoundError as exc:
        if exc.filename == dst and exc.filename2 is None:
            parent = os.path.dirname(dst)
            os.makedirs(name=parent, mode=dir_mode, exist_ok=True)
            return shutil.copy2(
                src=src,
                dst=dst,
                follow_symlinks=follow_symlinks,
            )
        raise


def create_dir_path(dir_path):
    """
    Create a required directory path if it does NOT exist.

    Args:
        src: The source filesystem path to copy. This must exist on the

    Returns:
        Returns the destination path.
    """
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        print(dir_path, ":directory already exists")
        pass


def unit_test():
    print(
        copy_path(
            src="../data/final_shp/test_image_01/test_image_01.dbf",
            dst="./final_shp/test_image_01/",
        )
    )
    create_dir_path("./data4/final_shp/test_image_01/")
    create_dir_path("./data4/final_shp/test_image_01/")


def create_maple_dir_structure():
    # Created dir structure that is required for the maple workflow
    # data/
    # ├── cln_data
    # ├── divided_img
    # ├── final_shp
    # ├── input_img_local
    # ├── neighbors
    # ├── output_img
    # ├── output_shp
    # ├── projected_shp
    # └── water_mask
    #     └── temp
    config = MPL_Config()

    create_dir_path(config.INPUT_IMAGE_DIR)
    create_dir_path(config.DIVIDED_IMAGE_DIR)
    create_dir_path(config.OUTPUT_SHP_DIR)
    create_dir_path(config.FINAL_SHP_DIR)
    create_dir_path(config.WATER_MASK_DIR)
    create_dir_path(config.TEMP_W_IMG_DIR)
    create_dir_path(config.OUTPUT_IMAGE_DIR)
    create_dir_path(os.path.join(config.WORKER_ROOT, "neighbors/"))
    create_dir_path(os.path.join(config.WORKER_ROOT, "projected_shp/"))
    create_dir_path(config.CLEAN_DATA_DIR)
    create_dir_path(config.MODEL_DIR)
    create_dir_path(config.MODEL_WEIGHTS_DIR)
    create_dir_path(config.RAY_SHAPEFILES)


if __name__ == "__main__":
    # We call this inside of the "if __name__" block to ensure that
    # the directory is only created when this file is ran as the entry point
    # and not when it is imported by other modules.
    create_maple_dir_structure()
