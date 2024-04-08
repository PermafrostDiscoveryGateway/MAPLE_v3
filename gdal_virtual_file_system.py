import os
from osgeo import gdal

class GDALVirtualFileSystem:
    def __init__(
        self,
        file_path: str,
        file_bytes: bytes
    ):
        vfs_dir_path = '/vsimem/vsidir/'
        file_name = os.path.basename(file_path)
        self.vfs_filename = os.path.join(vfs_dir_path, file_name)
        self.file_bytes = file_bytes

    def create_virtual_file(self) -> str:
        dst = gdal.VSIFOpenL(self.vfs_filename, 'wb+')
        gdal.VSIFWriteL(self.file_bytes, 1, len(self.file_bytes), dst)
        gdal.VSIFCloseL(dst)
        return self.vfs_filename

    def close_virtual_file(self):
        gdal.Unlink(self.vfs_filename)
