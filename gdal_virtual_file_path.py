"""
Class that manages the opening and closing of a single gdal file using
gdal's virtual file system.
The pattern is to instantiate the class using a with statement to
create a virtual file path.
For example:

with GDALVirtualFilePath(file_path, file_bytes) as virtual_file_path:
    ....

The with statement manages cleanup. The __enter__ is automatically called
when you enter the with statement block and __exit__ is automatically
called when you exit the block.

See https://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
for more information.
"""
import os
from osgeo import gdal

class GDALVirtualFilePath:
    def __init__(
        self,
        file_path: str,
        file_bytes: bytes
    ):
        vfs_dir_path = '/vsimem/vsidir/'
        file_name = os.path.basename(file_path)
        self.vfs_filename = os.path.join(vfs_dir_path, file_name)
        self.file_bytes = file_bytes

    def __enter__(self) -> str:
        dst = gdal.VSIFOpenL(self.vfs_filename, 'wb+')
        # avoid size limitations by writing in chunks
        # gdal.VSIFWriteL(self.file_bytes, 1, len(self.file_bytes), dst)

        chunk_size = 512 * 1024 * 1024  # 512 MB per chunk
        offset = 0

        while offset < len(self.file_bytes):
            chunk = self.file_bytes[offset:offset + chunk_size]
            bytes_written = gdal.VSIFWriteL(chunk, 1, len(chunk), dst)

            if bytes_written != len(chunk):
                raise RuntimeError(f"Failed to write the full chunk at offset {offset}")

            offset += chunk_size

        gdal.VSIFCloseL(dst)
        return self.vfs_filename

    def __exit__(self, exc_type, exc_value, exc_traceback):
        gdal.Unlink(self.vfs_filename)
