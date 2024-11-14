import fsspec
import dask.dataframe as dd
import zipfile
import fsspec
from fsspec.archive import AbstractArchiveFileSystem


class ZipFileSystem(AbstractArchiveFileSystem):
    protocol = "tzip"

    def __init__(
        self,
        fo="",
        **kwargs,
    ):
        super().__init__(self, **kwargs)
        fo = fsspec.open(fo, mode='rb')
        self.of = fo
        self.fo = fo.__enter__()
        self.zip = zipfile.ZipFile(self.fo, mode='r')
        self.dir_cache = None

    @classmethod
    def _strip_protocol(cls, path):
        return super()._strip_protocol(path).lstrip("/")

    def __del__(self):
        if hasattr(self, "zip"):
            self.close()
            del self.zip

    def close(self):
        self.zip.close()

    def _open(
        self,
        path,
        **kwargs,
    ):
        path = self._strip_protocol(path)
        out = self.zip.open(path, 'r')
        return out

fsspec.register_implementation('tzip', ZipFileSystem)
with fsspec.open('tzip://a.csv', fo='a.zip') as f:
    print(f.read(1))
df = dd.read_csv('tzip://a.csv', storage_options={'fo':'a.zip'})
print(df.head())
