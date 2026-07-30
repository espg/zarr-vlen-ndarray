"""Typed variable-length ndarray data type (``vlen-ndarray``) for Zarr v3.

Importing this package registers the data type and codec with zarr-python.
With the package *installed*, importing it is not even necessary: zarr-python
discovers both through the ``zarr.data_type`` and ``zarr.codecs`` entry-point
groups, so a plain ``zarr.open`` resolves vlen-ndarray stores.
"""

from zarr.dtype import data_type_registry
from zarr.registry import register_codec

from zarr_vlen_ndarray.codec import CODEC_NAME, VlenNDArrayCodec
from zarr_vlen_ndarray.dtype import DTYPE_NAME, VlenNDArray, VlenScalar, unbox

try:
    from zarr_vlen_ndarray._version import __version__
except ImportError:  # pragma: no cover - editable install before build
    __version__ = "0.0.0+unknown"

__all__ = [
    "CODEC_NAME",
    "DTYPE_NAME",
    "VlenNDArray",
    "VlenNDArrayCodec",
    "VlenScalar",
    "__version__",
    "unbox",
]

# Eager registration (idempotent) so `import zarr_vlen_ndarray` alone is
# sufficient even where entry-point discovery is unavailable.
# The ignore shares the root cause noted on the class: np.ndarray scalars sit
# outside zarr's TBaseScalar bound.
data_type_registry.register(VlenNDArray._zarr_v3_name, VlenNDArray)  # type: ignore[arg-type]
register_codec(CODEC_NAME, VlenNDArrayCodec)
