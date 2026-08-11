"""Typed ndarray elements for Zarr v3: the ``ndarray`` data type and the
``vlen-ndarray`` array->bytes codec.

Importing this package registers the data type and codec with zarr-python.
The package also declares both entry-point groups, but only one of them is
live upstream: zarr-python lazy-loads the ``zarr.codecs`` entry point on
demand, while ``zarr.data_type`` entry points are collected into the data type
registry's lazy-load list that zarr-python never flushes (pinned by
``tests/test_registration.py``). So ``import zarr_vlen_ndarray`` is still
required before opening an ndarray store.
"""

from zarr.dtype import data_type_registry
from zarr.registry import register_codec

from zarr_vlen_ndarray.codec import CODEC_NAME, VlenNDArrayCodec
from zarr_vlen_ndarray.dtype import DTYPE_NAME, NDArray, VlenScalar, unbox

try:
    from zarr_vlen_ndarray._version import __version__
except ImportError:  # pragma: no cover - editable install before build
    __version__ = "0.0.0+unknown"

#: Unambiguous alias for :class:`NDArray`, whose name follows zarr-python's
#: ZDType convention (class named after the registered data type) and so
#: collides with ``numpy.typing.NDArray`` at the import site.
NDArrayDType = NDArray

__all__ = [
    "CODEC_NAME",
    "DTYPE_NAME",
    "NDArray",
    "NDArrayDType",
    "VlenNDArrayCodec",
    "VlenScalar",
    "__version__",
    "unbox",
]

# Eager registration (idempotent) so `import zarr_vlen_ndarray` alone is
# sufficient even where entry-point discovery is unavailable.
# The ignore shares the root cause noted on the class: np.ndarray scalars sit
# outside zarr's TBaseScalar bound.
data_type_registry.register(NDArray._zarr_v3_name, NDArray)  # type: ignore[arg-type]
register_codec(CODEC_NAME, VlenNDArrayCodec)
