"""The ``vlen-ndarray`` Zarr v3 data type.

Each array element is a variable-length ndarray: shape ``(n, *inner_shape)``
with ``n`` varying per element and ``inner_shape`` + scalar ``dtype`` fixed by
the data type's configuration. In-memory representation is a NumPy object
array whose cells are ndarrays.

Zarr v3 metadata form::

    {"name": "vlen-ndarray", "configuration": {"dtype": "float32", "inner_shape": [2]}}

Normative spec: SPEC.md in this repository.

Scalar (fill value) representation
----------------------------------
NumPy cannot treat a naked ndarray as a scalar: zarr's fill machinery
(``np.full(..., fill_value=scalar)``, ``out[selection] = scalar``, and
``NDBuffer.all_equal(scalar)``) would broadcast it by shape and crash. The
scalar produced by :meth:`VlenNDArray.cast_scalar` (which zarr stores as the
array's in-memory fill value) is therefore a 0-d object ndarray "box" whose
single item is a :class:`VlenScalar` — an ndarray subclass whose ``==``
compares whole cells and returns a plain bool. NumPy broadcasting unwraps the
box on assignment, so cells filled from the fill value hold a ``VlenScalar``
of shape ``(n, *inner_shape)`` that behaves like a regular ndarray in every
other respect.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast, overload

import numpy as np
from zarr.core.dtype.common import HasObjectCodec
from zarr.dtype import DataTypeValidationError, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat
    from zarr.core.dtype.common import DTypeJSON, DTypeSpec_V2, DTypeSpec_V3

DTYPE_NAME = "vlen-ndarray"

# Fixed-size Zarr v3 core data types permitted as the inner scalar type.
# These names coincide with NumPy dtype names. Variable-size and parameterized
# types (strings, datetimes, ...) are excluded: the wire format requires a
# fixed, known scalar item size.
ALLOWED_SCALAR_DTYPES = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    }
)


class VlenScalar(np.ndarray):
    """A cell value that compares as a scalar.

    Identical to a regular ndarray except that ``==`` / ``!=`` compare the
    whole cell (via ``np.array_equal``) and return a plain bool. This is what
    lets zarr's fill-value equality checks (``NDBuffer.all_equal``) work on
    object arrays whose cells are ndarrays of differing shapes.
    """

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        try:
            return bool(np.array_equal(np.asarray(self), np.asarray(other)))
        except Exception:
            return False

    def __ne__(self, other: object) -> bool:  # type: ignore[override]
        return not self.__eq__(other)

    # ndarrays are unhashable; keep it that way after overriding __eq__.
    __hash__ = None  # type: ignore[assignment]


class _BoxedScalar(np.ndarray):
    """0-d object ndarray carrying one cell, usable as a scalar by zarr.

    zarr stores the fill value inside ``ArraySpec``, which the sharding codec
    hashes (``lru_cache``), and compares between specs — so the box defines
    whole-cell ``==`` (returning bool) and a consistent ``__hash__``.
    """

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        try:
            return bool(np.array_equal(np.asarray(unbox(self)), np.asarray(unbox(other))))
        except Exception:
            return False

    def __ne__(self, other: object) -> bool:  # type: ignore[override]
        return not self.__eq__(other)

    def __hash__(self) -> int:  # type: ignore[override]
        cell = self[()]
        return hash((cell.tobytes(), cell.shape))


def _box(cell: np.ndarray) -> np.ndarray:
    """Wrap a cell in a 0-d object ndarray so NumPy treats it as a scalar."""
    boxed = np.empty((), dtype=object).view(_BoxedScalar)
    boxed[()] = cell.view(VlenScalar)
    return boxed


def unbox(value: object) -> object:
    """Unwrap 0-d object-array boxes around a cell.

    Slice reads return cells directly (NumPy broadcasting distributes the
    boxed value), but *scalar* reads of object-dtype zarr arrays return a 0-d
    object ndarray wrapping the cell — zarr-python does the same for its own
    vlen types (e.g. ``variable_length_bytes``). ``unbox`` recovers the cell
    ndarray; it is a no-op for anything that is not a 0-d object array.
    """
    while isinstance(value, np.ndarray) and value.dtype == object and value.ndim == 0:
        value = value[()]
    return value


# np.ndarray is outside zarr-python's TBaseScalar bound (np.generic | str | bytes),
# which predates container scalars; the runtime contract only requires the scalar
# methods below, so the parametrization is ignored for the type checker.
@dataclass(frozen=True, kw_only=True)
class VlenNDArray(ZDType[np.dtypes.ObjectDType, np.ndarray], HasObjectCodec):  # type: ignore[type-var]
    """Zarr v3 data type for variable-length ndarray elements.

    Parameters
    ----------
    dtype : str
        Zarr v3 core data type name of the inner scalar (e.g. ``"float32"``,
        ``"uint64"``). Must be a fixed-size numeric/boolean type.
    inner_shape : tuple[int, ...]
        Trailing (fixed) dimensions of each element. ``()`` means elements are
        1-D arrays of shape ``(n,)``; ``(2,)`` means elements have shape
        ``(n, 2)``; the leading dimension ``n`` varies per element.
    """

    dtype_cls = np.dtypes.ObjectDType
    _zarr_v3_name: ClassVar[str] = DTYPE_NAME
    # Marks this dtype as requiring an object codec; the id names the
    # `vlen-ndarray` array->bytes codec from this package.
    object_codec_id: ClassVar[str] = DTYPE_NAME

    dtype: str = "float32"
    inner_shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # Normalize (allow np.dtype / np.float32 / lists) then validate.
        name = np.dtype(self.dtype).name
        object.__setattr__(self, "dtype", name)
        if name not in ALLOWED_SCALAR_DTYPES:
            raise ValueError(
                f"dtype must be one of {sorted(ALLOWED_SCALAR_DTYPES)}, got {self.dtype!r}"
            )
        shape = tuple(int(dim) for dim in self.inner_shape)
        object.__setattr__(self, "inner_shape", shape)
        if any(dim < 1 for dim in shape):
            raise ValueError(f"inner_shape dimensions must be >= 1, got {shape}")

    @property
    def item_dtype(self) -> np.dtype[Any]:
        """The inner scalar dtype, explicitly little-endian (the wire byte order)."""
        return np.dtype(self.dtype).newbyteorder("<")

    @property
    def item_nbytes(self) -> int:
        """Bytes per inner item: scalar itemsize times prod(inner_shape)."""
        count = 1
        for dim in self.inner_shape:
            count *= dim
        return self.item_dtype.itemsize * count

    # -- native dtype ------------------------------------------------------

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype[Any]) -> Self:
        # The NumPy object dtype is ambiguous (it backs several Zarr data
        # types), so this data type is never inferred from a native dtype:
        # construct VlenNDArray(...) explicitly.
        raise DataTypeValidationError(
            f"Cannot infer {cls._zarr_v3_name!r} from the native dtype {dtype}; "
            "construct VlenNDArray(dtype=..., inner_shape=...) explicitly."
        )

    def to_native_dtype(self) -> np.dtypes.ObjectDType:
        return self.dtype_cls()

    # -- JSON metadata (Zarr v3 only) ---------------------------------------

    @classmethod
    def _check_json_v2(cls, data: DTypeJSON) -> bool:
        return False

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> bool:
        return (
            isinstance(data, Mapping)
            and set(data.keys()) == {"name", "configuration"}
            and data["name"] == cls._zarr_v3_name
            and isinstance(data["configuration"], Mapping)
            and set(data["configuration"].keys()) == {"dtype", "inner_shape"}
            and isinstance(data["configuration"]["dtype"], str)
            and isinstance(data["configuration"]["inner_shape"], Sequence)
            and not isinstance(data["configuration"]["inner_shape"], (str, bytes))
            and all(isinstance(dim, int) for dim in data["configuration"]["inner_shape"])
        )

    @classmethod
    def _from_json_v2(cls, data: DTypeJSON) -> Self:
        raise DataTypeValidationError(
            f"{cls._zarr_v3_name!r} is a Zarr v3-only data type; got v2 metadata {data!r}"
        )

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> Self:
        if cls._check_json_v3(data):
            config = cast("Mapping[str, Any]", cast("Mapping[str, Any]", data)["configuration"])
            return cls(
                dtype=config["dtype"],
                inner_shape=tuple(config["inner_shape"]),
            )
        raise DataTypeValidationError(
            f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected "
            f'{{"name": "{cls._zarr_v3_name}", "configuration": '
            f'{{"dtype": "<scalar>", "inner_shape": [...]}}}}'
        )

    @overload
    def to_json(self, zarr_format: Literal[2]) -> DTypeSpec_V2: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> DTypeSpec_V3: ...

    def to_json(self, zarr_format: ZarrFormat) -> DTypeSpec_V2 | DTypeSpec_V3:
        if zarr_format == 3:
            return {
                "name": self._zarr_v3_name,
                "configuration": {
                    "dtype": self.dtype,
                    "inner_shape": list(self.inner_shape),
                },
            }
        raise ValueError(
            f"{self._zarr_v3_name!r} is a Zarr v3-only data type; "
            f"cannot serialize to zarr_format={zarr_format}"
        )

    # -- cell coercion (plain arrays) ----------------------------------------

    def coerce_cell(self, data: object) -> np.ndarray:
        """Coerce a cell value to a C-contiguous little-endian ``(n, *inner_shape)`` array.

        This is the element normalization the codec applies before serializing
        each cell (``coerce_cell(cell).tobytes()`` is the cell's wire payload).
        """
        if isinstance(data, np.ndarray) and data.dtype == object and data.ndim == 0:
            data = data[()]  # unwrap a boxed scalar
        if not isinstance(data, (np.ndarray, list, tuple)):
            raise TypeError(
                f"Cannot convert object {data!r} with type {type(data)} to a scalar "
                f"compatible with the data type {self}."
            )
        arr = np.asarray(data, dtype=self.item_dtype)
        if arr.size == 0:
            return arr.reshape((0, *self.inner_shape))
        expected_ndim = 1 + len(self.inner_shape)
        if arr.ndim == expected_ndim and arr.shape[1:] == self.inner_shape:
            return np.ascontiguousarray(arr)
        raise TypeError(
            f"Cannot convert array of shape {arr.shape} to a scalar of the data type "
            f"{self}: expected shape (n, {', '.join(map(str, self.inner_shape))})."
        )

    def payload_to_cell(self, payload: bytes) -> np.ndarray:
        """Decode one element's raw little-endian payload bytes to an ndarray.

        The returned array is a read-only view over ``payload``.
        """
        if len(payload) % self.item_nbytes != 0:
            raise ValueError(
                f"Payload of {len(payload)} bytes is not a whole number of "
                f"{self.item_nbytes}-byte items for the data type {self}."
            )
        return np.frombuffer(payload, dtype=self.item_dtype).reshape((-1, *self.inner_shape))

    # -- scalars (boxed; see module docstring) --------------------------------

    def _check_scalar(self, data: object) -> bool:
        return isinstance(data, (np.ndarray, list, tuple))

    def cast_scalar(self, data: object) -> np.ndarray:
        """Cast to the boxed scalar form zarr uses as the array's fill value."""
        if not self._check_scalar(data):
            raise TypeError(
                f"Cannot convert object {data!r} with type {type(data)} to a scalar "
                f"compatible with the data type {self}."
            )
        return _box(self.coerce_cell(data))

    def default_scalar(self) -> np.ndarray:
        return _box(np.empty((0, *self.inner_shape), dtype=self.item_dtype))

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        # Scalars (fill values) serialize as base64 of the raw little-endian
        # payload bytes, mirroring zarr's variable-length bytes data type.
        return base64.b64encode(self.coerce_cell(data).tobytes()).decode("ascii")

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.ndarray:
        if isinstance(data, str):
            payload = base64.b64decode(data.encode("ascii"))
            return _box(self.payload_to_cell(payload))
        raise TypeError(f"Invalid type: {data}. Expected a base64-encoded string.")
