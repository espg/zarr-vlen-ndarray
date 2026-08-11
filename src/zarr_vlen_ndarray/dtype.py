"""The ``ndarray`` Zarr v3 data type.

Each array element is itself an ndarray of a fixed scalar ``dtype`` whose
shape matches a configured pattern in which ``None`` (JSON ``null``) marks a
variable-length dimension: ``shape=(None, 2)`` describes elements of shape
``(n, 2)`` with ``n`` varying per element, ``shape=(3, 2)`` describes fixed
``(3, 2)`` elements. In-memory representation is a NumPy object array whose
cells are ndarrays.

Zarr v3 metadata form::

    {"name": "ndarray", "configuration": {"dtype": "float32", "shape": [null, 2]}}

The data type accepts the full shape-pattern grammar; each paired
``array -> bytes`` codec constrains the patterns it serves (the packaged
``vlen-ndarray`` codec serves exactly one variable dimension, in the leading
position). Normative spec: SPEC.md in this repository.

Scalar (fill value) representation
----------------------------------
NumPy cannot treat a naked ndarray as a scalar: zarr's fill machinery
(``np.full(..., fill_value=scalar)``, ``out[selection] = scalar``, and
``NDBuffer.all_equal(scalar)``) would broadcast it by shape and crash. The
scalar produced by :meth:`NDArray.cast_scalar` (which zarr stores as the
array's in-memory fill value) is therefore a 0-d object ndarray "box" whose
single item is a :class:`VlenScalar` — an ndarray subclass whose ``==``
compares whole cells and returns a plain bool. NumPy broadcasting unwraps the
box on assignment, so cells filled from the fill value hold a ``VlenScalar``
of the fill element's shape that behaves like a regular ndarray in every
other respect.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast, overload

import numpy as np
from zarr.core.dtype.common import HasObjectCodec
from zarr.dtype import ZDType

# zarr >= 3.2 canonical location for DataTypeValidationError; importing it from
# zarr.dtype there goes through a deprecation shim, which also breaks mypy's
# view of the symbol ("object" not callable). zarr 3.1.x (the floor) has it in
# zarr.dtype only.
if TYPE_CHECKING:
    from zarr.errors import DataTypeValidationError
else:
    try:
        from zarr.errors import DataTypeValidationError
    except ImportError:  # pragma: no cover - zarr 3.1.x
        from zarr.dtype import DataTypeValidationError

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat
    from zarr.core.dtype.common import DTypeJSON, DTypeSpec_V2, DTypeSpec_V3

DTYPE_NAME = "ndarray"

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


def _normalize_scalar_dtype(value: object) -> str:
    """Normalize a user-supplied inner scalar dtype to its Zarr v3 core name.

    Accepts anything ``np.dtype`` understands — a core data type name, a NumPy
    dtype object, a NumPy scalar type, or a byte-order-agnostic/little-endian
    type string — and returns the canonical core name.

    Explicitly big-endian spellings (``">f4"``) are rejected rather than
    normalized: the wire format is always little-endian, so silently returning
    ``float32`` would give the caller the opposite byte order from the one
    requested, and ``">f4"`` is not a value the registered ``schema.json``
    accepts for ``configuration.dtype``.
    """
    if isinstance(value, str) and value in ALLOWED_SCALAR_DTYPES:
        return value
    try:
        np_dtype = np.dtype(value)  # type: ignore[call-overload]
    except TypeError as exc:
        raise ValueError(
            f"dtype must be one of {sorted(ALLOWED_SCALAR_DTYPES)}, got {value!r}"
        ) from exc
    if np_dtype.byteorder == ">":
        raise ValueError(
            f"dtype must not request big-endian byte order, got {value!r}: "
            "ndarray elements are always encoded little-endian."
        )
    name = np_dtype.name
    if name not in ALLOWED_SCALAR_DTYPES:
        raise ValueError(f"dtype must be one of {sorted(ALLOWED_SCALAR_DTYPES)}, got {value!r}")
    return name


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
class NDArray(ZDType[np.dtypes.ObjectDType, np.ndarray], HasObjectCodec):  # type: ignore[type-var]
    """Zarr v3 data type for ndarray elements.

    Parameters
    ----------
    dtype : str
        Zarr v3 core data type name of the inner scalar (e.g. ``"float32"``,
        ``"uint64"``). Must be a fixed-size numeric/boolean type.
    shape : tuple[int | None, ...]
        Element shape pattern: one member per element dimension, each either
        an integer ``>= 1`` (fixed extent) or ``None`` (variable extent, may
        differ between elements; JSON ``null``). ``(None,)`` means elements
        are 1-D arrays of shape ``(n,)``; ``(None, 2)`` means elements have
        shape ``(n, 2)``; ``(3, 2)`` means fixed ``(3, 2)`` elements. Must be
        non-empty.
    """

    dtype_cls = np.dtypes.ObjectDType
    _zarr_v3_name: ClassVar[str] = DTYPE_NAME
    # Marks this dtype as requiring an object codec; the id names the
    # `vlen-ndarray` array->bytes codec from this package.
    object_codec_id: ClassVar[str] = "vlen-ndarray"

    dtype: str = "float32"
    shape: tuple[int | None, ...] = (None,)

    def __post_init__(self) -> None:
        # Normalize (allow np.dtype / np.float32 / lists) then validate.
        object.__setattr__(self, "dtype", _normalize_scalar_dtype(self.dtype))
        if len(self.shape) == 0:
            raise ValueError(
                "shape must have at least one dimension: a zero-dimensional element "
                "is a plain scalar, so use the scalar data type directly."
            )
        if any(isinstance(dim, bool) for dim in self.shape):
            # bool is a subclass of int; True would silently become 1.
            raise ValueError(
                f"shape dimensions must be integers or None, got {tuple(self.shape)!r}"
            )
        shape = tuple(None if dim is None else int(dim) for dim in self.shape)
        object.__setattr__(self, "shape", shape)
        if any(dim is not None and dim < 1 for dim in shape):
            raise ValueError(f"fixed shape dimensions must be >= 1, got {shape}")

    @property
    def variable_axes(self) -> tuple[int, ...]:
        """Indices of the variable (``None``) dimensions of the pattern."""
        return tuple(i for i, dim in enumerate(self.shape) if dim is None)

    @property
    def item_dtype(self) -> np.dtype[Any]:
        """The inner scalar dtype, explicitly little-endian (the wire byte order)."""
        return np.dtype(self.dtype).newbyteorder("<")

    @property
    def fixed_nbytes(self) -> int:
        """Bytes per unit of variable extent: scalar itemsize times prod(fixed dims).

        For a pattern with no variable dimension this is the exact element
        byte size; with exactly one variable dimension it is the byte size of
        one slice along that dimension (the divisor in the codec's
        ``n = payload_bytes / fixed_nbytes`` inference rule).
        """
        count = 1
        for dim in self.shape:
            if dim is not None:
                count *= dim
        return self.item_dtype.itemsize * count

    @property
    def _empty_shape(self) -> tuple[int, ...]:
        """The element shape with every variable extent set to 0."""
        return tuple(0 if dim is None else dim for dim in self.shape)

    def _matches_pattern(self, shape: tuple[int, ...]) -> bool:
        return len(shape) == len(self.shape) and all(
            expected is None or actual == expected
            for expected, actual in zip(self.shape, shape, strict=True)
        )

    # -- native dtype ------------------------------------------------------

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype[Any]) -> Self:
        # The NumPy object dtype is ambiguous (it backs several Zarr data
        # types), so this data type is never inferred from a native dtype:
        # construct NDArray(...) explicitly.
        raise DataTypeValidationError(
            f"Cannot infer {cls._zarr_v3_name!r} from the native dtype {dtype}; "
            "construct NDArray(dtype=..., shape=...) explicitly."
        )

    def to_native_dtype(self) -> np.dtypes.ObjectDType:
        return self.dtype_cls()

    # -- JSON metadata (Zarr v3 only) ---------------------------------------

    @classmethod
    def _check_json_v2(cls, data: DTypeJSON) -> bool:
        return False

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> bool:
        # This mirrors registry/data-types/ndarray/schema.json, so that
        # metadata the registered schema rejects never reaches the
        # constructor: anything this returns False for is reported by
        # _from_json_v3 as a DataTypeValidationError, which is the only
        # exception zarr's data type resolution loop catches.
        #
        # One deliberate divergence: JSON Schema's "type": "integer" also
        # admits zero-fractional spellings such as 2.0, which JSON Schema has
        # no way to exclude. The registry README states normatively that fixed
        # dimensions MUST be written as JSON integers, and this checker
        # enforces that (see tests/test_dtype.py's [None, 1.0] case).
        if not (
            isinstance(data, Mapping)
            and set(data.keys()) == {"name", "configuration"}
            and data["name"] == cls._zarr_v3_name
            and isinstance(data["configuration"], Mapping)
            and set(data["configuration"].keys()) == {"dtype", "shape"}
        ):
            return False
        config = data["configuration"]
        dtype = config["dtype"]
        shape = config["shape"]
        return (
            isinstance(dtype, str)
            and dtype in ALLOWED_SCALAR_DTYPES
            and isinstance(shape, Sequence)
            and not isinstance(shape, (str, bytes))
            and len(shape) >= 1
            # each member is null (variable) or an integer >= 1; bool is a
            # subclass of int, but the schema rejects it.
            and all(
                dim is None or (isinstance(dim, int) and not isinstance(dim, bool) and dim >= 1)
                for dim in shape
            )
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
                shape=tuple(config["shape"]),
            )
        raise DataTypeValidationError(
            f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected "
            f'{{"name": "{cls._zarr_v3_name}", "configuration": '
            f'{{"dtype": "<scalar>", "shape": [...]}}}}'
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
                    # None members serialize to JSON null (variable dims).
                    "shape": list(self.shape),
                },
            }
        raise ValueError(
            f"{self._zarr_v3_name!r} is a Zarr v3-only data type; "
            f"cannot serialize to zarr_format={zarr_format}"
        )

    # -- cell coercion (plain arrays) ----------------------------------------

    def coerce_cell(self, data: object) -> np.ndarray:
        """Coerce a cell value to a C-contiguous little-endian pattern-matching array.

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
        try:
            arr = np.asarray(data, dtype=self.item_dtype)
        except ValueError as exc:
            # e.g. a ragged list-of-lists: NumPy raises "setting an array
            # element with a sequence". Callers of coerce_cell/cast_scalar are
            # promised a TypeError for values that are not valid cells.
            raise TypeError(
                f"Cannot convert object {data!r} with type {type(data)} to a scalar "
                f"compatible with the data type {self}: expected a rectangular array "
                f"matching the pattern {list(self.shape)} (None dimensions are variable)."
            ) from exc
        if self._matches_pattern(arr.shape):
            return np.ascontiguousarray(arr)
        if arr.shape == (0,) and self.variable_axes:
            # An empty flat sequence (``[]``, ``np.empty(0)``) is an
            # unambiguous spelling of the empty element when the pattern has a
            # variable dimension. Any other zero-sized shape is a caller error
            # and is reported below rather than being silently reshaped.
            return arr.reshape(self._empty_shape)
        raise TypeError(
            f"Cannot convert array of shape {arr.shape} to a scalar of the data type "
            f"{self}: expected shape matching the pattern {list(self.shape)} "
            "(None dimensions are variable)."
        )

    def payload_to_cell(self, payload: bytes) -> np.ndarray:
        """Decode one element's raw little-endian payload bytes to an ndarray.

        The returned array is a read-only view over ``payload``. The payload
        length determines the variable extents: with no variable dimension
        the length must equal the fixed element size exactly; with one it
        must be a whole multiple of :attr:`fixed_nbytes` (the codec inference
        rule); with two or more only the empty payload is unambiguous.
        """
        if len(payload) % self.fixed_nbytes != 0:
            raise ValueError(
                f"Payload of {len(payload)} bytes is not a whole number of "
                f"{self.fixed_nbytes}-byte items for the data type {self}."
            )
        n = len(payload) // self.fixed_nbytes  # product of the variable extents
        variable_axes = self.variable_axes
        if not variable_axes:
            if n != 1:
                raise ValueError(
                    f"Payload of {len(payload)} bytes does not match the fixed element "
                    f"size of {self.fixed_nbytes} bytes for the data type {self}."
                )
            shape = cast("tuple[int, ...]", self.shape)
        elif len(variable_axes) == 1:
            shape = tuple(n if dim is None else dim for dim in self.shape)
        else:
            if n != 0:
                raise ValueError(
                    f"Payload of {len(payload)} bytes is ambiguous for the data type "
                    f"{self}: a pattern with multiple variable dimensions only has a "
                    "well-defined shape for the empty payload."
                )
            shape = self._empty_shape
        return np.frombuffer(payload, dtype=self.item_dtype).reshape(shape)

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
        # With a variable dimension: the empty element (variable extents 0).
        # All-fixed pattern: the zero-filled element. Matches the registered
        # fill-value recommendation.
        if self.variable_axes:
            return _box(np.empty(self._empty_shape, dtype=self.item_dtype))
        return _box(np.zeros(cast("tuple[int, ...]", self.shape), dtype=self.item_dtype))

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        # Scalars (fill values) serialize as base64 of the raw little-endian
        # payload bytes, mirroring zarr's variable-length bytes data type.
        return base64.b64encode(self.coerce_cell(data).tobytes()).decode("ascii")

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.ndarray:
        if isinstance(data, str):
            payload = base64.b64decode(data.encode("ascii"))
            return _box(self.payload_to_cell(payload))
        raise TypeError(f"Invalid type: {data}. Expected a base64-encoded string.")
