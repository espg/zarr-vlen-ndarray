"""Tests for the NDArray ZDType: metadata JSON, scalars, validation."""

import json

import numpy as np
import pytest
import zarr

from zarr_vlen_ndarray import NDArray, VlenNDArrayCodec, VlenScalar, unbox

# resolved from zarr.errors on zarr >= 3.2 and zarr.dtype on the 3.1.x floor;
# importing it from zarr.dtype directly goes through a deprecation shim
from zarr_vlen_ndarray.dtype import DataTypeValidationError

V3_JSON = {"name": "ndarray", "configuration": {"dtype": "float32", "shape": [None, 2]}}


def test_to_json_v3():
    zd = NDArray(dtype="float32", shape=(None, 2))
    assert zd.to_json(zarr_format=3) == V3_JSON


def test_from_json_v3_roundtrip():
    zd = NDArray._from_json_v3(V3_JSON)
    assert zd == NDArray(dtype="float32", shape=(None, 2))
    assert zd.shape == (None, 2)
    assert isinstance(zd.shape, tuple)


def test_json_roundtrip_flat_shape():
    zd = NDArray(dtype="uint64", shape=(None,))
    data = zd.to_json(zarr_format=3)
    assert data == {"name": "ndarray", "configuration": {"dtype": "uint64", "shape": [None]}}
    assert NDArray._from_json_v3(data) == zd


@pytest.mark.parametrize(
    "shape",
    [(None, 3, 2), (3, 2), (None, None), (2, None)],
    ids=["multi_dim_trailing", "all_fixed", "two_variable", "non_leading_variable"],
)
def test_json_roundtrip_pattern_families(shape):
    """The full shape-pattern grammar round-trips through metadata JSON."""
    zd = NDArray(dtype="float32", shape=shape)
    data = zd.to_json(zarr_format=3)
    assert data == {"name": "ndarray", "configuration": {"dtype": "float32", "shape": list(shape)}}
    assert NDArray._from_json_v3(data) == zd


@pytest.mark.parametrize(
    "bad",
    [
        "ndarray",
        {"name": "ndarray"},
        {"name": "other", "configuration": {"dtype": "float32", "shape": [None]}},
        {"name": "ndarray", "configuration": {"dtype": "float32"}},
        {"name": "ndarray", "configuration": {"dtype": "float32", "shape": [None, 2], "x": 1}},
        {"name": "ndarray", "configuration": {"dtype": "float32", "shape": "2"}},
        {"name": "ndarray", "configuration": {"dtype": 32, "shape": [None, 2]}},
        # the registered schema.json enumerates the core data type names, so a
        # byte-order-prefixed spelling is not conformant metadata
        {"name": "ndarray", "configuration": {"dtype": ">f4", "shape": [None, 2]}},
        {"name": "ndarray", "configuration": {"dtype": "<f4", "shape": [None, 2]}},
        # a non-core / nonexistent scalar name must not reach NumPy
        {"name": "ndarray", "configuration": {"dtype": "banana", "shape": [None]}},
        {"name": "ndarray", "configuration": {"dtype": "str", "shape": [None]}},
        # shape members must be null or integers >= 1; bool is a subclass of int
        {"name": "ndarray", "configuration": {"dtype": "float32", "shape": [None, 0]}},
        {"name": "ndarray", "configuration": {"dtype": "float32", "shape": [None, -1]}},
        {"name": "ndarray", "configuration": {"dtype": "float32", "shape": [None, True]}},
        # deliberate divergence from schema.json: JSON Schema's "type":
        # "integer" admits a zero-fractional spelling like 1.0 and cannot
        # exclude it, so the registry README states normatively that fixed
        # dimensions MUST be written as JSON integers, and this checker is
        # what enforces it.
        {"name": "ndarray", "configuration": {"dtype": "float32", "shape": [None, 1.0]}},
        # the empty pattern (0-d elements) is not permitted
        {"name": "ndarray", "configuration": {"dtype": "float32", "shape": []}},
        {"name": "ndarray", "configuration": {"dtype": ["float32"], "shape": [None]}},
        # the old (pre-restructure) vlen-ndarray grammar must not parse
        {"name": "vlen-ndarray", "configuration": {"dtype": "float32", "inner_shape": [2]}},
        {"name": "ndarray", "configuration": {"dtype": "float32", "inner_shape": [2]}},
    ],
)
def test_check_json_v3_rejects(bad):
    assert not NDArray._check_json_v3(bad)
    # every rejection must surface as DataTypeValidationError: it is the only
    # exception zarr's data type resolution loop catches, so a TypeError or a
    # bare ValueError escaping from NumPy would abort resolution entirely
    # instead of falling through to "No Zarr data type found ...".
    with pytest.raises(DataTypeValidationError):
        NDArray._from_json_v3(bad)


def test_v2_unsupported():
    zd = NDArray(dtype="float32", shape=(None, 2))
    assert not NDArray._check_json_v2({"name": "|O", "object_codec_id": None})
    with pytest.raises(DataTypeValidationError):
        NDArray._from_json_v2({"name": "|O", "object_codec_id": None})
    with pytest.raises(ValueError, match="v3-only"):
        zd.to_json(zarr_format=2)


def test_dtype_normalization_and_validation():
    assert NDArray(dtype=np.dtype("float32"), shape=[None, 2]).dtype == "float32"
    assert NDArray(dtype="<f4").dtype == "float32"
    assert NDArray(dtype=np.float32).dtype == "float32"
    with pytest.raises(ValueError, match="dtype must be one of"):
        NDArray(dtype="str")
    with pytest.raises(ValueError, match="dtype must be one of"):
        NDArray(dtype="banana")
    with pytest.raises(ValueError, match="fixed shape dimensions must be >= 1"):
        NDArray(dtype="float32", shape=(None, 0))
    with pytest.raises(ValueError, match="shape dimensions must be integers or None"):
        NDArray(dtype="float32", shape=(None, True))
    with pytest.raises(ValueError, match="at least one dimension"):
        NDArray(dtype="float32", shape=())


def test_variable_axes():
    assert NDArray(dtype="float32", shape=(None, 2)).variable_axes == (0,)
    assert NDArray(dtype="float32", shape=(3, 2)).variable_axes == ()
    assert NDArray(dtype="float32", shape=(None, None)).variable_axes == (0, 1)
    assert NDArray(dtype="float32", shape=(2, None)).variable_axes == (1,)


@pytest.mark.parametrize("spelling", [">f4", ">u8", np.dtype(">f4")])
def test_rejects_big_endian_dtype(spelling):
    """Big-endian spellings must not be normalized to their little-endian name.

    The wire format is always little-endian, so silently accepting `">f4"`
    would hand a caller who explicitly asked for big-endian the opposite byte
    order — and `">f4"` is not a value the registered schema.json accepts.
    """
    with pytest.raises(ValueError, match="big-endian"):
        NDArray(dtype=spelling)


def test_item_dtype_stays_little_endian():
    """Pinning little-endian on the wire is a separate, intended mechanism."""
    zd = NDArray(dtype="float64", shape=(None, 2))
    assert zd.item_dtype.byteorder in ("<", "=", "|")
    assert zd.item_dtype == np.dtype("<f8")


def test_from_native_dtype_never_infers():
    with pytest.raises(DataTypeValidationError, match="construct NDArray"):
        NDArray.from_native_dtype(np.dtype("O"))


def test_fixed_nbytes():
    zd = NDArray(dtype="uint64", shape=(None,))
    assert zd.item_dtype == np.dtype("<u8")
    assert zd.fixed_nbytes == 8
    assert NDArray(dtype="float32", shape=(None, 2)).fixed_nbytes == 8
    assert NDArray(dtype="float32", shape=(None, 3, 2)).fixed_nbytes == 24
    # all-fixed: the exact element size
    assert NDArray(dtype="int16", shape=(3, 2)).fixed_nbytes == 12
    # multiple variable dims: only the fixed dims contribute
    assert NDArray(dtype="float32", shape=(None, None, 2)).fixed_nbytes == 8


def test_coerce_cell():
    zd = NDArray(dtype="float32", shape=(None, 2))
    cell = zd.coerce_cell([[1.0, 2.0], [3.0, 4.0]])
    assert cell.shape == (2, 2)
    assert cell.dtype == np.dtype("<f4")
    assert cell.flags.c_contiguous
    # a flat empty sequence is an unambiguous spelling of the empty cell
    assert zd.coerce_cell(np.empty(0)).shape == (0, 2)
    assert zd.coerce_cell([]).shape == (0, 2)
    assert zd.coerce_cell(np.empty((0, 2), dtype="<f4")).shape == (0, 2)
    with pytest.raises(TypeError, match="expected shape matching"):
        zd.coerce_cell(np.ones((2, 3), dtype="<f4"))
    with pytest.raises(TypeError):
        zd.coerce_cell("nope")


def test_coerce_cell_ragged_input_is_a_type_error():
    """A ragged list-of-lists must surface as the TypeError coerce_cell and
    cast_scalar promise, not as NumPy's "setting an array element with a
    sequence" ValueError."""
    zd = NDArray(dtype="float32", shape=(None, 2))
    with pytest.raises(TypeError, match="rectangular array"):
        zd.coerce_cell([[1.0, 2.0], [3.0]])
    with pytest.raises(TypeError, match="rectangular array"):
        zd.cast_scalar([[1.0, 2.0], [3.0]])


def test_coerce_cell_multi_dim_trailing():
    zd = NDArray(dtype="float32", shape=(None, 3, 2))
    cell = zd.coerce_cell(np.ones((4, 3, 2), dtype="<f4"))
    assert cell.shape == (4, 3, 2)
    assert zd.coerce_cell([]).shape == (0, 3, 2)
    with pytest.raises(TypeError, match="expected shape matching"):
        zd.coerce_cell(np.ones((4, 2, 3), dtype="<f4"))


def test_coerce_cell_all_fixed():
    zd = NDArray(dtype="float32", shape=(3, 2))
    assert zd.coerce_cell(np.ones((3, 2), dtype="<f4")).shape == (3, 2)
    # no variable dimension: the empty-flat-sequence spelling has no target
    with pytest.raises(TypeError, match="expected shape matching"):
        zd.coerce_cell([])
    with pytest.raises(TypeError, match="expected shape matching"):
        zd.coerce_cell(np.ones((2, 2), dtype="<f4"))


def test_coerce_cell_multiple_variable_dims():
    zd = NDArray(dtype="float32", shape=(None, None))
    assert zd.coerce_cell(np.ones((4, 5), dtype="<f4")).shape == (4, 5)
    assert zd.coerce_cell([]).shape == (0, 0)


@pytest.mark.parametrize("shape", [(0, 5), (0, 2, 2), (2, 0), (0, 0)])
def test_coerce_cell_rejects_empty_with_incompatible_shape(shape):
    """An empty array whose trailing dims disagree is a caller bug, not a cell.

    The size-0 fast path must not short-circuit ahead of the shape check.
    """
    zd = NDArray(dtype="float32", shape=(None, 2))
    with pytest.raises(TypeError, match="expected shape matching"):
        zd.coerce_cell(np.empty(shape, dtype="<f4"))


def test_payload_to_cell():
    zd = NDArray(dtype="float32", shape=(None, 2))
    cell = np.arange(6, dtype="<f4").reshape(3, 2)
    out = zd.payload_to_cell(cell.tobytes())
    assert np.array_equal(out, cell)
    assert zd.payload_to_cell(b"").shape == (0, 2)
    with pytest.raises(ValueError, match="whole number"):
        zd.payload_to_cell(b"\x00" * 7)


def test_payload_to_cell_multi_dim_trailing():
    zd = NDArray(dtype="float32", shape=(None, 3, 2))
    cell = np.arange(12, dtype="<f4").reshape(2, 3, 2)
    assert np.array_equal(zd.payload_to_cell(cell.tobytes()), cell)
    assert zd.payload_to_cell(b"").shape == (0, 3, 2)


def test_payload_to_cell_all_fixed():
    zd = NDArray(dtype="int16", shape=(3, 2))
    cell = np.arange(6, dtype="<i2").reshape(3, 2)
    assert np.array_equal(zd.payload_to_cell(cell.tobytes()), cell)
    # exact element size required: no variable extent to absorb multiples
    with pytest.raises(ValueError, match="fixed element size"):
        zd.payload_to_cell(cell.tobytes() * 2)
    with pytest.raises(ValueError, match="fixed element size"):
        zd.payload_to_cell(b"")


def test_payload_to_cell_multiple_variable_dims():
    zd = NDArray(dtype="float32", shape=(None, None))
    # only the empty payload has a well-defined shape
    assert zd.payload_to_cell(b"").shape == (0, 0)
    with pytest.raises(ValueError, match="ambiguous"):
        zd.payload_to_cell(b"\x00" * 8)


def test_scalar_json_roundtrip():
    zd = NDArray(dtype="float32", shape=(None, 2))
    cell = np.arange(4, dtype="<f4").reshape(2, 2)
    encoded = zd.to_json_scalar(cell, zarr_format=3)
    assert isinstance(encoded, str)
    decoded = zd.from_json_scalar(encoded, zarr_format=3)
    assert np.array_equal(unbox(decoded), cell)
    # the default fill is the empty cell, base64 ""
    assert zd.to_json_scalar(zd.default_scalar(), zarr_format=3) == ""
    with pytest.raises(TypeError, match="base64"):
        zd.from_json_scalar(123, zarr_format=3)


def test_default_scalar_by_pattern_family():
    # variable dimension present: the empty element
    assert np.asarray(unbox(NDArray(dtype="float32", shape=(None, 2)).default_scalar())).shape == (
        0,
        2,
    )
    assert np.asarray(
        unbox(NDArray(dtype="float32", shape=(None, None)).default_scalar())
    ).shape == (0, 0)
    # all-fixed: the zero-filled element
    fixed_default = np.asarray(unbox(NDArray(dtype="int16", shape=(3, 2)).default_scalar()))
    assert fixed_default.shape == (3, 2)
    assert not fixed_default.any()


def test_cast_scalar_is_boxed():
    zd = NDArray(dtype="float32", shape=(None, 2))
    boxed = zd.cast_scalar(np.ones((2, 2), dtype="<f4"))
    assert boxed.ndim == 0 and boxed.dtype == object
    cell = unbox(boxed)
    assert isinstance(cell, VlenScalar)
    assert cell.shape == (2, 2)
    # casting a boxed scalar is idempotent
    reboxed = zd.cast_scalar(boxed)
    assert np.array_equal(np.asarray(unbox(reboxed)), np.asarray(cell))
    with pytest.raises(TypeError, match="Cannot convert"):
        zd.cast_scalar(1.5)


def test_vlen_scalar_equality_semantics():
    a = np.arange(6, dtype="<f4").reshape(3, 2).view(VlenScalar)
    assert a == np.arange(6, dtype="<f4").reshape(3, 2)
    assert a != np.zeros((3, 2), dtype="<f4")
    # differing shapes compare unequal instead of raising
    assert a != np.zeros((0, 2), dtype="<f4")
    empty = np.empty((0, 2), dtype="<f4").view(VlenScalar)
    assert empty == np.empty((0, 2), dtype="<f4")


def test_ndarray_dtype_alias():
    """`NDArray` follows zarr-python's ZDType naming convention but collides
    with `numpy.typing.NDArray`; the alias is the unambiguous import."""
    import zarr_vlen_ndarray

    assert zarr_vlen_ndarray.NDArrayDType is NDArray
    assert "NDArrayDType" in zarr_vlen_ndarray.__all__


def test_unbox_passthrough():
    x = np.ones((2, 2))
    assert unbox(x) is x
    assert unbox("s") == "s"


# -- non-conformant metadata seen through zarr ------------------------------


def _store_with_config(tmp_path, config):
    """Build a valid store, then rewrite `data_type.configuration` in place."""
    path = tmp_path / "a"
    zarr.create_array(
        store=path,
        shape=(2,),
        chunks=(2,),
        dtype=NDArray(dtype="float32", shape=(None, 2)),
        serializer=VlenNDArrayCodec(),
    )
    meta_path = path / "zarr.json"
    meta = json.loads(meta_path.read_text())
    meta["data_type"]["configuration"] = config
    meta_path.write_text(json.dumps(meta))
    return path


@pytest.mark.parametrize(
    "config",
    [
        {"dtype": ">f4", "shape": [None, 2]},
        {"dtype": "banana", "shape": [None, 2]},
        {"dtype": "float32", "shape": [None, 0]},
        {"dtype": "float32", "shape": [None, True]},
        {"dtype": "float32", "shape": []},
        {"dtype": "float32", "inner_shape": [2]},
    ],
    ids=["big_endian", "unknown_dtype", "zero_dim", "bool_dim", "empty_shape", "old_grammar"],
)
def test_open_array_rejects_non_conformant_metadata(tmp_path, config):
    """Metadata the registered schema rejects must not open, and must fail
    with zarr's own "no matching data type" ValueError rather than a NumPy
    TypeError/ValueError escaping data type resolution."""
    path = _store_with_config(tmp_path, config)
    with pytest.raises(ValueError, match="No Zarr data type found"):
        zarr.open_array(path, mode="r")
