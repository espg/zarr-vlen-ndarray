"""Tests for the VlenNDArray ZDType: metadata JSON, scalars, validation."""

import numpy as np
import pytest
from zarr.dtype import DataTypeValidationError

from zarr_vlen_ndarray import VlenNDArray, VlenScalar, unbox

V3_JSON = {"name": "vlen-ndarray", "configuration": {"dtype": "float32", "inner_shape": [2]}}


def test_to_json_v3():
    zd = VlenNDArray(dtype="float32", inner_shape=(2,))
    assert zd.to_json(zarr_format=3) == V3_JSON


def test_from_json_v3_roundtrip():
    zd = VlenNDArray._from_json_v3(V3_JSON)
    assert zd == VlenNDArray(dtype="float32", inner_shape=(2,))
    assert zd.inner_shape == (2,)
    assert isinstance(zd.inner_shape, tuple)


def test_json_roundtrip_scalar_inner_shape():
    zd = VlenNDArray(dtype="uint64", inner_shape=())
    data = zd.to_json(zarr_format=3)
    assert data == {"name": "vlen-ndarray", "configuration": {"dtype": "uint64", "inner_shape": []}}
    assert VlenNDArray._from_json_v3(data) == zd


@pytest.mark.parametrize(
    "bad",
    [
        "vlen-ndarray",
        {"name": "vlen-ndarray"},
        {"name": "other", "configuration": {"dtype": "float32", "inner_shape": []}},
        {"name": "vlen-ndarray", "configuration": {"dtype": "float32"}},
        {"name": "vlen-ndarray", "configuration": {"dtype": "float32", "inner_shape": [2], "x": 1}},
        {"name": "vlen-ndarray", "configuration": {"dtype": "float32", "inner_shape": "2"}},
        {"name": "vlen-ndarray", "configuration": {"dtype": 32, "inner_shape": [2]}},
    ],
)
def test_check_json_v3_rejects(bad):
    assert not VlenNDArray._check_json_v3(bad)
    with pytest.raises(DataTypeValidationError):
        VlenNDArray._from_json_v3(bad)


def test_v2_unsupported():
    zd = VlenNDArray(dtype="float32", inner_shape=(2,))
    assert not VlenNDArray._check_json_v2({"name": "|O", "object_codec_id": None})
    with pytest.raises(DataTypeValidationError):
        VlenNDArray._from_json_v2({"name": "|O", "object_codec_id": None})
    with pytest.raises(ValueError, match="v3-only"):
        zd.to_json(zarr_format=2)


def test_dtype_normalization_and_validation():
    assert VlenNDArray(dtype=np.dtype("float32"), inner_shape=[2]).dtype == "float32"
    assert VlenNDArray(dtype="<f4").dtype == "float32"
    with pytest.raises(ValueError, match="dtype must be one of"):
        VlenNDArray(dtype="str")
    with pytest.raises(ValueError, match="inner_shape"):
        VlenNDArray(dtype="float32", inner_shape=(0,))


def test_from_native_dtype_never_infers():
    with pytest.raises(DataTypeValidationError, match="construct VlenNDArray"):
        VlenNDArray.from_native_dtype(np.dtype("O"))


def test_item_dtype_little_endian():
    zd = VlenNDArray(dtype="uint64", inner_shape=())
    assert zd.item_dtype == np.dtype("<u8")
    assert zd.item_nbytes == 8
    zd2 = VlenNDArray(dtype="float32", inner_shape=(2,))
    assert zd2.item_nbytes == 8


def test_coerce_cell():
    zd = VlenNDArray(dtype="float32", inner_shape=(2,))
    cell = zd.coerce_cell([[1.0, 2.0], [3.0, 4.0]])
    assert cell.shape == (2, 2)
    assert cell.dtype == np.dtype("<f4")
    assert cell.flags.c_contiguous
    # empty in any shape coerces to (0, 2)
    assert zd.coerce_cell(np.empty(0)).shape == (0, 2)
    with pytest.raises(TypeError, match="expected shape"):
        zd.coerce_cell(np.ones((2, 3), dtype="<f4"))
    with pytest.raises(TypeError):
        zd.coerce_cell("nope")


def test_payload_to_cell():
    zd = VlenNDArray(dtype="float32", inner_shape=(2,))
    cell = np.arange(6, dtype="<f4").reshape(3, 2)
    out = zd.payload_to_cell(cell.tobytes())
    assert np.array_equal(out, cell)
    assert zd.payload_to_cell(b"").shape == (0, 2)
    with pytest.raises(ValueError, match="whole number"):
        zd.payload_to_cell(b"\x00" * 7)


def test_scalar_json_roundtrip():
    zd = VlenNDArray(dtype="float32", inner_shape=(2,))
    cell = np.arange(4, dtype="<f4").reshape(2, 2)
    encoded = zd.to_json_scalar(cell, zarr_format=3)
    assert isinstance(encoded, str)
    decoded = zd.from_json_scalar(encoded, zarr_format=3)
    assert np.array_equal(unbox(decoded), cell)
    # the default fill is the empty cell, base64 ""
    assert zd.to_json_scalar(zd.default_scalar(), zarr_format=3) == ""
    with pytest.raises(TypeError, match="base64"):
        zd.from_json_scalar(123, zarr_format=3)


def test_cast_scalar_is_boxed():
    zd = VlenNDArray(dtype="float32", inner_shape=(2,))
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


def test_unbox_passthrough():
    x = np.ones((2, 2))
    assert unbox(x) is x
    assert unbox("s") == "s"
