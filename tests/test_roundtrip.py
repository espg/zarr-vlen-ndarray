"""Write/read round-trips through zarr, including sharding and fill behavior."""

import numpy as np
import pytest
import zarr
from zarr.codecs import ZstdCodec

from conftest import as_object_array
from zarr_vlen_ndarray import VlenNDArray, VlenNDArrayCodec, VlenScalar, unbox

FLOAT32_2COL = VlenNDArray(dtype="float32", inner_shape=(2,))
UINT64_FLAT = VlenNDArray(dtype="uint64", inner_shape=())


def _assert_cells_equal(expected_cells, got):
    assert len(expected_cells) == len(got)
    for expected, cell in zip(expected_cells, got):
        cell = np.asarray(unbox(cell))
        assert cell.dtype == expected.dtype
        assert np.array_equal(cell, expected)


@pytest.mark.parametrize(
    ("zdtype", "cells_fixture"),
    [(FLOAT32_2COL, "float32_cells"), (UINT64_FLAT, "uint64_cells")],
    ids=["float32_2col", "uint64_flat"],
)
def test_roundtrip(tmp_path, request, zdtype, cells_fixture):
    cells = request.getfixturevalue(cells_fixture)
    arr = zarr.create_array(
        store=tmp_path / "a",
        shape=(len(cells),),
        chunks=(2,),
        dtype=zdtype,
        serializer=VlenNDArrayCodec(),
        compressors=ZstdCodec(level=3),
    )
    arr[:] = as_object_array(cells)
    _assert_cells_equal(cells, arr[:])
    # scalar reads of object-dtype zarr arrays return the cell wrapped in a
    # 0-d object array (zarr-python behaves the same for its own vlen types);
    # unbox() recovers the cell
    single = arr[0]
    assert isinstance(single, np.ndarray)
    assert single.dtype == object and single.ndim == 0
    assert np.array_equal(np.asarray(unbox(single)), cells[0])


def test_reopen_preserves_dtype_and_values(tmp_path, float32_cells):
    arr = zarr.create_array(
        store=tmp_path / "a",
        shape=(len(float32_cells),),
        chunks=(3,),
        dtype=FLOAT32_2COL,
        serializer=VlenNDArrayCodec(),
    )
    arr[:] = as_object_array(float32_cells)

    reopened = zarr.open_array(tmp_path / "a", mode="r")
    assert reopened.metadata.data_type == FLOAT32_2COL
    assert reopened.metadata.to_dict()["data_type"] == {
        "name": "vlen-ndarray",
        "configuration": {"dtype": "float32", "inner_shape": [2]},
    }
    _assert_cells_equal(float32_cells, reopened[:])


def test_empty_cells_only_chunk_elided_and_read_back(tmp_path):
    """A chunk whose every cell equals the fill value (the empty cell) is
    elided by zarr's write_empty_chunks=False default and reads back as
    empty cells — this exercises VlenScalar's whole-cell equality."""
    arr = zarr.create_array(
        store=tmp_path / "a",
        shape=(4,),
        chunks=(2,),
        dtype=FLOAT32_2COL,
        serializer=VlenNDArrayCodec(),
    )
    cells = [
        np.empty((0, 2), dtype="<f4"),
        np.empty((0, 2), dtype="<f4"),
        np.ones((2, 2), dtype="<f4"),
        np.empty((0, 2), dtype="<f4"),
    ]
    arr[:] = as_object_array(cells)
    # chunk 0 is all-fill: not written
    assert not (tmp_path / "a" / "c" / "0").exists()
    assert (tmp_path / "a" / "c" / "1").exists()
    _assert_cells_equal(cells, arr[:])


def test_unwritten_chunks_read_as_empty_cells(tmp_path):
    arr = zarr.create_array(
        store=tmp_path / "a",
        shape=(6,),
        chunks=(3,),
        dtype=UINT64_FLAT,
        serializer=VlenNDArrayCodec(),
    )
    written = np.array([1, 2, 3], dtype="<u8")
    staged = np.empty(1, dtype=object)
    staged[0] = written
    arr[0:1] = staged

    # slice read across the never-written chunk: cells are (0,)-shaped arrays
    tail = arr[3:6]
    for cell in tail:
        assert isinstance(cell, VlenScalar)
        assert cell.shape == (0,)
    # scalar read of a cell in a never-written chunk returns the boxed fill
    # (a zarr limitation for container scalars; unbox() recovers the cell)
    boxed = arr[4]
    cell = unbox(boxed)
    assert np.asarray(cell).shape == (0,)
    # partial write into chunk 0 backfilled cells 1-2 with the fill
    _assert_cells_equal([written, np.empty(0, "<u8"), np.empty(0, "<u8")], arr[0:3])


def test_zero_length_array(tmp_path):
    arr = zarr.create_array(
        store=tmp_path / "a",
        shape=(0,),
        chunks=(1,),
        dtype=FLOAT32_2COL,
        serializer=VlenNDArrayCodec(),
    )
    assert arr[:].shape == (0,)


def test_large_n(tmp_path):
    rng = np.random.default_rng(0)
    big = rng.random((500_000, 2)).astype("<f4")
    cells = [big, np.empty((0, 2), dtype="<f4"), big[:17]]
    arr = zarr.create_array(
        store=tmp_path / "a",
        shape=(3,),
        chunks=(3,),
        dtype=FLOAT32_2COL,
        serializer=VlenNDArrayCodec(),
        compressors=ZstdCodec(level=3),
    )
    arr[:] = as_object_array(cells)
    _assert_cells_equal(cells, arr[:])


def test_sharding_codec_wrapped(tmp_path, float32_cells):
    arr = zarr.create_array(
        store=tmp_path / "a",
        shape=(len(float32_cells),),
        shards=(6,),
        chunks=(2,),
        dtype=FLOAT32_2COL,
        serializer=VlenNDArrayCodec(),
        compressors=ZstdCodec(level=3),
    )
    arr[:] = as_object_array(float32_cells)
    _assert_cells_equal(float32_cells, arr[:])
    codecs = zarr.open_array(tmp_path / "a", mode="r").metadata.to_dict()["codecs"]
    assert codecs[0]["name"] == "sharding_indexed"
    inner = codecs[0]["configuration"]["codecs"]
    assert inner[0]["name"] == "vlen-ndarray"


def test_codec_requires_vlen_ndarray_dtype(tmp_path):
    with pytest.raises(ValueError, match="only compatible"):
        zarr.create_array(
            store=tmp_path / "a",
            shape=(4,),
            chunks=(4,),
            dtype="float32",
            serializer=VlenNDArrayCodec(),
        )


def test_missing_serializer_error_is_actionable(tmp_path):
    """Without an explicit serializer, zarr's default-serializer lookup names
    the vlen-ndarray codec in its error (via object_codec_id)."""
    with pytest.raises(ValueError, match="vlen-ndarray"):
        zarr.create_array(
            store=tmp_path / "a",
            shape=(4,),
            chunks=(4,),
            dtype=FLOAT32_2COL,
        )
