"""Byte-identity proof: the vlen-ndarray codec's encoded chunk bytes are
byte-identical to what zarr's vlen-bytes codec produces for the equivalent raw
little-endian payload bytes — both uncompressed and through a zstd-compressed
chain. This is the property that makes migrating a bytes-typed store a
metadata-only rewrite (englacial/zagg#210)."""

import struct
import warnings

import numpy as np
import pytest
import zarr
from zarr.codecs import ZstdCodec
from zarr.dtype import VariableLengthBytes

from conftest import as_object_array
from zarr_vlen_ndarray import VlenNDArray, VlenNDArrayCodec


def _create_bytes_array(store, n, compressors):
    # zarr's variable_length_bytes dtype is spec-unstable upstream
    # (zarr-python#3517); the warning is irrelevant to the byte comparison.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return zarr.create_array(
            store=store,
            shape=(n,),
            chunks=(n,),
            dtype=VariableLengthBytes(),
            compressors=compressors,
        )


@pytest.mark.parametrize("compressors", [None, ZstdCodec(level=3)], ids=["raw", "zstd3"])
@pytest.mark.parametrize("case", ["float32_2col", "uint64_flat"])
def test_chunk_bytes_identical_to_vlen_bytes(
    tmp_path, case, compressors, float32_cells, uint64_cells
):
    if case == "float32_2col":
        cells = float32_cells
        zdtype = VlenNDArray(dtype="float32", inner_shape=(2,))
    else:
        cells = uint64_cells
        zdtype = VlenNDArray(dtype="uint64", inner_shape=())
    n = len(cells)

    a = _create_bytes_array(tmp_path / "bytes", n, compressors)
    a[:] = as_object_array([c.tobytes() for c in cells])

    b = zarr.create_array(
        store=tmp_path / "typed",
        shape=(n,),
        chunks=(n,),
        dtype=zdtype,
        serializer=VlenNDArrayCodec(),
        compressors=compressors,
    )
    b[:] = as_object_array(cells)

    bytes_chunk = (tmp_path / "bytes" / "c" / "0").read_bytes()
    typed_chunk = (tmp_path / "typed" / "c" / "0").read_bytes()
    assert typed_chunk == bytes_chunk


def test_golden_framing(tmp_path):
    """Pin the wire format itself: u32le cell count, then per cell a u32le
    byte length followed by the raw little-endian payload (numcodecs
    VLenBytes framing)."""
    cells = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f4"),
        np.empty((0, 2), dtype="<f4"),
        np.array([[5.5, 6.5]], dtype="<f4"),
    ]
    zdtype = VlenNDArray(dtype="float32", inner_shape=(2,))
    arr = zarr.create_array(
        store=tmp_path / "a",
        shape=(3,),
        chunks=(3,),
        dtype=zdtype,
        serializer=VlenNDArrayCodec(),
        compressors=None,
    )
    arr[:] = as_object_array(cells)

    expected = struct.pack("<I", len(cells))
    for cell in cells:
        payload = cell.tobytes()
        expected += struct.pack("<I", len(payload)) + payload
    assert (tmp_path / "a" / "c" / "0").read_bytes() == expected


def test_relabel_bytes_store_reads_identically(tmp_path, float32_cells):
    """A vlen-bytes store's chunk objects, re-labeled vlen-ndarray via a
    metadata-only zarr.json rewrite, read back as the typed cells."""
    import json

    n = len(float32_cells)
    a = _create_bytes_array(tmp_path / "s", n, ZstdCodec(level=3))
    a[:] = as_object_array([c.tobytes() for c in float32_cells])

    meta_path = tmp_path / "s" / "zarr.json"
    meta = json.loads(meta_path.read_text())
    meta["data_type"] = {
        "name": "vlen-ndarray",
        "configuration": {"dtype": "float32", "inner_shape": [2]},
    }
    meta["codecs"] = [{"name": "vlen-ndarray", "configuration": {}}] + [
        c for c in meta["codecs"] if c["name"] != "vlen-bytes"
    ]
    meta["fill_value"] = ""
    meta_path.write_text(json.dumps(meta))

    reread = zarr.open_array(tmp_path / "s", mode="r")
    assert isinstance(reread.metadata.data_type, VlenNDArray)
    for expected, got in zip(float32_cells, reread[:]):
        assert np.array_equal(np.asarray(got), expected)
