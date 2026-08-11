"""Codec object semantics: dict form, config parsing, framing bounds."""

import json

import numpy as np
import pytest
import zarr

from conftest import as_object_array
from zarr_vlen_ndarray import NDArray, VlenNDArrayCodec
from zarr_vlen_ndarray import codec as codec_module


def test_to_dict():
    assert VlenNDArrayCodec().to_dict() == {"name": "vlen-ndarray", "configuration": {}}


@pytest.mark.parametrize(
    "data",
    [
        {"name": "vlen-ndarray"},
        {"name": "vlen-ndarray", "configuration": {}},
    ],
)
def test_from_dict(data):
    assert VlenNDArrayCodec.from_dict(data) == VlenNDArrayCodec()


def test_from_dict_rejects_wrong_name():
    with pytest.raises(ValueError):
        VlenNDArrayCodec.from_dict({"name": "vlen-bytes"})


def test_compute_encoded_size_not_implemented():
    from zarr.core.array_spec import ArraySpec  # noqa: F401

    with pytest.raises(NotImplementedError):
        VlenNDArrayCodec().compute_encoded_size(0, None)


# -- u32 framing bounds -----------------------------------------------------
#
# The registration says: "a chunk MUST NOT contain more than 2^32 - 1 elements,
# and an element's serialized payload MUST NOT exceed 2^32 - 1 bytes ...
# Encoders MUST report an error rather than truncate or overflow either count."
# numcodecs stores the low 32 bits of both counts silently, so the guard lives
# in `_encode_single`. The real bounds need >4 GiB of data to reach, so these
# tests shrink the bound instead of the data.

ZDTYPE = NDArray(dtype="float32", shape=(None, 2))


def _write(path, cells):
    arr = zarr.create_array(
        store=path,
        shape=(len(cells),),
        chunks=(len(cells),),
        dtype=ZDTYPE,
        serializer=VlenNDArrayCodec(),
    )
    arr[:] = as_object_array(cells)
    return arr


def test_encode_rejects_oversized_element_payload(tmp_path, monkeypatch):
    cells = [np.ones((3, 2), dtype="<f4"), np.empty((0, 2), dtype="<f4")]
    # 24 bytes per non-empty payload; bound of 8 admits only the empty one
    monkeypatch.setattr(codec_module, "U32_MAX", 8)
    with pytest.raises(ValueError, match="length prefix is a u32le"):
        _write(tmp_path / "a", cells)


def test_encode_rejects_oversized_chunk_element_count(tmp_path, monkeypatch):
    # non-empty cells: an all-fill chunk is elided before the codec is called
    cells = [np.ones((1, 2), dtype="<f4")] * 3
    monkeypatch.setattr(codec_module, "U32_MAX", 2)
    with pytest.raises(ValueError, match="framed element count is a u32le"):
        _write(tmp_path / "a", cells)


def test_encode_within_bounds_still_writes(tmp_path, monkeypatch):
    """The guards must not fire on payloads and counts that fit."""
    cells = [np.ones((3, 2), dtype="<f4"), np.empty((0, 2), dtype="<f4")]
    monkeypatch.setattr(codec_module, "U32_MAX", 24)
    arr = _write(tmp_path / "a", cells)
    assert arr[:][0].shape == (3, 2)


def test_decode_reports_element_count_mismatch(tmp_path):
    """A chunk whose framed element count disagrees with the chunk shape must
    fail with a message naming the codec and both counts, not with NumPy's
    "cannot reshape array of size 2 into shape (3,)"."""
    path = tmp_path / "a"
    _write(path, [np.ones((3, 2), dtype="<f4"), np.empty((0, 2), dtype="<f4")])
    # widen the array and its chunk grid so the stored 2-element chunk is read
    # into a 3-element chunk
    meta_path = path / "zarr.json"
    meta = json.loads(meta_path.read_text())
    meta["shape"] = [3]
    meta["chunk_grid"]["configuration"]["chunk_shape"] = [3]
    meta_path.write_text(json.dumps(meta))

    arr = zarr.open_array(path, mode="r")
    with pytest.raises(ValueError, match="must equal the product of the chunk shape"):
        arr[:]
