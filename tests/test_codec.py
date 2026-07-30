"""Codec object semantics: dict form, config parsing."""

import pytest

from zarr_vlen_ndarray import VlenNDArrayCodec


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
