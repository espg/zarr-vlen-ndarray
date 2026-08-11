"""Registration mechanics and the without-package failure mode.

Entry-point status (pinned by these tests, zarr-python 3.1.0-3.2.1):

- the `zarr.codecs` entry point WORKS: zarr lazy-loads codec classes by name
  on demand (`zarr.registry.get_codec_class`).
- the `zarr.data_type` entry point is collected by zarr into
  `data_type_registry._lazy_load_list` but zarr never calls `_lazy_load()`,
  so data type discovery via entry points is currently inert upstream. Until
  that is fixed, readers must `import zarr_vlen_ndarray` (which registers
  eagerly) before opening a store.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import zarr
from zarr.dtype import data_type_registry

from conftest import as_object_array
from zarr_vlen_ndarray import NDArray, VlenNDArrayCodec

ZDTYPE = NDArray(dtype="float32", shape=(None, 2))

NO_DTYPE_ERROR = (
    "No Zarr data type found that matches {'name': 'ndarray', "
    "'configuration': {'dtype': 'float32', 'shape': [None, 2]}}"
)


def _build_store(path):
    arr = zarr.create_array(
        store=path,
        shape=(2,),
        chunks=(2,),
        dtype=ZDTYPE,
        serializer=VlenNDArrayCodec(),
    )
    cells = [np.arange(6, dtype="<f4").reshape(3, 2), np.empty((0, 2), dtype="<f4")]
    arr[:] = as_object_array(cells)
    return cells


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True
    )


def test_fresh_interpreter_with_import(tmp_path):
    """The documented usage: import zarr_vlen_ndarray, then open."""
    _build_store(tmp_path / "a")
    result = _run(
        f"""
        import numpy as np
        import zarr
        import zarr_vlen_ndarray  # registers the data type and codec
        arr = zarr.open_array(r"{tmp_path / "a"}", mode="r")
        assert type(arr.metadata.data_type).__name__ == "NDArray"
        cells = arr[:]
        assert cells[0].shape == (3, 2)
        assert np.array_equal(cells[0], np.arange(6, dtype="<f4").reshape(3, 2))
        assert cells[1].shape == (0, 2)
        """
    )
    assert result.returncode == 0, result.stderr


def test_data_type_entry_point_correct_but_upstream_never_flushes(tmp_path):
    """The `zarr.data_type` entry point is collected and loads our class when
    flushed — but zarr-python never flushes the lazy list itself, so without
    an explicit import the store does not open. If this test starts failing
    on a new zarr because the no-import branch succeeds, upstream fixed the
    flush: update the README and delete the first branch."""
    _build_store(tmp_path / "a")
    # without import and without flush: the documented failure
    result = _run(
        f"""
        import sys
        assert "zarr_vlen_ndarray" not in sys.modules
        import zarr
        try:
            zarr.open_array(r"{tmp_path / "a"}", mode="r")
        except ValueError as e:
            assert str(e) == {NO_DTYPE_ERROR!r}, str(e)
        else:
            raise AssertionError("expected ValueError; upstream may have fixed entry-point flushing")
        """
    )
    assert result.returncode == 0, result.stderr
    # the entry point itself is correct: flushing the lazy list imports and
    # registers this package without any explicit import
    result = _run(
        f"""
        import sys
        assert "zarr_vlen_ndarray" not in sys.modules
        import zarr
        from zarr.core.dtype import data_type_registry
        assert "ndarray" in [e.name for e in data_type_registry._lazy_load_list]
        data_type_registry._lazy_load()
        assert "zarr_vlen_ndarray" in sys.modules
        arr = zarr.open_array(r"{tmp_path / "a"}", mode="r")
        assert arr[:][0].shape == (3, 2)
        """
    )
    assert result.returncode == 0, result.stderr


def test_codec_entry_point_lazy_loads(tmp_path):
    """The `zarr.codecs` entry point works without an explicit import."""
    result = _run(
        """
        import sys
        assert "zarr_vlen_ndarray" not in sys.modules
        from zarr.registry import get_codec_class
        cls = get_codec_class("vlen-ndarray")
        assert cls.__name__ == "VlenNDArrayCodec"
        """
    )
    assert result.returncode == 0, result.stderr


def test_without_package_failure_mode(tmp_path):
    """What a vanilla-zarr user (package not installed) sees. Verbatim:

        ValueError: No Zarr data type found that matches {'name':
        'ndarray', 'configuration': {'dtype': 'float32',
        'shape': [None, 2]}}

    Data type resolution happens before codec resolution when parsing array
    metadata, so unregistering the data type alone reproduces exactly what a
    vanilla-zarr user observes."""
    _build_store(tmp_path / "a")

    data_type_registry._lazy_load()
    data_type_registry.unregister("ndarray")
    try:
        with pytest.raises(ValueError) as excinfo:
            zarr.open_array(tmp_path / "a", mode="r")
        assert str(excinfo.value) == NO_DTYPE_ERROR
    finally:
        data_type_registry.register("ndarray", NDArray)

    # sanity: works again after re-registration
    reopened = zarr.open_array(tmp_path / "a", mode="r")
    assert reopened[:][0].shape == (3, 2)


def test_import_registration_is_idempotent():
    import importlib

    import zarr_vlen_ndarray

    importlib.reload(zarr_vlen_ndarray)
    assert data_type_registry.get("ndarray") is NDArray
