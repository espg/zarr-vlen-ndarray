# zarr-vlen-ndarray

Typed variable-length ndarray data type (`vlen-ndarray`) for Zarr v3.

Each array element is a variable-length ndarray of shape `(n, *inner_shape)`:
the leading dimension `n` varies per element; the trailing `inner_shape` and
the scalar `dtype` are fixed in the data type's configuration. One
parameterized type covers, e.g., `(n, 2)` float32 centroid sets and `(n,)`
uint64 location lists.

```json
{
  "name": "vlen-ndarray",
  "configuration": {"dtype": "float32", "inner_shape": [2]}
}
```

The package provides:

- **`VlenNDArray`** — a zarr-python `ZDType` for the data type above;
- **`VlenNDArrayCodec`** — the paired parameter-free `array -> bytes` codec,
  which serializes each element as its raw little-endian C-order bytes and
  frames chunks exactly like zarr's `vlen-bytes` codec (numcodecs `VLenBytes`
  framing: u32le count, then per element u32le length + payload).

The normative extension text is in [SPEC.md](SPEC.md) and the
[`registry/`](registry/) directory (zarr-extensions house style).

## Why: byte identity with `bytes` + `vlen-bytes`

For any chunk, the encoded bytes are **byte-identical** to what zarr's
`vlen-bytes` codec produces for the equivalent raw-bytes payloads (proved in
`tests/test_byte_identity.py`, including through a zstd chain). A store that
keeps ragged data as the `bytes` data type with raw-LE-bytes elements — the
[`zagg-ragged/1`](https://github.com/englacial/zagg/issues/340) convention —
can be upgraded to the typed form by rewriting `zarr.json` only; chunk
objects are untouched. Downgrade is equally metadata-only, which is the
escape hatch for readers without this package.

This package is phase 1 of
[englacial/zagg#210](https://github.com/englacial/zagg/issues/210) and
defines the element type for the `zagg-ragged/2` store revision
([englacial/zagg#340](https://github.com/englacial/zagg/issues/340)).

## Install

Not yet on PyPI (see [docs/PUBLISHING.md](docs/PUBLISHING.md)). From source:

```sh
pip install git+https://github.com/espg/zarr-vlen-ndarray
```

Requires Python >= 3.11, `zarr >= 3.1.0` (the ZDType API), `numcodecs >= 0.14`.

## Usage

```python
import numpy as np
import zarr
import zarr_vlen_ndarray  # registers the data type and codec
from zarr_vlen_ndarray import VlenNDArray, VlenNDArrayCodec

zdtype = VlenNDArray(dtype="float32", inner_shape=(2,))
arr = zarr.create_array(
    store="example.zarr",
    shape=(4,),
    chunks=(4,),
    dtype=zdtype,
    serializer=VlenNDArrayCodec(),   # required: zarr has no default serializer for extension dtypes
    compressors=zarr.codecs.ZstdCodec(level=3),
)

cells = np.empty(4, dtype=object)   # object-array staging, as with zarr's vlen types
cells[:] = [
    np.random.random((3, 2)).astype("float32"),
    np.empty((0, 2), dtype="float32"),          # empty cell
    np.random.random((7, 2)).astype("float32"),
    np.random.random((1, 2)).astype("float32"),
]
arr[:] = cells

arr[:][0]        # -> float32 ndarray of shape (3, 2)
```

Reading requires `import zarr_vlen_ndarray` first (see the registration
section below); after that, plain `zarr.open` works.

### Semantics worth knowing

- **Slice reads** return object arrays whose cells are float32/uint64/...
  ndarrays. Decoded cells are **read-only views** over the decoded payload
  (zero-copy); call `.copy()` to mutate.
- **Scalar reads** (`arr[0]`) return the cell wrapped in a 0-d object array —
  zarr-python does the same for its own vlen types. `zarr_vlen_ndarray.unbox`
  recovers the cell.
- **Fill values**: the default fill is the empty cell `(0, *inner_shape)`
  (metadata `fill_value: ""`, base64 of zero bytes). Cells materialized from
  the fill are `VlenScalar` instances — ndarray subclasses whose `==`/`!=`
  compare the whole cell and return a plain bool (this is what makes zarr's
  empty-chunk elision work for ndarray cells; in every other respect they
  behave like regular ndarrays).

## Registration and the no-package failure mode

The package declares both zarr entry points:

```toml
[project.entry-points."zarr.data_type"]
vlen-ndarray = "zarr_vlen_ndarray:VlenNDArray"

[project.entry-points."zarr.codecs"]
vlen-ndarray = "zarr_vlen_ndarray:VlenNDArrayCodec"
```

and also registers eagerly on import. Status with current zarr-python
(3.1.0–3.2.1, pinned by `tests/test_registration.py`): the **codec** entry
point is discovered lazily by zarr as designed, but zarr collects
`zarr.data_type` entry points into the data type registry's lazy-load list
**without ever flushing it**, so data type discovery via entry points is
currently inert upstream. Practical rule: **`import zarr_vlen_ndarray` before
opening a store**. When upstream flushes the lazy list, the explicit import
becomes unnecessary automatically.

A vanilla zarr user *without* this package installed who opens a
`vlen-ndarray` store gets (verbatim):

```
ValueError: No Zarr data type found that matches {'name': 'vlen-ndarray', 'configuration': {'dtype': 'float32', 'inner_shape': [2]}}
```

The fix is `pip install zarr-vlen-ndarray` + `import zarr_vlen_ndarray` — or,
without installing anything, the metadata-only downgrade to `bytes` +
`vlen-bytes` described above.

## Registry status

The `vlen-ndarray` name is **not yet registered** with
[zarr-developers/zarr-extensions](https://github.com/zarr-developers/zarr-extensions)
(no conflicting entry exists as of 2026-07-30). The prepared submission —
registry-formatted spec files plus a PR description — is in
[`registry/`](registry/) and
[`registry_submission_draft.md`](registry_submission_draft.md).

## Development

```sh
uv sync --extra test
uv run pytest -v
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy src
```

CI runs the test matrix on Python 3.11–3.13 against latest zarr, plus a floor
job against `zarr==3.1.0`.

## License

MIT
