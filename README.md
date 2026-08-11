# zarr-vlen-ndarray

The `ndarray` data type and the `vlen-ndarray` codec for Zarr v3.

Each array element is itself an ndarray of a fixed scalar `dtype` whose shape
matches a configured pattern in which `null` marks a variable-length
dimension: `[null, 2]` describes elements of shape `(n, 2)` with `n` varying
per element, `[null]` describes `(n,)` elements. One parameterized type
covers, e.g., `(n, 2)` float32 centroid sets and `(n,)` uint64 location
lists.

```json
{
  "name": "ndarray",
  "configuration": {"dtype": "float32", "shape": [null, 2]}
}
```

The package provides:

- **`NDArray`** — a zarr-python `ZDType` for the data type above (also
  exported as `NDArrayDType`, for import sites that already use
  `numpy.typing.NDArray`). The data type accepts the full shape-pattern
  grammar (fixed shapes, one variable dimension, several); each paired
  `array -> bytes` codec declares the subset of patterns it serves.
- **`VlenNDArrayCodec`** — the paired parameter-free `array -> bytes` codec
  (`vlen-ndarray`), which serves exactly the patterns with one variable
  dimension in the leading position (`[null, d1, ..., dk]`). It serializes
  each element as its raw little-endian C-order bytes and frames chunks
  exactly like zarr's `vlen-bytes` codec (numcodecs `VLenBytes` framing:
  u32le count, then per element u32le length + payload); on decode the
  leading extent is inferred as `n = payload_bytes / (itemsize × d1 × ... ×
  dk)`, so no per-element shape header is stored.

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

From [PyPI](https://pypi.org/project/zarr-vlen-ndarray/):

```sh
pip install zarr-vlen-ndarray
```

Or from source:

```sh
pip install git+https://github.com/espg/zarr-vlen-ndarray
```

Requires Python >= 3.11, `zarr >= 3.1.0` (the ZDType API), `numcodecs >= 0.14`.

## Usage

```python
import numpy as np
import zarr
import zarr_vlen_ndarray  # registers the data type and codec
from zarr_vlen_ndarray import NDArray, VlenNDArrayCodec

zdtype = NDArray(dtype="float32", shape=(None, 2))
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

- **Shape patterns gate codecs.** `NDArray` accepts any pattern (`(None, 2)`,
  `(None, 3, 2)`, `(3, 2)`, `(None, None)`, ...), but `VlenNDArrayCodec`
  refuses, at array-creation time, any pattern that is not exactly one
  variable dimension in the leading position. Fixed-shape patterns belong
  with a fixed-size codec (the core `bytes` codec, spec-wise). A *single*
  variable dimension is length-inferable wherever it sits — `NDArray` decodes
  `(2, None)` payloads fine — so the leading-position restriction is this
  codec's design choice (it is what makes growing an element append bytes,
  preserving byte identity with `vlen-bytes` stores), not a limit of the
  inference rule; a header-free codec for the non-leading case could be
  registered separately. Only patterns with **two or more** variable
  dimensions genuinely need a per-element shape-headered codec, and none
  exists yet.
- **Slice reads** return object arrays whose cells are float32/uint64/...
  ndarrays. Decoded cells are **read-only views** over the decoded payload
  (zero-copy); call `.copy()` to mutate.
- **Scalar reads** (`arr[0]`) return the cell wrapped in a 0-d object array —
  zarr-python does the same for its own vlen types. `zarr_vlen_ndarray.unbox`
  recovers the cell.
- **Fill values**: with a variable dimension in the pattern, the default fill
  is the empty cell (variable extents 0; metadata `fill_value: ""`, the base64
  encoding of an empty byte string); for all-fixed patterns it is the
  zero-filled element. Cells
  materialized from the fill are `VlenScalar` instances — ndarray subclasses
  whose `==`/`!=` compare the whole cell and return a plain bool (this is
  what makes zarr's empty-chunk elision work for ndarray cells; in every
  other respect they behave like regular ndarrays).

## Registration and the no-package failure mode

The package declares both zarr entry points:

```toml
[project.entry-points."zarr.data_type"]
ndarray = "zarr_vlen_ndarray:NDArray"

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

A vanilla zarr user *without* this package installed who opens an `ndarray`
store gets (verbatim):

```
ValueError: No Zarr data type found that matches {'name': 'ndarray', 'configuration': {'dtype': 'float32', 'shape': [None, 2]}}
```

The fix is `pip install zarr-vlen-ndarray` + `import zarr_vlen_ndarray` — or,
without installing anything, the metadata-only downgrade to `bytes` +
`vlen-bytes` described above.

## Registry status

The `ndarray` data type and `vlen-ndarray` codec names are **submitted but
not yet registered** with
[zarr-developers/zarr-extensions](https://github.com/zarr-developers/zarr-extensions):
see [zarr-extensions#71](https://github.com/zarr-developers/zarr-extensions/pull/71).
The PR originally registered a single `vlen-ndarray` data type + codec pair;
per review feedback there, it was restructured into the general `ndarray`
data type plus the constrained `vlen-ndarray` codec that this package now
implements. Until that PR merges, treat the names as provisional. The
registry-formatted spec files in [`registry/`](registry/) are the ones under
review.

**Migrating a pre-restructure store.** Releases before the restructure wrote
`{"name": "vlen-ndarray", "configuration": {"dtype": ..., "inner_shape": [...]}}`
as the `data_type`. That name no longer resolves, so such a store fails to
open with `ValueError: No Zarr data type found that matches ...`. The chunk
bytes are unaffected by the restructure, so the fix is a metadata-only
rewrite of `zarr.json`: replace `data_type` with `{"name": "ndarray",
"configuration": {"dtype": <same dtype>, "shape": [null, *inner_shape]}}` and
leave `codecs` and `fill_value` untouched.

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
