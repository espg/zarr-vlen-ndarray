# The `ndarray` / `vlen-ndarray` extension

This repository defines two Zarr v3 extensions: the **`ndarray` data type**,
whose elements are themselves n-dimensional arrays with a shape pattern in
which `null` marks a variable-length dimension, and the paired parameter-free
**`vlen-ndarray` `array -> bytes` codec**, which serves the patterns with
exactly one variable dimension in the leading position and serializes them
with `vlen-bytes` framing.

The normative extension text lives in the [`registry/`](registry/) directory,
formatted as [zarr-extensions](https://github.com/zarr-developers/zarr-extensions)
registry entries (these are the files intended for the registry submission):

- **Data type**: [`registry/data-types/ndarray/README.md`](registry/data-types/ndarray/README.md)
  (with [`schema.json`](registry/data-types/ndarray/schema.json))
- **Codec**: [`registry/codecs/vlen-ndarray/README.md`](registry/codecs/vlen-ndarray/README.md)
  (with [`schema.json`](registry/codecs/vlen-ndarray/schema.json))

Conformance requirements in those documents use [RFC2119] terminology, per
the zarr-extensions document conventions.

[RFC2119]: https://tools.ietf.org/html/rfc2119

## Summary (non-normative)

Metadata form:

```json
{
  "data_type": {
    "name": "ndarray",
    "configuration": {"dtype": "float32", "shape": [null, 2]}
  },
  "fill_value": "",
  "codecs": [{"name": "vlen-ndarray"}, {"name": "zstd", "configuration": {"level": 3}}]
}
```

- Each array element is an ndarray with scalar type `configuration.dtype` (a
  fixed-size numeric/boolean core data type) whose shape matches the pattern
  `configuration.shape`: fixed (integer) dimensions are equal across
  elements, `null` dimensions vary per element.
- **The shape pattern gates the codec**: each extension `array -> bytes` codec
  declares which patterns it serves, and the data type itself states the rule
  for the core `bytes` codec (fixed-shape patterns only, C-order scalars,
  little-endian). The `vlen-ndarray` codec registered here serves exactly one
  variable dimension in the leading position (`[null, d1, ..., dk]`). A single
  variable dimension is length-inferable wherever it sits, so the leading
  restriction is that codec's design choice — it keeps growth append-only and
  hence byte-identical to `vlen-bytes` — and a header-free codec for the
  non-leading case could be registered separately. Only patterns with two or
  more variable dimensions need per-element shape-headered codecs
  (`[shape] + [elements]`, or `dimensionality + [shape] + [elements]`), which
  are not registered — the grammar is general so they can be added without
  revising the data type.
- The `vlen-ndarray` codec serializes each element as its raw little-endian,
  C-order bytes and frames the chunk exactly as the registered `vlen-bytes`
  codec (u32le element count; per element a u32le byte length + payload). On
  decode the leading extent is inferred as `n = payload_bytes / (itemsize ×
  d1 × ... × dk)` — no per-element shape header is stored.
- The fill value is the base64 encoding of a fill element's raw bytes; with a
  variable dimension in the pattern, `""` (the empty element, variable
  extents `0`) is the default.

## The byte-identity guarantee

For any chunk, the bytes produced by the `vlen-ndarray` codec are
**byte-identical** to the bytes the `vlen-bytes` codec produces for the
object array of per-element `tobytes()` payloads. The two on-disk forms

- `data_type: "bytes"` + `codecs: [vlen-bytes, ...]` with raw-LE-bytes
  elements (the [`zagg-ragged/1`](https://github.com/englacial/zagg/issues/340)
  convention), and
- `data_type: ndarray` + `codecs: [vlen-ndarray, ...]` (`zagg-ragged/2`)

differ **only in metadata**; chunk objects can be shared or migrated by
rewriting `zarr.json` alone. This property is pinned by
`tests/test_byte_identity.py`, including through a zstd-compressed chain and
a relabel-and-reread test.

## Relationship to zagg

This data type is stage 3–4 of the plan ratified on
[englacial/zagg#210](https://github.com/englacial/zagg/issues/210): the
`zagg-ragged/2` store revision types the ragged t-digest fields
(`(n, 2)` float32 centroid sets, `(n,)` uint64 location lists) that revision
`/1` stores as documented raw bytes. The `/1` and `/2` wire formats are
identical by construction (above); `/2` moves the element type description
from the attributes convention into the type system. The `zagg-ragged` store
spec itself is tracked in [englacial/zagg#340](https://github.com/englacial/zagg/issues/340).

## Registry status

Submitted, not yet registered. The registry PR is
[zarr-extensions#71](https://github.com/zarr-developers/zarr-extensions/pull/71),
which registers the `registry/` files verbatim. As originally opened, the PR
registered a single `vlen-ndarray` data type + codec pair with a
`{dtype, inner_shape}` configuration; per review feedback on the PR thread it
was restructured into the general `ndarray` data type (shape pattern with
`null`-marked variable dimensions) plus the constrained `vlen-ndarray`
codec — the dtype is general, the codec carries the constraints. Until that
PR merges, the names should be treated as provisional.

Related upstream discussions:

- [zarr-extensions#57](https://github.com/zarr-developers/zarr-extensions/issues/57) —
  generic container data types (`variable_length[<base>]`); `ndarray` is a
  concrete point in that design space.
- [zarr-extensions#62](https://github.com/zarr-developers/zarr-extensions/pull/62) —
  fixed-size list data type; an adjacent design for the fixed-shape pattern
  family, which the core `bytes` codec already serves.
- [zarr-python#2618](https://github.com/zarr-developers/zarr-python/issues/2618) —
  native ragged array support. If zarr-python grows native support with a
  compatible wire format, this package can be retired without data migration.
