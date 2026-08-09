# The `vlen-ndarray` extension

This repository defines the `vlen-ndarray` Zarr v3 extension: a **data type**
whose elements are variable-length ndarrays of shape `(n, *inner_shape)`, and
the paired parameter-free **`array -> bytes` codec** that serializes them with
`vlen-bytes` framing.

The normative extension text lives in the [`registry/`](registry/) directory,
formatted as [zarr-extensions](https://github.com/zarr-developers/zarr-extensions)
registry entries (these are the files intended for the registry submission —
see [`registry_submission_draft.md`](registry_submission_draft.md)):

- **Data type**: [`registry/data-types/vlen-ndarray/README.md`](registry/data-types/vlen-ndarray/README.md)
  (with [`schema.json`](registry/data-types/vlen-ndarray/schema.json))
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
    "name": "vlen-ndarray",
    "configuration": {"dtype": "float32", "inner_shape": [2]}
  },
  "fill_value": "",
  "codecs": [{"name": "vlen-ndarray"}, {"name": "zstd", "configuration": {"level": 3}}]
}
```

- Each array element is an ndarray of shape `(n, *inner_shape)` and scalar
  type `configuration.dtype` (a fixed-size numeric/boolean core data type);
  `n` varies per element and is implied by the element's encoded byte length.
- The codec serializes each element as its raw little-endian, C-order bytes
  and frames the chunk exactly as the registered `vlen-bytes` codec (u32le
  element count; per element a u32le byte length + payload).
- The fill value is the base64 encoding of a fill element's raw bytes; `""`
  (the empty element, `n = 0`) is the default.

## The byte-identity guarantee

For any chunk, the bytes produced by the `vlen-ndarray` codec are
**byte-identical** to the bytes the `vlen-bytes` codec produces for the
object array of per-element `tobytes()` payloads. The two on-disk forms

- `data_type: "bytes"` + `codecs: [vlen-bytes, ...]` with raw-LE-bytes
  elements (the [`zagg-ragged/1`](https://github.com/englacial/zagg/issues/340)
  convention), and
- `data_type: vlen-ndarray` + `codecs: [vlen-ndarray, ...]` (`zagg-ragged/2`)

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
identical by construction (above); `/2` moves `{dtype, inner_shape}` from the
attributes convention into the type system. The `zagg-ragged` store spec
itself is tracked in [englacial/zagg#340](https://github.com/englacial/zagg/issues/340).

## Registry status

Not yet registered. `vlen-ndarray` does not conflict with any entry in
zarr-developers/zarr-extensions as of 2026-08-08 (re-checked against the
`data-types/` and `codecs/` listings at upstream `4da7b37`). The prepared
submission is
[`registry_submission_draft.md`](registry_submission_draft.md); until the
registry PR merges, the name should be treated as provisional.

Related upstream discussions:

- [zarr-extensions#57](https://github.com/zarr-developers/zarr-extensions/issues/57) —
  generic container data types (`variable_length[<base>]`); `vlen-ndarray` is
  a concrete, fixed-inner-shape point in that design space.
- [zarr-python#2618](https://github.com/zarr-developers/zarr-python/issues/2618) —
  native ragged array support. If zarr-python grows native support with a
  compatible wire format, this package can be retired without data migration.
