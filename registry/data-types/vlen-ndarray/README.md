# Vlen-ndarray data type

This document defines `vlen-ndarray`, a data type for arrays whose elements
are variable-length n-dimensional arrays: each element is an array of shape
`(n, *inner_shape)`, where the leading dimension `n` varies per element and
the trailing dimensions (`inner_shape`) and the scalar data type are fixed by
the configuration.

## Background

Ragged data — a regular grid of cells where each cell holds a variable number
of fixed-size records — is common in scientific workflows: per-cell point
observations, t-digest centroid sets, index lists, event records. Zarr v3 has
no native ragged array support; the common workaround stores each element's
raw bytes under the [`"bytes"`](../bytes/README.md) data type and documents
the element interpretation out of band (e.g. in `attributes`). The
`vlen-ndarray` data type promotes that interpretation into the type system:
the array is self-describing, and implementations can return typed arrays per
element instead of opaque byte strings.

`vlen-ndarray` is deliberately encoding-compatible with the `bytes` data type:
when the paired [`vlen-ndarray` codec](../../codecs/vlen-ndarray/README.md) is
used, encoded chunks are byte-identical to `bytes` + `vlen-bytes` chunks whose
elements are each element's raw little-endian bytes. Converting such a store
to `vlen-ndarray` is a metadata-only change.

## Data type representation

### Name

The name of this data type is the string `"vlen-ndarray"`.

### Configuration

This data type requires a configuration object with exactly two keys:

- `"dtype"`: A string naming the scalar data type of the inner arrays. This
  MUST be one of the fixed-size boolean, integer, floating point, or complex
  [core data types](https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html#core-data-types):
  `"bool"`, `"int8"`, `"int16"`, `"int32"`, `"int64"`, `"uint8"`, `"uint16"`,
  `"uint32"`, `"uint64"`, `"float16"`, `"float32"`, `"float64"`,
  `"complex64"`, or `"complex128"`. Variable-size and parameterized data
  types MUST NOT be used.
- `"inner_shape"`: A JSON array of integers, each greater than or equal to 1,
  giving the fixed trailing dimensions of every element. An empty array means
  each element is a one-dimensional array of shape `(n,)`.

An element of an array with this data type is an ndarray of shape
`(n, *inner_shape)` with scalar type `dtype`, where `n` is a non-negative
integer that may differ between elements. `n` is not stored in metadata; it is
implied by each element's encoded byte length.

### Examples

Array metadata for a one-dimensional array whose elements are `(n, 2)` float32
arrays:

```json
{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [12288],
  "data_type": {
    "name": "vlen-ndarray",
    "configuration": {
      "dtype": "float32",
      "inner_shape": [2]
    }
  },
  "chunk_grid": {
    "name": "regular",
    "configuration": {"chunk_shape": [1024]}
  },
  "chunk_key_encoding": {"name": "default"},
  "fill_value": "",
  "codecs": [
    {"name": "vlen-ndarray"},
    {"name": "zstd", "configuration": {"level": 3, "checksum": false}}
  ]
}
```

The following configuration describes elements that are one-dimensional
`(n,)` uint64 arrays:

```json
{
  "dtype": "uint64",
  "inner_shape": []
}
```

## Fill value representation

The `fill_value` metadata member MUST be a string containing the
[base64](https://en.wikipedia.org/wiki/Base64)-encoded raw little-endian bytes
of the fill element, whose byte length MUST be a whole multiple of the
element item size (the size in bytes of one `(1, *inner_shape)` item). The
empty string encodes the empty element of shape `(0, *inner_shape)`, which is
the recommended and default fill value.

## Codecs

Arrays with this data type MUST use the
[`vlen-ndarray`](../../codecs/vlen-ndarray/README.md) `array -> bytes` codec
(directly, or as the inner codec chain of the
[`sharding_indexed`](../../codecs/sharding_indexed/README.md) codec).

## Notes on interoperability and compatibility

- The encoded chunk representation is byte-identical to the
  [`bytes`](../bytes/README.md) data type with the
  [`vlen-bytes`](../../codecs/vlen-bytes/README.md) codec, where each element
  is the raw little-endian bytes of the element array in C (row-major) order.
  Implementations without `vlen-ndarray` support can therefore fall back to
  reading such arrays as `bytes` + `vlen-bytes` after a metadata substitution,
  and existing `bytes`-typed stores following this convention can be upgraded
  by rewriting metadata only.
- A reference implementation is provided by the
  [`zarr-vlen-ndarray`](https://github.com/espg/zarr-vlen-ndarray) Python
  package (registered with zarr-python via entry points).
- This data type serves the `zagg-ragged/2` store revision of the
  [zagg](https://github.com/englacial/zagg) aggregation pipeline; the `/1`
  revision is the `bytes`-typed convention described above.

## Change log

No changes yet.

## Current maintainers

* [Shane Grigsby](https://github.com/espg)
