# Vlen-ndarray codec

Defines an `array -> bytes` codec that serializes arrays of variable-length
n-dimensional array elements, as described by the
[`vlen-ndarray`](../../data-types/vlen-ndarray/README.md) data type.

## Codec name

The value of the `name` member in the codec object MUST be `vlen-ndarray`.

## Configuration parameters

None. The element scalar data type and inner shape come from the array's
`vlen-ndarray` data type configuration.

## Example

For example, the array metadata below specifies that the array contains
variable-length `(n, 2)` float32 elements:

```json
{
    "data_type": {
        "name": "vlen-ndarray",
        "configuration": {"dtype": "float32", "inner_shape": [2]}
    },
    "codecs": [{
        "name": "vlen-ndarray"
    }]
}
```

## Format and algorithm

This is an `array -> bytes` codec.

This codec is only compatible with the
[`"vlen-ndarray"`](../../data-types/vlen-ndarray/README.md) data type.

Each element of shape `(n, *inner_shape)` is serialized to its raw bytes in
little-endian byte order and C (row-major) element order; the element of
`n = 0` serializes to zero bytes. The resulting variable-length byte strings
are then framed exactly as in the [`vlen-bytes`](../vlen-bytes/README.md)
codec: the encoded chunk is prefixed with a 32-bit little-endian unsigned
integer (u32le) giving the number of elements in the chunk, followed by a
sequence of encoded elements in lexicographical order, each encoded by a
u32le byte count followed by the bytes themselves.

Consequently, for any chunk, the encoded representation is byte-identical to
encoding the per-element raw bytes with the `vlen-bytes` codec. On decode,
each element's item count `n` is recovered from its byte length, which MUST
be a whole multiple of the element item size; decoders MUST report an error
otherwise.

See https://numcodecs.readthedocs.io/en/stable/other/vlen.html#vlenbytes for
details about the framing.

## Change log

No changes yet.

## Current maintainers

* [Shane Grigsby](https://github.com/espg)
