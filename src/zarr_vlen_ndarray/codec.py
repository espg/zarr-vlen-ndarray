"""The ``vlen-ndarray`` array->bytes codec.

Converts the in-memory object-array-of-ndarrays to/from an object-array-of-
bytes and delegates the wire format to numcodecs' ``VLenBytes`` — the exact
codec behind Zarr's registered ``vlen-bytes`` codec. Encoded chunk bytes are
therefore byte-identical to what ``vlen-bytes`` produces for the equivalent
raw little-endian payloads (each element serialized as
``ndarray.astype('<...').tobytes()`` in C order). See SPEC.md.

The codec is parameter-free: the inner scalar dtype and inner_shape come from
the array's ``vlen-ndarray`` data type at encode/decode time, so it MUST be
paired with that data type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numcodecs.vlen import VLenBytes
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.common import parse_named_configuration

from zarr_vlen_ndarray.dtype import VlenNDArray

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.common import JSON, ChunkCoords

CODEC_NAME = "vlen-ndarray"

# Parameter-free, so a module-level instance is fine (mirrors zarr's vlen codecs).
_vlen_bytes_codec = VLenBytes()


@dataclass(frozen=True)
class VlenNDArrayCodec(ArrayBytesCodec):
    """Array->bytes codec for the ``vlen-ndarray`` data type."""

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, CODEC_NAME, require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": CODEC_NAME, "configuration": {}}

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return self

    # `dtype` / `chunk_grid` are annotated Any because their supertype
    # annotations differ across supported zarr versions (3.1.x uses ChunkGrid,
    # 3.2.x chunk-grid metadata unions).
    def validate(
        self,
        *,
        shape: ChunkCoords,
        dtype: Any,
        chunk_grid: Any,
    ) -> None:
        if not isinstance(dtype, VlenNDArray):
            raise ValueError(
                f"The {CODEC_NAME!r} codec is only compatible with the "
                f"{VlenNDArray._zarr_v3_name!r} data type, got {dtype}."
            )

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        zdtype = chunk_spec.dtype
        assert isinstance(zdtype, VlenNDArray)
        raw_bytes = chunk_bytes.as_array_like()
        payloads = _vlen_bytes_codec.decode(raw_bytes)
        assert payloads.dtype == np.object_
        decoded = np.empty(payloads.shape, dtype=object)
        for i, payload in enumerate(payloads):
            decoded[i] = zdtype.payload_to_cell(payload)
        decoded = decoded.reshape(chunk_spec.shape)
        return chunk_spec.prototype.nd_buffer.from_numpy_array(decoded)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        zdtype = chunk_spec.dtype
        assert isinstance(zdtype, VlenNDArray)
        cells = chunk_array.as_numpy_array().reshape(-1)
        as_bytes = np.empty(cells.shape, dtype=object)
        for i, cell in enumerate(cells):
            if cell is None:
                as_bytes[i] = b""
            else:
                as_bytes[i] = zdtype.coerce_cell(cell).tobytes()
        return chunk_spec.prototype.buffer.from_bytes(_vlen_bytes_codec.encode(as_bytes))

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        # Same posture as zarr's vlen codecs: size is data-dependent.
        raise NotImplementedError(
            "compute_encoded_size is not implemented for variable-length codecs"
        )
