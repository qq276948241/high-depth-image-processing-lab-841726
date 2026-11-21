#
# The Python Imaging Library.
#
# Deterministic serialization of the normalized model and quantizer output.
#
# See the README file for information on usage and redistribution.
#
"""Canonical serialization.

Round-tripping through this module is lossless for every field the model can
hold, including 16-bit samples, palettes with alpha and transparency.  It is
used by the reproducibility tests and by the offline fixture generator.
"""

from __future__ import annotations

import json
import sys
from array import array
from typing import TYPE_CHECKING

from .errors import FormatError
from .model import ImageData, Palette

if TYPE_CHECKING:
    from typing import Any

_MAGIC = b"HDIM\x01"


def dump(data: ImageData) -> bytes:
    """Serialize to a self-describing, versioned byte stream."""
    data_object: object = data
    if not isinstance(data_object, ImageData):
        msg = f"expected ImageData, got {type(data_object).__name__}"
        raise FormatError(msg)
    data = data_object
    header = {
        "format": data.format,
        "mode": data.mode,
        "width": data.width,
        "height": data.height,
        "bit_depth": data.bit_depth,
        "storage_bits": data.storage_bits,
        "channels": data.channels,
        "row_stride": data.row_stride,
        "orientation": data.orientation,
        "interlaced": data.interlaced,
        "transparency": data.transparency,
        "palette": data.palette.rgba_bytes().hex() if data.palette else None,
        "info": _jsonable(data.info),
        "sample_bytes": len(data.tobytes()),
    }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return _MAGIC + len(encoded).to_bytes(4, "big") + encoded + data.tobytes()


def load(payload: bytes) -> ImageData:
    """Inverse of :func:`dump`; rejects anything that is not a dump."""
    payload = bytes(payload)
    if not payload.startswith(_MAGIC):
        msg = "not a highdepth serialized image"
        raise FormatError(msg)
    offset = len(_MAGIC)
    if len(payload) < offset + 4:
        msg = "truncated highdepth header"
        raise FormatError(msg)
    length = int.from_bytes(payload[offset : offset + 4], "big")
    offset += 4
    if len(payload) < offset + length:
        msg = "truncated highdepth header"
        raise FormatError(msg)
    header: dict[str, Any] = json.loads(payload[offset : offset + length])
    offset += length
    samples = payload[offset:]
    if len(samples) != header.pop("sample_bytes"):
        msg = "sample payload length does not match the header"
        raise FormatError(msg)
    palette_hex = header.pop("palette")
    palette = Palette.from_rgba(bytes.fromhex(palette_hex)) if palette_hex else None
    info = header.pop("info")
    if header["storage_bits"] == 16:
        values = array("H")
        values.frombytes(samples)
        if sys.byteorder == "little":
            # ``dump`` writes canonical big-endian words.
            values.byteswap()
        stored: bytes | array[int] = values
    else:
        stored = samples
    return ImageData(samples=stored, palette=palette, info=info, **header)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


__all__ = ["dump", "load"]
