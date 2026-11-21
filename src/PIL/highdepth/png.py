#
# The Python Imaging Library.
#
# Strict, streaming PNG decoder used by :mod:`PIL.highdepth`.
#
# See the README file for information on usage and redistribution.
#
"""PNG decoder.

The decoder validates every chunk length and the IEND position *before*
inflating image data, so truncated files, absurd chunk lengths and trailing
garbage produce stable errors instead of large allocations.  Bit depths 1, 2,
4, 8 and 16 are all preserved: 16-bit data stays 16-bit in the returned
:class:`~PIL.highdepth.model.ImageData`.
"""

from __future__ import annotations

import struct
import zlib
from array import array
from typing import TYPE_CHECKING

from .codecs import deflate_decode, unfilter_row, unpack_bits
from .errors import (
    ChecksumError,
    ChunkLengthError,
    FormatError,
    IllegalPaletteError,
    TrailingDataError,
    TruncatedDataError,
    UnsupportedFeatureError,
)
from .model import (
    ImageData,
    Palette,
    check_image_size,
    checked_add,
    checked_mul,
    row_bytes,
    storage_bytes,
)
from .source import Source, open_source

if TYPE_CHECKING:
    from .budget import CancellationToken, MemoryBudget

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MAX_CHUNK = 2**31 - 1

# colour type -> (channels, allowed depths)
_COLOR_TYPES = {
    0: (1, (1, 2, 4, 8, 16)),
    2: (3, (8, 16)),
    3: (1, (1, 2, 4, 8)),
    4: (2, (8, 16)),
    6: (4, (8, 16)),
}

_ADAM7 = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)

_COMPRESSION = 0
_FILTER_METHOD = 0
_INTERLACE_NONE = 0
_INTERLACE_ADAM7 = 1


class _Chunk:
    __slots__ = ("type", "data", "offset")

    def __init__(self, type_: bytes, data: bytes, offset: int) -> None:
        self.type = type_
        self.data = data
        self.offset = offset


class _ApngFrame:
    """One animation frame: its compressed data plus its control geometry."""

    __slots__ = ("idat", "geometry")

    def __init__(self, idat: list[bytes], geometry: tuple[int, ...]) -> None:
        self.idat = idat
        self.geometry = geometry


Control = tuple[int, int, int, int, int, int, int]


def _iter_chunks(source: Source) -> list[_Chunk]:
    """Split a PNG into chunks, validating every length against the source size."""
    if (
        source.size < len(PNG_MAGIC)
        or source.read_exact(0, len(PNG_MAGIC), "PNG signature") != PNG_MAGIC
    ):
        msg = "not a PNG stream"
        raise FormatError(msg)
    chunks: list[_Chunk] = []
    offset = len(PNG_MAGIC)
    seen_iend = False
    while not seen_iend:
        if offset + 8 > source.size:
            msg = f"truncated PNG: chunk header at offset {offset}"
            raise TruncatedDataError(msg)
        raw = source.read_exact(offset, 8, "chunk header")
        (length,) = struct.unpack(">I", raw[:4])
        type_ = raw[4:]
        if length > MAX_CHUNK:
            msg = f"PNG chunk length {length} exceeds the maximum"
            raise ChunkLengthError(msg)
        if not all(65 <= byte <= 90 or 97 <= byte <= 122 for byte in type_):
            msg = f"invalid PNG chunk type {type_!r}"
            raise FormatError(msg)
        payload_offset = offset + 8
        crc_offset = checked_add(payload_offset, length, "PNG chunk")
        if crc_offset + 4 > source.size:
            msg = (
                f"truncated PNG: chunk {type_.decode('latin-1')} declares "
                f"{length} bytes but the file ends early"
            )
            raise TruncatedDataError(msg)
        data = source.read_exact(payload_offset, length, "chunk data")
        (crc,) = struct.unpack(">I", source.read_exact(crc_offset, 4, "chunk crc"))
        if zlib.crc32(type_ + data) & 0xFFFFFFFF != crc:
            msg = f"bad CRC for PNG chunk {type_.decode('latin-1')}"
            raise ChecksumError(msg)
        chunks.append(_Chunk(type_, data, offset))
        offset = crc_offset + 4
        if type_ == b"IEND":
            seen_iend = True
    if offset != source.size:
        msg = f"{source.size - offset} trailing bytes after PNG IEND"
        raise TrailingDataError(msg)
    if not chunks or chunks[0].type != b"IHDR":
        msg = "PNG stream does not start with IHDR"
        raise FormatError(msg)
    return chunks


def _parse_ihdr(chunk: _Chunk) -> tuple[int, int, int, int, int, int]:
    if len(chunk.data) != 13:
        msg = f"IHDR has {len(chunk.data)} bytes, expected 13"
        raise ChunkLengthError(msg)
    width, height, depth, color, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", chunk.data
    )
    if width == 0 or height == 0:
        msg = f"PNG has zero size {(width, height)}"
        raise FormatError(msg)
    if color not in _COLOR_TYPES:
        msg = f"invalid PNG colour type {color}"
        raise FormatError(msg)
    channels, depths = _COLOR_TYPES[color]
    if depth not in depths:
        msg = f"invalid PNG bit depth {depth} for colour type {color}"
        raise FormatError(msg)
    if compression != _COMPRESSION:
        msg = f"unsupported PNG compression method {compression}"
        raise UnsupportedFeatureError(msg)
    if filter_method != _FILTER_METHOD:
        msg = f"unsupported PNG filter method {filter_method}"
        raise UnsupportedFeatureError(msg)
    if interlace not in (_INTERLACE_NONE, _INTERLACE_ADAM7):
        msg = f"unsupported PNG interlace method {interlace}"
        raise UnsupportedFeatureError(msg)
    return width, height, depth, color, channels, interlace


def _parse_plte(chunk: _Chunk, color: int) -> tuple[tuple[int, int, int], ...]:
    if color in (0, 4):
        msg = "PLTE chunk is not allowed for grayscale PNG"
        raise FormatError(msg)
    if len(chunk.data) % 3 or not 0 < len(chunk.data) <= 768:
        msg = f"PLTE has invalid length {len(chunk.data)}"
        raise ChunkLengthError(msg)
    return tuple(
        (chunk.data[i], chunk.data[i + 1], chunk.data[i + 2])
        for i in range(0, len(chunk.data), 3)
    )


def _parse_trns(
    chunk: _Chunk, color: int, depth: int, entries: int
) -> tuple[int | None, bytes | None]:
    data = chunk.data
    if color == 0:
        if len(data) != 2:
            msg = f"tRNS for grayscale must be 2 bytes, got {len(data)}"
            raise ChunkLengthError(msg)
        (value,) = struct.unpack(">H", data)
        if value > (1 << depth) - 1:
            msg = "tRNS value exceeds the sample range"
            raise FormatError(msg)
        return value, None
    if color == 2:
        if len(data) != 6:
            msg = f"tRNS for truecolour must be 6 bytes, got {len(data)}"
            raise ChunkLengthError(msg)
        return None, data
    if color == 3:
        if len(data) > entries:
            msg = f"tRNS has {len(data)} entries but PLTE has {entries}"
            raise ChunkLengthError(msg)
        return None, data
    if data:
        msg = "tRNS is not allowed for images with an alpha channel"
        raise FormatError(msg)
    return None, None


def _tile_size(
    width: int, height: int, interlace: int
) -> list[tuple[int, int, int, int, int, int]]:
    """Return ``(x0, y0, sub_width, sub_height, dx, dy)`` passes."""
    if interlace == _INTERLACE_NONE:
        return [(0, 0, width, height, 1, 1)]
    passes = []
    for x0, y0, dx, dy in _ADAM7:
        sub_width = (width - x0 + dx - 1) // dx if width > x0 else 0
        sub_height = (height - y0 + dy - 1) // dy if height > y0 else 0
        passes.append((x0, y0, sub_width, sub_height, dx, dy))
    return passes


def decode_png(
    source: object,
    *,
    budget: MemoryBudget | None = None,
    token: CancellationToken | None = None,
) -> ImageData:
    """Decode a PNG from any supported source into an :class:`ImageData`.

    The returned samples are planar-interleaved, top-left origin.  Interlaced
    images are de-interlaced.  Palette images keep their indices and gain an
    RGBA palette with any ``tRNS`` alpha merged in.
    """
    src = open_source(source, budget=budget, token=token)
    owns = src is not source
    try:
        return _decode_png(src, budget, token)
    finally:
        if owns:
            src.close()


def _decode_png(
    src: Source, budget: MemoryBudget | None, token: CancellationToken | None
) -> ImageData:
    if token is not None:
        token.raise_if_cancelled()
    chunks = _iter_chunks(src)
    width, height, depth, color, channels, interlace = _parse_ihdr(chunks[0])
    check_image_size(width, height)

    # Reject over-budget allocations before inflating anything.
    if budget is not None:
        storage = storage_bytes(width, height, channels, 8 if depth <= 8 else 16)
        budget.check(storage, "PNG image")

    palette_entries: tuple[tuple[int, int, int], ...] | None = None
    transparency_value: int | None = None
    transparency_bytes: bytes | None = None
    idat: list[bytes] = []
    seen_plte = False
    seen_trns = False
    seen_idat = False
    seen_iend = False
    metadata: dict[str, object] = {}
    frames: list[_ApngFrame] = []
    animation: tuple[int, int] | None = None
    pending_control: Control | None = None

    for chunk in chunks[1:]:
        if token is not None:
            token.raise_if_cancelled()
        ctype = chunk.type
        if ctype == b"IDAT":
            if not seen_idat:
                if color == 3 and not seen_plte:
                    msg = "palette PNG has no PLTE before IDAT"
                    raise IllegalPaletteError(msg)
                seen_idat = True
            idat.append(chunk.data)
            if pending_control is not None:
                frames.append(_ApngFrame(list(idat), pending_control))
                pending_control = None
        elif ctype == b"fdAT":
            if len(chunk.data) < 4:
                msg = "fdAT chunk is shorter than its sequence number"
                raise ChunkLengthError(msg)
            if pending_control is None:
                msg = "fdAT chunk without a preceding fcTL"
                raise FormatError(msg)
            frames.append(_ApngFrame([chunk.data[4:]], pending_control))
            pending_control = None
        elif ctype == b"fcTL":
            if len(chunk.data) != 26:
                msg = f"fcTL chunk has {len(chunk.data)} bytes, expected 26"
                raise ChunkLengthError(msg)
            sequence, fw, fh, fx, fy, _delay_num, _delay_den, dispose, blend = (
                struct.unpack(">IIIIIHHBB", chunk.data)
            )
            del sequence, dispose, blend
            if pending_control is not None:
                msg = "consecutive fcTL chunks with no frame data"
                raise FormatError(msg)
            pending_control = (fw, fh, fx, fy, 0, 0, 0)
        elif ctype == b"acTL":
            if len(chunk.data) != 8:
                msg = f"acTL chunk has {len(chunk.data)} bytes, expected 8"
                raise ChunkLengthError(msg)
            frame_total, plays = struct.unpack(">II", chunk.data)
            if frame_total == 0:
                msg = "acTL declares zero frames"
                raise FormatError(msg)
            animation = (frame_total, plays)
        elif ctype == b"IEND":
            seen_iend = True
            break
        elif ctype == b"PLTE":
            if seen_idat:
                msg = "PLTE appears after IDAT"
                raise FormatError(msg)
            if seen_plte:
                msg = "duplicate PLTE"
                raise FormatError(msg)
            palette_entries = _parse_plte(chunk, color)
            seen_plte = True
        elif ctype == b"tRNS":
            if seen_trns or seen_idat:
                msg = "misplaced or duplicate tRNS"
                raise FormatError(msg)
            transparency_value, transparency_bytes = _parse_trns(
                chunk, color, depth, len(palette_entries or ())
            )
            seen_trns = True
        elif ctype in {b"iCCP", b"sRGB", b"gAMA", b"pHYs", b"eXIf", b"cHRM", b"sBIT"}:
            metadata[ctype.decode("latin-1")] = chunk.data
        elif ctype[0:1].isupper():
            msg = f"unknown critical PNG chunk {ctype.decode('latin-1')}"
            raise UnsupportedFeatureError(msg)
        # Unknown ancillary chunks are ignored, as the specification permits.

    if not seen_iend:
        msg = "PNG stream has no IEND"
        raise FormatError(msg)
    if not idat:
        msg = "PNG has no IDAT chunk"
        raise FormatError(msg)
    if color == 3:
        if palette_entries is None:
            msg = "palette PNG has no PLTE"
            raise IllegalPaletteError(msg)
        if len(palette_entries) > (1 << depth):
            msg = "PLTE has more entries than the bit depth allows"
            raise IllegalPaletteError(msg)
    elif palette_entries is not None and color != 2:
        msg = "PLTE is not allowed for this colour type"
        raise FormatError(msg)

    if color == 3:
        entries = list(palette_entries or ())
        alpha = [255] * len(entries)
        if transparency_bytes is not None:
            for index, value in enumerate(transparency_bytes):
                alpha[index] = value
        palette = Palette(
            tuple((*entries[index], alpha[index]) for index in range(len(entries)))
        )
    else:
        palette = None

    stride = row_bytes(width, channels, depth)
    raw_length = 0
    for _x0, _y0, sub_w, sub_h, _dx, _dy in _tile_size(width, height, interlace):
        if sub_w == 0 or sub_h == 0:
            continue
        sub_stride = row_bytes(sub_w, channels, depth)
        raw_length = checked_add(
            raw_length,
            checked_mul(sub_h, sub_stride + 1, "inflated size"),
            "inflated size",
        )
    if budget is not None:
        # The inflated buffer is only used while unfiltering one pass at a time.
        budget.check(raw_length, "PNG inflated data")

    samples = _inflate_image(
        idat, width, height, channels, depth, interlace, raw_length, token
    )

    transparency: int | tuple[int, ...] | None = None
    if transparency_value is not None:
        transparency = transparency_value
    elif color == 2 and transparency_bytes is not None:
        transparency = struct.unpack(">HHH", transparency_bytes)

    bit_depth = depth
    stored: bytes | array[int]
    if color == 3:
        mode = "P"
        storage_bits = 8
        stored_channels = 1
        # Indices are kept unpacked in 8-bit storage; ``bit_depth`` records the
        # original storage depth for round-tripping and palette sizing.
        stored = bytes(samples)
    elif depth <= 8:
        mode = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}[channels]
        storage_bits = 8
        stored_channels = channels
        stored = bytes(samples)
    else:
        mode = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}[channels]
        storage_bits = 16
        stored_channels = channels
        stored = array("H", samples)

    if animation is not None:
        declared, plays = animation
        if len(frames) != declared:
            msg = (
                f"acTL declares {declared} frames but the stream contains "
                f"{len(frames)}"
            )
            raise ChunkLengthError(msg)
        metadata["apng_frames"] = declared
        metadata["apng_plays"] = plays
        metadata["apng_control"] = tuple(frame.geometry for frame in frames)
        metadata["apng_fdats"] = tuple(frame.idat for frame in frames)

    if color == 3:
        data = ImageData(
            format="PNG",
            mode="P",
            width=width,
            height=height,
            bit_depth=depth,
            storage_bits=8,
            channels=1,
            samples=stored,
            palette=palette,
            transparency=None,
            row_stride=stride,
            interlaced=interlace == _INTERLACE_ADAM7,
            info=metadata,
        )
    else:
        data = ImageData(
            format="PNG",
            mode=mode,
            width=width,
            height=height,
            bit_depth=bit_depth,
            storage_bits=storage_bits,
            channels=stored_channels,
            samples=stored,
            transparency=transparency,
            row_stride=stride,
            interlaced=interlace == _INTERLACE_ADAM7,
            info=metadata,
        )
    return data


def decode_png_frame(data: ImageData, page: int) -> ImageData:
    """Decode an APNG frame (``page >= 1``) from data already parsed.

    Frame 0 is the default image returned by :func:`decode_png`.  Later frames
    are decoded from the ``fdAT`` payloads recorded in ``data.info``.  Each
    frame is a complete image at the animation's canvas size; the control
    rectangle is reported through ``info`` so callers can honour dispose and
    blend operations.
    """
    fdats = data.info.get("apng_fdats")
    controls = data.info.get("apng_control")
    if not fdats or controls is None:
        msg = f"frame {page} is not available; the stream is not animated"
        raise IndexError(msg)
    if not 0 <= page < len(fdats):
        msg = f"APNG frame {page} out of range ({len(fdats)} frames)"
        raise IndexError(msg)

    depth = data.bit_depth if data.mode != "P" else data.bit_depth
    channels = data.channels
    stride = data.row_length_bytes
    width, height = data.width, data.height
    interlace = 1 if data.interlaced else 0
    raw_length = 0
    for _x0, _y0, sub_w, sub_h, _dx, _dy in _tile_size(width, height, interlace):
        if sub_w == 0 or sub_h == 0:
            continue
        raw_length = checked_add(
            raw_length,
            checked_mul(sub_h, row_bytes(sub_w, channels, depth) + 1, "inflated size"),
            "inflated size",
        )
    samples = _inflate_image(
        list(fdats[page]), width, height, channels, depth, interlace, raw_length, None
    )
    if data.storage_bits == 16:
        stored: bytes | array[int] = array("H", samples)
    else:
        stored = bytes(samples)
    info = {key: value for key, value in data.info.items() if key != "apng_fdats"}
    geometry = controls[page]
    info["frame"] = page
    info["offset_x"] = geometry[2]
    info["offset_y"] = geometry[3]
    return ImageData(
        format="PNG",
        mode=data.mode,
        width=width,
        height=height,
        bit_depth=data.bit_depth,
        storage_bits=data.storage_bits,
        channels=channels,
        samples=stored,
        palette=data.palette,
        transparency=data.transparency,
        row_stride=stride,
        interlaced=data.interlaced,
        info=info,
    )


def _inflate_image(
    idat: list[bytes],
    width: int,
    height: int,
    channels: int,
    depth: int,
    interlace: int,
    raw_length: int,
    token: CancellationToken | None,
) -> list[int]:
    """Inflate, unfilter and de-interlace one image or one animation frame."""
    raw = deflate_decode(b"".join(idat), raw_length)
    bpp = max(1, (channels * depth + 7) // 8)
    samples: list[int] = [0] * (width * height * channels)
    position = 0
    for x0, y0, sub_w, sub_h, dx, dy in _tile_size(width, height, interlace):
        if sub_w == 0 or sub_h == 0:
            continue
        sub_stride = row_bytes(sub_w, channels, depth)
        previous: bytes | None = None
        for row_index in range(sub_h):
            if token is not None:
                token.raise_if_cancelled()
            filter_type = raw[position]
            position += 1
            row = bytearray(raw[position : position + sub_stride])
            position += sub_stride
            decoded = unfilter_row(filter_type, row, previous, bpp)
            previous = decoded
            values = unpack_bits(decoded, depth, sub_w * channels)
            y = y0 + row_index * dy
            base = y * width * channels
            if dx == 1 and x0 == 0:
                start = base
                samples[start : start + sub_w * channels] = values
            else:
                for index, value in enumerate(values):
                    x = x0 + (index // channels) * dx
                    samples[base + x * channels + index % channels] = value
    return samples


__all__ = ["PNG_MAGIC", "decode_png"]
