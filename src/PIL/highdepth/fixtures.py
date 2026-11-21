#
# The Python Imaging Library.
#
# Deterministic offline fixtures for the high depth test-suite.
#
# See the README file for information on usage and redistribution.
#
"""Fixture construction in memory.

Every fixture is *generated* by deterministic code at test time rather than
checked in as a pre-made image, so the bytes exercised by the suite are always
the bytes the writer produced.  The generators cover the failure modes the
suite cares about: truncated rows, absurd chunk lengths, missing palettes,
trailing bytes, and an animation with several frames.
"""

from __future__ import annotations

import struct
import zlib
from typing import TYPE_CHECKING

from .codecs import pack_bits

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .model import ImageData, Palette


def _png_chunk(type_: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + type_
        + payload
        + struct.pack(">I", zlib.crc32(type_ + payload) & 0xFFFFFFFF)
    )


def png_chunk(type_: bytes, payload: bytes) -> bytes:
    """Build a PNG chunk with a valid CRC (public for corruption tests)."""
    return _png_chunk(type_, payload)


def png_ihdr(
    width: int,
    height: int,
    depth: int,
    color_type: int,
    interlace: int = 0,
    compression: int = 0,
    filter_method: int = 0,
) -> bytes:
    return _png_chunk(
        b"IHDR",
        struct.pack(
            ">IIBBBBB",
            width,
            height,
            depth,
            color_type,
            compression,
            filter_method,
            interlace,
        ),
    )


def grayscale16_png(values: Sequence[int], width: int, height: int) -> bytes:
    """A 16-bit grayscale PNG built from exact sample values."""
    assert len(values) == width * height
    payload = bytearray()
    for y in range(height):
        payload.append(0)
        payload += pack_bits(list(values[y * width : (y + 1) * width]), 16)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_ihdr(width, height, 16, 0)
        + _png_chunk(b"IDAT", zlib.compress(bytes(payload), 9))
        + _png_chunk(b"IEND", b"")
    )


def rgb16_png(values: Sequence[int], width: int, height: int) -> bytes:
    """A 16-bit RGB PNG; ``values`` is interleaved R,G,B per pixel."""
    assert len(values) == width * height * 3
    payload = bytearray()
    for y in range(height):
        payload.append(0)
        payload += pack_bits(list(values[y * width * 3 : (y + 1) * width * 3]), 16)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_ihdr(width, height, 16, 2)
        + _png_chunk(b"IDAT", zlib.compress(bytes(payload), 9))
        + _png_chunk(b"IEND", b"")
    )


def palette_png(
    indices: Sequence[int],
    width: int,
    height: int,
    palette: Palette,
    depth: int = 8,
    transparency: bytes | None = None,
) -> bytes:
    """A palette PNG at the requested index depth."""
    assert len(indices) == width * height
    payload = bytearray()
    for y in range(height):
        payload.append(0)
        payload += pack_bits(list(indices[y * width : (y + 1) * width]), depth)
    chunks = (
        b"\x89PNG\r\n\x1a\n"
        + png_ihdr(width, height, depth, 3)
        + _png_chunk(b"PLTE", palette.rgb_bytes()[: 256 * 3])
    )
    if transparency is None and palette.has_alpha:
        transparency = palette.alpha_bytes()
    if transparency is not None:
        chunks += _png_chunk(b"tRNS", transparency)
    return (
        chunks
        + _png_chunk(b"IDAT", zlib.compress(bytes(payload), 9))
        + _png_chunk(b"IEND", b"")
    )


def truncated_png(payload: bytes, keep: int) -> bytes:
    """Cut ``payload`` to ``keep`` bytes, producing a truncated file."""
    return payload[:keep]


def png_with_trailing_bytes(payload: bytes, extra: bytes = b"garbage") -> bytes:
    return payload + extra


def png_with_bad_chunk_length(payload: bytes) -> bytes:
    """Rewrite the IDAT length field to an impossible value, fixing nothing."""
    position = payload.find(b"IDAT")
    assert position > 0
    length_offset = position - 4
    return payload[:length_offset] + struct.pack(">I", 0x7FFFFFFF) + payload[position:]


def png_without_palette(indices: Sequence[int], width: int, height: int) -> bytes:
    """A colour-type-3 PNG with no PLTE chunk."""
    payload = bytearray()
    for y in range(height):
        payload.append(0)
        payload += bytes(indices[y * width : (y + 1) * width])
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_ihdr(width, height, 8, 3)
        + _png_chunk(b"IDAT", zlib.compress(bytes(payload), 9))
        + _png_chunk(b"IEND", b"")
    )


def png_with_truncated_row(values: Sequence[int], width: int, height: int) -> bytes:
    """Declare ``height`` rows but inflate one row short."""
    payload = bytearray()
    for y in range(height - 1):
        payload.append(0)
        payload += pack_bits(list(values[y * width : (y + 1) * width]), 16)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_ihdr(width, height, 16, 0)
        + _png_chunk(b"IDAT", zlib.compress(bytes(payload), 9))
        + _png_chunk(b"IEND", b"")
    )


def bmp_rgb565(
    values: Sequence[int], width: int, height: int, top_down: bool = False
) -> bytes:
    """A 16 bpp BITFIELDS BMP with RGB565 masks."""
    stride = ((width * 16 + 31) // 32) * 4
    pixels = bytearray()
    rows = range(height) if top_down else range(height - 1, -1, -1)
    for row in rows:
        line = bytearray()
        for x in range(width):
            line += struct.pack("<H", values[row * width + x])
        line += b"\x00" * (stride - len(line))
        pixels += line
    masks = struct.pack("<III", 0xF800, 0x07E0, 0x001F)
    dib = struct.pack(
        "<IiiHHIIiiII",
        40,
        width,
        -height if top_down else height,
        1,
        16,
        3,
        len(pixels),
        2835,
        2835,
        0,
        0,
    )
    data_offset = 14 + 40 + 12
    header = b"BM" + struct.pack("<IHHI", data_offset + len(pixels), 0, 0, data_offset)
    return header + dib + masks + bytes(pixels)


def bmp_palette8(
    indices: Sequence[int], width: int, height: int, palette: Palette
) -> bytes:
    stride = ((width * 8 + 31) // 32) * 4
    pixels = bytearray()
    for row in range(height - 1, -1, -1):
        line = bytearray(indices[row * width : (row + 1) * width])
        line += b"\x00" * (stride - len(line))
        pixels += line
    colors = bytearray()
    for red, green, blue, _alpha in palette.entries:
        colors += bytes((blue, green, red, 0))
    dib = struct.pack(
        "<IiiHHIIiiII",
        40,
        width,
        height,
        1,
        8,
        0,
        len(pixels),
        2835,
        2835,
        len(palette),
        0,
    )
    data_offset = 14 + 40 + len(colors)
    header = b"BM" + struct.pack("<IHHI", data_offset + len(pixels), 0, 0, data_offset)
    return header + dib + bytes(colors) + bytes(pixels)


def bmp_truncated(payload: bytes, keep: int) -> bytes:
    return payload[:keep]


def tiff_with_truncated_strip(payload: bytes, strip_length: int = 1000) -> bytes:
    """set a strip byte count larger than the remaining file."""
    position = payload.find(b"II\x2a\x00")
    assert position == 0
    # Place a copy of the strip-count tag value out of range by truncating the
    # file: the writer's strip data is at offset 8 for a single-strip page.
    return payload[: 8 + max(1, strip_length // 4)]


def multiftiff_pages(frames: Sequence[ImageData]) -> list[ImageData]:
    """Return the frames unchanged; the writer performs the multi-page write."""
    return list(frames)


def render_reference_tiles(
    data: ImageData, tile: int = 3
) -> list[tuple[tuple[int, int], list[tuple[int, ...]]]]:
    """Extract reference tiles for per-channel comparisons.

    Returns ``[((x, y), [pixel, ...]), ...]`` with pixels row-major within the
    tile.  Used to check that streaming tiles match a full decode cell by cell.
    """
    out = []
    for y in range(0, data.height, tile):
        for x in range(0, data.width, tile):
            pixels = [
                data.sample_at(column, row)
                for row in range(y, min(y + tile, data.height))
                for column in range(x, min(x + tile, data.width))
            ]
            out.append(((x, y), pixels))
    return out


__all__ = [
    "bmp_palette8",
    "bmp_rgb565",
    "bmp_truncated",
    "grayscale16_png",
    "multiftiff_pages",
    "palette_png",
    "png_chunk",
    "png_ihdr",
    "png_with_bad_chunk_length",
    "png_with_trailing_bytes",
    "png_with_truncated_row",
    "png_without_palette",
    "render_reference_tiles",
    "rgb16_png",
    "tiff_with_truncated_strip",
    "truncated_png",
]
