#
# The Python Imaging Library.
#
# Strict BMP decoder used by :mod:`PIL.highdepth`.
#
# See the README file for information on usage and redistribution.
#
"""BMP decoder.

Supports 1, 2, 4, 8, 16, 24 and 32 bit images, raw and RLE compressed, with a
bottom-up or top-down declaration.  Every image is returned top-left origin
with its declared bit depth.  16-bit RGB555/RGB565 and 32-bit images keep
eight bits per channel (the format's own precision); the bit depth field
records the on-disk sample width so callers can tell a 16 bpp file from a
24 bpp one.

The decoder computes the absolute offset of every row before reading it and
verifies it against the declared file size, so a truncated file, a short
palette or an overflowing stride fails before any pixel buffer is allocated.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from .errors import (
    ChunkLengthError,
    FormatError,
    IllegalPaletteError,
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

BMP_MAGIC = b"BM"
_BI_RGB = 0
_BI_RLE8 = 1
_BI_RLE4 = 2
_BI_BITFIELDS = 3
_BI_JPEG = 4
_BI_PNG = 5
_BI_ALPHABITFIELDS = 6

_HEADER_SIZES = {12, 40, 52, 56, 64, 108, 124}


def _parse_masks(
    source: Source, header_offset: int, header_size: int, bits: int
) -> tuple[tuple[int, int, int, int], int]:
    """Return the RGBA masks and the offset just past them."""
    if header_size >= 52:
        data = source.read_exact(
            header_offset + 40, 12 if header_size < 56 else 16, "BMP masks"
        )
        red, green, blue = struct.unpack("<III", data[:12])
        alpha = struct.unpack("<I", data[12:16])[0] if header_size >= 56 else 0
        return (red, green, blue, alpha), header_offset + header_size
    data = source.read_exact(header_offset + header_size, 12, "BMP masks")
    red, green, blue = struct.unpack("<III", data)
    return (red, green, blue, 0), header_offset + header_size + 12


def _mask_shift(mask: int) -> tuple[int, int]:
    if mask == 0:
        return 0, 0
    shift = 0
    while not mask & 1:
        mask >>= 1
        shift += 1
    width = 0
    while mask & 1:
        mask >>= 1
        width += 1
    return shift, width


def _extract_masked(value: int, mask: int) -> int:
    shift, width = _mask_shift(mask)
    if width == 0:
        return 0
    sample = (value >> shift) & ((1 << width) - 1)
    maximum = (1 << width) - 1
    return (sample * 255 + maximum // 2) // maximum


def _decode_rle(data: bytes, width: int, height: int, depth: int) -> list[int]:
    """Decode BI_RLE8 / BI_RLE4 into packed indices, top-left origin."""
    stride = (width * depth + 7) // 8
    pixels = [0] * (stride * height)
    x = 0
    y = height - 1
    position = 0
    end_of_bitmap = False
    while position + 1 < len(data) and not end_of_bitmap:
        count = data[position]
        value = data[position + 1]
        position += 2
        if count:
            run = count
            if depth == 8:
                samples = [value] * run
            else:
                samples = (
                    [(value >> 4) & 0xF, value & 0xF][:run]
                    if run <= 2
                    else [
                        (value >> 4) & 0xF if i % 2 == 0 else value & 0xF
                        for i in range(run)
                    ]
                )
            for sample in samples:
                if x >= width:
                    msg = "RLE run crosses the row boundary"
                    raise FormatError(msg)
                if depth == 8:
                    pixels[y * stride + x] = sample
                else:
                    byte_index = y * stride + x // 2
                    if x % 2 == 0:
                        pixels[byte_index] = (pixels[byte_index] & 0x0F) | (sample << 4)
                    else:
                        pixels[byte_index] = (pixels[byte_index] & 0xF0) | sample
                x += 1
        elif value == 0:  # end of line
            x = 0
            y -= 1
            if y < -1:
                msg = "RLE stream declares more rows than the image"
                raise FormatError(msg)
        elif value == 1:  # end of bitmap
            end_of_bitmap = True
        elif value == 2:  # delta
            if position + 1 >= len(data):
                msg = "truncated RLE delta"
                raise TruncatedDataError(msg)
            x += data[position]
            y -= data[position + 1]
            position += 2
        else:  # absolute mode
            run = value
            if depth == 8:
                literal = data[position : position + run]
                if len(literal) != run:
                    msg = "truncated RLE literal run"
                    raise TruncatedDataError(msg)
                position += run
                for sample in literal:
                    if x >= width:
                        msg = "RLE run crosses the row boundary"
                        raise FormatError(msg)
                    pixels[y * stride + x] = sample
                    x += 1
            else:
                byte_count = (run + 1) // 2
                literal = data[position : position + byte_count]
                if len(literal) != byte_count:
                    msg = "truncated RLE literal run"
                    raise TruncatedDataError(msg)
                position += byte_count
                for index in range(run):
                    byte = literal[index // 2]
                    sample = (byte >> 4) & 0xF if index % 2 == 0 else byte & 0xF
                    if x >= width:
                        msg = "RLE run crosses the row boundary"
                        raise FormatError(msg)
                    byte_index = y * stride + x // 2
                    if x % 2 == 0:
                        pixels[byte_index] = (pixels[byte_index] & 0x0F) | (sample << 4)
                    else:
                        pixels[byte_index] = (pixels[byte_index] & 0xF0) | sample
                    x += 1
            if position % 2:
                position += 1
    if not end_of_bitmap:
        msg = "RLE stream has no end-of-bitmap marker"
        raise TruncatedDataError(msg)
    return pixels


def decode_bmp(
    source: object,
    *,
    budget: MemoryBudget | None = None,
    token: CancellationToken | None = None,
) -> ImageData:
    """Decode a BMP from any supported source into an :class:`ImageData`."""
    src = open_source(source, budget=budget, token=token)
    owns = src is not source
    try:
        return _decode_bmp(src, budget, token)
    finally:
        if owns:
            src.close()


def _decode_bmp(
    src: Source, budget: MemoryBudget | None, token: CancellationToken | None
) -> ImageData:
    if token is not None:
        token.raise_if_cancelled()
    if src.size < 14:
        msg = "truncated BMP header"
        raise TruncatedDataError(msg)
    header = src.read_exact(0, 14, "BMP file header")
    if header[:2] != BMP_MAGIC:
        msg = "not a BMP stream"
        raise FormatError(msg)
    (declared_size,) = struct.unpack("<I", header[2:6])
    (data_offset,) = struct.unpack("<I", header[10:14])
    if declared_size not in (0, src.size):
        # Many writers leave this zero or count only the pixel data; accept it
        # but never trust it for bounds checking.
        if declared_size > src.size:
            msg = f"BMP declares {declared_size} bytes but only {src.size} exist"
            raise TruncatedDataError(msg)

    if data_offset < 14 or data_offset > src.size:
        msg = f"BMP data offset {data_offset} is outside the file"
        raise FormatError(msg)
    dib = src.read_exact(14, 8, "BMP DIB header")
    (header_size,) = struct.unpack("<I", dib[:4])
    if header_size not in _HEADER_SIZES:
        msg = f"unsupported BMP header size {header_size}"
        raise UnsupportedFeatureError(msg)
    if src.size < 14 + header_size:
        msg = "truncated BMP DIB header"
        raise TruncatedDataError(msg)
    dib_data = src.read_exact(14, header_size, "BMP DIB header")

    if header_size == 12:
        width, height, planes, bits = struct.unpack("<HHHH", dib_data[4:12])
        compression = _BI_RGB
        colors_used = 0
        palette_entry_size = 3
    else:
        width, height, planes, bits, compression = struct.unpack(
            "<iiHHI", dib_data[4:20]
        )
        (
            _image_size,
            _x_ppm,
            _y_ppm,
            colors_used,
            _important,
        ) = struct.unpack("<IIiiI", dib_data[20:40])
        palette_entry_size = 4

    if planes != 1:
        msg = f"BMP has {planes} planes, expected 1"
        raise FormatError(msg)
    top_down = height < 0
    height = abs(height)
    if width <= 0 or height <= 0:
        msg = f"invalid BMP size {(width, height)}"
        raise FormatError(msg)
    check_image_size(width, height)

    if bits == 32 and compression == _BI_RGB and header_size != 12:
        masks = (0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
        mask_end = 14 + header_size
    elif compression in (_BI_BITFIELDS, _BI_ALPHABITFIELDS):
        masks, mask_end = _parse_masks(src, 14, header_size, bits)
    else:
        masks = (0, 0, 0, 0)
        mask_end = 14 + header_size

    if compression == _BI_RGB:
        if bits not in (1, 2, 4, 8, 16, 24, 32):
            msg = f"unsupported BMP pixel depth {bits}"
            raise UnsupportedFeatureError(msg)
    elif compression in (_BI_RLE8, _BI_RLE4):
        if bits not in (8, 4) or (bits == 8) != (compression == _BI_RLE8):
            msg = f"RLE compression {compression} is inconsistent with {bits} bpp"
            raise FormatError(msg)
    elif compression in (_BI_BITFIELDS, _BI_ALPHABITFIELDS):
        if bits not in (16, 32):
            msg = f"bitfield compression is unsupported at {bits} bpp"
            raise UnsupportedFeatureError(msg)
    elif compression in (_BI_JPEG, _BI_PNG):
        msg = "embedded JPEG/PNG BMP streams are not supported"
        raise UnsupportedFeatureError(msg)
    else:
        msg = f"unsupported BMP compression {compression}"
        raise UnsupportedFeatureError(msg)

    if compression == _BI_RLE8 or compression == _BI_RLE4:
        storage = storage_bytes(width, height, 1, 8)
        if budget is not None:
            budget.check(storage, "BMP image")
        pixel_start = mask_end
        if data_offset < pixel_start:
            msg = "BMP data offset overlaps the headers"
            raise FormatError(msg)
        payload = src.read_exact(data_offset, src.size - data_offset, "BMP RLE data")
        if token is not None:
            token.raise_if_cancelled()
        indices = _decode_rle(payload, width, height, bits)
        rle_palette = _read_palette(
            src, mask_end, data_offset, bits, colors_used, palette_entry_size
        )
        return _palette_image(
            width,
            height,
            bits,
            indices,
            rle_palette,
            src.size,
            data_offset,
            payload_length=len(payload),
        )

    # Uncompressed / bitfields path.
    size = storage_bytes(width, height, 3 if bits > 8 else 1, 8)
    if budget is not None:
        budget.check(size, "BMP image")
    palette: Palette | None = (
        _read_palette(src, mask_end, data_offset, bits, colors_used, palette_entry_size)
        if bits <= 8
        else None
    )

    stride = ((width * bits + 31) // 32) * 4
    payload_size = checked_mul(stride, height, "BMP pixel data")
    if checked_add(data_offset, payload_size, "BMP pixel data") > src.size:
        msg = (
            f"BMP pixel data needs {payload_size} bytes at offset {data_offset}, "
            f"file has {src.size - data_offset}"
        )
        raise TruncatedDataError(msg)
    raw = src.read_exact(data_offset, payload_size, "BMP pixel data")

    row_order = range(height) if top_down else range(height - 1, -1, -1)
    if bits <= 8:
        row_length = row_bytes(width, 1, bits)
        unpacked: list[int] = []
        from .codecs import unpack_bits

        for row_index in row_order:
            if token is not None:
                token.raise_if_cancelled()
            start = row_index * stride
            unpacked.extend(unpack_bits(raw[start : start + row_length], bits, width))
        return _palette_image(
            width,
            height,
            bits,
            unpacked,
            palette,
            src.size,
            data_offset,
            payload_length=payload_size,
        )

    channels = 4 if bits == 32 and masks[3] else 3
    out = bytearray(width * height * channels)
    for out_y, row_index in enumerate(row_order):
        if token is not None:
            token.raise_if_cancelled()
        start = row_index * stride
        line = raw[start : start + width * (bits // 8)]
        target = out_y * width * channels
        if bits == 16:
            for x in range(width):
                (value,) = struct.unpack_from("<H", line, x * 2)
                red = _extract_masked(value, masks[0])
                green = _extract_masked(value, masks[1])
                blue = _extract_masked(value, masks[2])
                position = target + x * channels
                out[position] = red
                out[position + 1] = green
                out[position + 2] = blue
                if channels == 4:
                    out[position + 3] = 255
        elif bits == 24:
            for x in range(width):
                position = target + x * 3
                blue, green, red = line[x * 3 : x * 3 + 3]
                out[position] = red
                out[position + 1] = green
                out[position + 2] = blue
        else:  # 32
            for x in range(width):
                (value,) = struct.unpack_from("<I", line, x * 4)
                red = _extract_masked(value, masks[0])
                green = _extract_masked(value, masks[1])
                blue = _extract_masked(value, masks[2])
                alpha = _extract_masked(value, masks[3]) if masks[3] else 255
                position = target + x * channels
                out[position] = red
                out[position + 1] = green
                out[position + 2] = blue
                if channels == 4:
                    out[position + 3] = alpha
    mode = "RGBA" if channels == 4 else "RGB"
    return ImageData(
        format="BMP",
        mode=mode,
        width=width,
        height=height,
        bit_depth=8,
        storage_bits=8,
        channels=channels,
        samples=bytes(out),
        row_stride=row_bytes(width, channels, 8),
        info={"bits_per_pixel": bits, "compression": compression, "top_down": top_down},
    )


def _read_palette(
    src: Source,
    start: int,
    data_offset: int,
    bits: int,
    colors_used: int,
    entry_size: int,
) -> Palette:
    maximum = 1 << bits if bits <= 8 else 0
    entries = colors_used or maximum
    if entries <= 0:
        msg = "BMP palette has no entries"
        raise IllegalPaletteError(msg)
    if entries > 256:
        msg = f"BMP palette has {entries} entries, maximum is 256"
        raise IllegalPaletteError(msg)
    length = entries * entry_size
    if checked_add(start, length, "BMP palette") > data_offset:
        msg = "BMP palette overlaps the pixel data"
        raise ChunkLengthError(msg)
    data = src.read_exact(start, length, "BMP palette")
    entries_list = []
    for index in range(entries):
        base = index * entry_size
        blue, green, red = data[base : base + 3]
        entries_list.append((red, green, blue, 255))
    return Palette(tuple(entries_list))


def _palette_image(
    width: int,
    height: int,
    bits: int,
    indices: list[int],
    palette: Palette | None,
    total_size: int,
    data_offset: int,
    *,
    payload_length: int,
) -> ImageData:
    if palette is None:
        msg = "indexed BMP has no colour table"
        raise IllegalPaletteError(msg)
    max_index = max(indices) if indices else 0
    if max_index >= len(palette):
        msg = (
            f"palette index {max_index} is outside the {len(palette)} entry "
            "colour table"
        )
        raise IllegalPaletteError(msg)
    bit_depth = max(1, max_index.bit_length(), min(bits, 8) if bits <= 8 else 8)
    if bit_depth > 8:
        bit_depth = 8
    return ImageData(
        format="BMP",
        mode="P",
        width=width,
        height=height,
        bit_depth=bit_depth,
        storage_bits=8,
        channels=1,
        samples=bytes(indices),
        palette=palette,
        row_stride=row_bytes(width, 1, 8),
        info={
            "bits_per_pixel": bits,
            "data_offset": data_offset,
            "payload_length": payload_length,
            "file_size": total_size,
        },
    )


__all__ = ["BMP_MAGIC", "decode_bmp"]
