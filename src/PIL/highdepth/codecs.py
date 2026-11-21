#
# The Python Imaging Library.
#
# Sample packing, PNG filters and legacy codecs used by the decoders.
#
# See the README file for information on usage and redistribution.
#
"""Low level, deterministic sample helpers.

Every routine here is pure, takes and returns plain Python objects and never
allocates based on an untrusted value without the caller having performed the
size checks first.
"""

from __future__ import annotations

import sys
from array import array

from .errors import (
    ChunkLengthError,
    FormatError,
    TruncatedDataError,
    UnsupportedFeatureError,
)

PNG_FILTERS = (0, 1, 2, 3, 4)


def unpack_bits(
    row: bytes, bit_depth: int, count: int, *, big_endian: bool = True
) -> list[int]:
    """Unpack ``count`` MSB-first samples of ``bit_depth`` bits from ``row``.

    ``big_endian`` controls only 16-bit word order; packed 1/2/4-bit samples
    are always MSB-first as PNG requires.
    """
    if bit_depth == 8:
        if len(row) < count:
            msg = f"row has {len(row)} bytes, need {count}"
            raise TruncatedDataError(msg)
        return list(row[:count])
    if bit_depth == 16:
        if len(row) < count * 2:
            msg = f"row has {len(row)} bytes, need {count * 2}"
            raise TruncatedDataError(msg)
        values = array("H")
        values.frombytes(row[: count * 2])
        host_big = sys.byteorder == "big"
        if big_endian != host_big:
            values.byteswap()
            # ``byteswap`` changes what each element means; using it directly
            # is correct, so nothing else to do.
        return list(values)
    if bit_depth not in (1, 2, 4):
        msg = f"unsupported packed bit depth {bit_depth}"
        raise UnsupportedFeatureError(msg)
    out: list[int] = []
    mask = (1 << bit_depth) - 1
    per_byte = 8 // bit_depth
    needed = (count + per_byte - 1) // per_byte
    if len(row) < needed:
        msg = f"packed row has {len(row)} bytes, need {needed}"
        raise TruncatedDataError(msg)
    for index in range(needed):
        byte = row[index]
        for slot in range(per_byte):
            if len(out) == count:
                return out
            shift = 8 - bit_depth * (slot + 1)
            out.append((byte >> shift) & mask)
    return out


def pack_bits(
    samples: list[int] | tuple[int, ...], bit_depth: int, *, big_endian: bool = True
) -> bytes:
    """Pack samples MSB-first, padding the final byte with zero bits."""
    if bit_depth == 8:
        return bytes(samples)
    if bit_depth == 16:
        out = bytearray()
        for value in samples:
            if big_endian:
                out.append((value >> 8) & 0xFF)
                out.append(value & 0xFF)
            else:
                out.append(value & 0xFF)
                out.append((value >> 8) & 0xFF)
        return bytes(out)
    if bit_depth not in (1, 2, 4):
        msg = f"unsupported packed bit depth {bit_depth}"
        raise UnsupportedFeatureError(msg)
    mask = (1 << bit_depth) - 1
    per_byte = 8 // bit_depth
    out = bytearray()
    byte = 0
    used = 0
    for value in samples:
        if not 0 <= value <= mask:
            msg = f"sample {value} does not fit in {bit_depth} bits"
            raise FormatError(msg)
        byte = (byte << bit_depth) | value
        used += 1
        if used == per_byte:
            out.append(byte)
            byte = 0
            used = 0
    if used:
        out.append(byte << (bit_depth * (per_byte - used)))
    return bytes(out)


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def unfilter_row(
    filter_type: int, row: bytearray, previous: bytes | None, bpp: int
) -> bytes:
    """Reverse one PNG filter in place and return the unfiltered row."""
    if filter_type == 0:
        return bytes(row)
    if filter_type not in PNG_FILTERS:
        msg = f"unknown PNG filter type {filter_type}"
        raise FormatError(msg)
    length = len(row)
    prev = previous if previous is not None else bytes(length)
    if len(prev) != length:
        msg = "previous row length mismatch"
        raise FormatError(msg)
    if filter_type == 1:  # Sub
        for i in range(bpp, length):
            row[i] = (row[i] + row[i - bpp]) & 0xFF
    elif filter_type == 2:  # Up
        for i in range(length):
            row[i] = (row[i] + prev[i]) & 0xFF
    elif filter_type == 3:  # Average
        for i in range(length):
            left = row[i - bpp] if i >= bpp else 0
            row[i] = (row[i] + ((left + prev[i]) >> 1)) & 0xFF
    else:  # Paeth
        for i in range(length):
            left = row[i - bpp] if i >= bpp else 0
            upper_left = prev[i - bpp] if i >= bpp else 0
            row[i] = (row[i] + paeth(left, prev[i], upper_left)) & 0xFF
    return bytes(row)


def filter_row(filter_type: int, row: bytes, previous: bytes | None, bpp: int) -> bytes:
    """Apply one PNG filter, mirroring :func:`unfilter_row` exactly."""
    if filter_type not in PNG_FILTERS:
        msg = f"unknown PNG filter type {filter_type}"
        raise FormatError(msg)
    length = len(row)
    prev = previous if previous is not None else bytes(length)
    out = bytearray(length)
    if filter_type == 0:
        out[:] = row
    elif filter_type == 1:
        for i in range(length):
            left = row[i - bpp] if i >= bpp else 0
            out[i] = (row[i] - left) & 0xFF
    elif filter_type == 2:
        for i in range(length):
            out[i] = (row[i] - prev[i]) & 0xFF
    elif filter_type == 3:
        for i in range(length):
            left = row[i - bpp] if i >= bpp else 0
            out[i] = (row[i] - ((left + prev[i]) >> 1)) & 0xFF
    else:
        for i in range(length):
            left = row[i - bpp] if i >= bpp else 0
            upper_left = prev[i - bpp] if i >= bpp else 0
            out[i] = (row[i] - paeth(left, prev[i], upper_left)) & 0xFF
    return bytes(out)


def lzw_decode(data: bytes, expected: int, early_change: int = 1) -> bytes:
    """Decode TIFF LZW data.

    ``early_change`` of 1 matches libtiff's default and is what Pillow writes.
    Code lengths widen one code early, which is the classic LZW subtlety.
    """
    if expected < 0:
        msg = "negative expected output size"
        raise FormatError(msg)
    out = bytearray()
    dictionary: list[bytes] = [bytes((i,)) for i in range(256)] + [b"", b""]
    code_width = 9
    bit_buffer = 0
    bit_count = 0
    position = 0

    def next_code(width: int) -> int:
        nonlocal bit_buffer, bit_count, position
        while bit_count < width:
            if position >= len(data):
                return -1
            bit_buffer = (bit_buffer << 8) | data[position]
            position += 1
            bit_count += 8
        bit_count -= width
        return (bit_buffer >> bit_count) & ((1 << width) - 1)

    previous: bytes | None = None
    while True:
        code = next_code(code_width)
        if code < 0:
            break
        if code == 256:  # clear
            dictionary = [bytes((i,)) for i in range(256)] + [b"", b""]
            code_width = 9
            previous = None
            continue
        if code == 257:  # end of information
            break
        if code < len(dictionary):
            entry = dictionary[code]
        elif code == len(dictionary) and previous is not None:
            entry = previous + previous[:1]
        else:
            msg = f"invalid LZW code {code} at dictionary size {len(dictionary)}"
            raise FormatError(msg)
        out += entry
        if previous is not None:
            dictionary.append(previous + entry[:1])
            if len(dictionary) + early_change >= (1 << code_width) and code_width < 12:
                code_width += 1
        previous = entry
        if len(out) > expected:
            msg = f"LZW stream decoded to more than {expected} bytes"
            raise ChunkLengthError(msg)
    if len(out) != expected:
        msg = f"LZW stream decoded to {len(out)} bytes, expected {expected}"
        raise TruncatedDataError(msg)
    return bytes(out)


def packbits_decode(data: bytes, expected: int) -> bytes:
    out = bytearray()
    position = 0
    length = len(data)
    while position < length:
        header = data[position]
        position += 1
        if header < 128:
            count = header + 1
            chunk = data[position : position + count]
            if len(chunk) != count:
                msg = "truncated PackBits literal run"
                raise TruncatedDataError(msg)
            out += chunk
            position += count
        elif header > 128:
            if position >= length:
                msg = "truncated PackBits repeat run"
                raise TruncatedDataError(msg)
            out += bytes((data[position],)) * (257 - header)
            position += 1
        # header == 128 is a no-op
        if len(out) > expected:
            msg = f"PackBits stream decoded to more than {expected} bytes"
            raise ChunkLengthError(msg)
    if len(out) != expected:
        msg = f"PackBits stream decoded to {len(out)} bytes, expected {expected}"
        raise TruncatedDataError(msg)
    return bytes(out)


def deflate_decode(data: bytes, expected: int) -> bytes:
    import zlib

    try:
        raw = zlib.decompress(data)
    except zlib.error as exc:
        msg = f"invalid deflate stream: {exc}"
        raise FormatError(msg) from exc
    if len(raw) != expected:
        if len(raw) < expected:
            msg = f"deflate stream decoded to {len(raw)} bytes, expected {expected}"
            raise TruncatedDataError(msg)
        msg = f"deflate stream decoded to more than {expected} bytes"
        raise ChunkLengthError(msg)
    return raw


def apply_predictor(
    raw: bytes,
    rows: int,
    row_bytes: int,
    bits_per_sample: int,
    samples_per_pixel: int,
    predictor: int,
    *,
    big_endian: bool = True,
) -> bytes:
    """Reverse a TIFF horizontal predictor (predictor 2)."""
    if predictor in (0, 1):
        return raw
    if predictor != 2:
        msg = f"unsupported TIFF predictor {predictor}"
        raise UnsupportedFeatureError(msg)
    if bits_per_sample not in (8, 16):
        msg = "horizontal predictor requires 8 or 16 bits per sample"
        raise UnsupportedFeatureError(msg)
    width = row_bytes // (bits_per_sample // 8)
    stride = samples_per_pixel
    out = bytearray(raw)
    if bits_per_sample == 8:
        for r in range(rows):
            base = r * row_bytes
            for i in range(stride, width):
                out[base + i] = (out[base + i] + out[base + i - stride]) & 0xFF
    else:
        values = array("H")
        values.frombytes(raw)
        host_big = sys.byteorder == "big"
        if big_endian != host_big:
            values.byteswap()
        for r in range(rows):
            base = r * width
            for i in range(stride, width):
                values[base + i] = (
                    values[base + i] + values[base + i - stride]
                ) & 0xFFFF
        if big_endian != host_big:
            values.byteswap()
        out = bytearray(values.tobytes())
    return bytes(out)


def expand_to_8bit(value: int, bit_depth: int) -> int:
    """Scale an integer sample to full 8-bit range with half-up rounding."""
    if bit_depth == 8:
        return value
    maximum = (1 << bit_depth) - 1
    return (value * 255 + maximum // 2) // maximum


def rotate_samples(
    samples: list[int],
    width: int,
    height: int,
    channels: int,
    orientation: int,
) -> tuple[list[int], int, int]:
    """Map an EXIF orientation to top-left origin and return the new size."""
    if orientation == 1:
        return samples, width, height

    def at(x: int, y: int) -> list[int]:
        start = (y * width + x) * channels
        return samples[start : start + channels]

    out: list[int] = []
    if orientation == 2:  # mirror horizontal
        for y in range(height):
            for x in range(width - 1, -1, -1):
                out.extend(at(x, y))
        return out, width, height
    if orientation == 3:  # rotate 180
        for y in range(height - 1, -1, -1):
            for x in range(width - 1, -1, -1):
                out.extend(at(x, y))
        return out, width, height
    if orientation == 4:  # mirror vertical
        for y in range(height - 1, -1, -1):
            for x in range(width):
                out.extend(at(x, y))
        return out, width, height
    if orientation == 5:  # transpose
        for x in range(width):
            for y in range(height):
                out.extend(at(x, y))
        return out, height, width
    if orientation == 6:  # rotate 90 clockwise
        for x in range(width):
            for y in range(height - 1, -1, -1):
                out.extend(at(x, y))
        return out, height, width
    if orientation == 7:  # transverse
        for x in range(width - 1, -1, -1):
            for y in range(height - 1, -1, -1):
                out.extend(at(x, y))
        return out, height, width
    if orientation == 8:  # rotate 270 clockwise
        for x in range(width - 1, -1, -1):
            for y in range(height):
                out.extend(at(x, y))
        return out, height, width
    msg = f"invalid orientation {orientation}"
    raise FormatError(msg)


__all__ = [
    "PNG_FILTERS",
    "apply_predictor",
    "deflate_decode",
    "expand_to_8bit",
    "filter_row",
    "lzw_decode",
    "pack_bits",
    "packbits_decode",
    "paeth",
    "rotate_samples",
    "unfilter_row",
    "unpack_bits",
]
