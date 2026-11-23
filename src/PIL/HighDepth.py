#
# The Python Imaging Library.
#
# High depth decoding helpers.
#
# This module adds a container-validating, sample-accurate decoding layer on
# top of the regular image plugins.  PNG, TIFF and BMP data can be read from a
# bytes object, a file path, an open binary file object or an iterable of byte
# chunks; all input methods return the same pixels and metadata.
#
# The container is validated and the uncompressed size is checked against a
# memory budget *before* a decoded buffer is allocated.  Truncated rows,
# contradictory chunk lengths, invalid colour tables, size overflows and
# illegal sample layouts raise a stable :exc:`DecodeError`.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

__lazy_modules__ = {"PIL._binary", "io", "os", "struct", "sys", "zlib"}

import io
import os
import struct
import sys
import zlib
from collections.abc import Iterable, Iterator
from typing import NamedTuple, Self, cast

from . import Image
from ._binary import i16le, i32le
from ._binary import i32be as i32
from ._util import is_path

TYPE_CHECKING = False

__all__ = [
    "DecodeError",
    "DecodedImage",
    "ImageInfo",
    "MAX_DECODED_BYTES",
    "MemoryBudget",
    "MemoryBudgetError",
    "PNG_PASSES",
    "StreamReader",
    "decode",
    "decode_planes",
    "probe",
    "read",
    "rescale_plane",
    "sample_to_unit",
    "unit_to_sample",
]


MAX_DECODED_BYTES = 512 * 1024 * 1024
"""Default upper bound, in bytes, for a decoded image plane."""

_PNG_MAGIC = b"\211PNG\r\n\032\n"

# Adam7 pass geometry: (x offset, y offset, x step, y step)
PNG_PASSES: tuple[tuple[int, int, int, int], ...] = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)

# Valid (bit depth, colour type) combinations, from the PNG specification.
_PNG_COMBINATIONS = {
    (1, 0),
    (2, 0),
    (4, 0),
    (8, 0),
    (16, 0),
    (8, 2),
    (16, 2),
    (1, 3),
    (2, 3),
    (4, 3),
    (8, 3),
    (8, 4),
    (16, 4),
    (8, 6),
    (16, 6),
}

_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}

_BMP_HEADER_SIZES = {12, 40, 52, 56, 64, 108, 124}
_BMP_BIT_DEPTHS = {1, 4, 8, 16, 24, 32}
_BMP_COMPRESSIONS = {0, 1, 2, 3, 4, 5}


class DecodeError(OSError):
    """Raised when a high depth image source is structurally invalid.

    Subclasses :exc:`OSError` to match the errors raised by the regular file
    plugins.
    """


class MemoryBudgetError(MemoryError):
    """Raised before allocating a decoded buffer larger than the budget."""


class MemoryBudget:
    """A decoded-size budget, checked before pixels are allocated.

    :param max_bytes: Maximum size of a decoded image plane, in bytes.
        Defaults to :data:`MAX_DECODED_BYTES`.
    """

    def __init__(self, max_bytes: int | None = None) -> None:
        if max_bytes is None:
            max_bytes = MAX_DECODED_BYTES
        if max_bytes <= 0:
            msg = "max_bytes must be positive"
            raise ValueError(msg)
        self.max_bytes = max_bytes

    def size(
        self,
        size: tuple[int, int],
        *,
        channels: int = 1,
        bytes_per_sample: int = 1,
    ) -> int:
        """Return the decoded size in bytes for an image of ``size``."""

        width, height = size
        if width < 0 or height < 0:
            msg = f"invalid image size {size!r}"
            raise DecodeError(msg)
        if channels < 1 or bytes_per_sample < 1:
            msg = "channels and bytes_per_sample must be positive"
            raise ValueError(msg)
        if width > (2**31 - 1) // bytes_per_sample // channels:
            msg = f"image width {width} overflows the pixel buffer"
            raise DecodeError(msg)
        total = width * height * channels * bytes_per_sample
        if total > 2**31 - 1:
            msg = f"decoded size {total} bytes overflows the pixel buffer"
            raise DecodeError(msg)
        return total

    def check(
        self,
        size: tuple[int, int],
        *,
        channels: int = 1,
        bytes_per_sample: int = 1,
    ) -> None:
        """Check that an image of ``size`` fits in the budget.

        :exception MemoryBudgetError: The decoded image is too large.
        """

        total = self.size(size, channels=channels, bytes_per_sample=bytes_per_sample)
        if total > self.max_bytes:
            msg = (
                f"decoded image needs {total} bytes, "
                f"exceeding memory budget of {self.max_bytes} bytes"
            )
            raise MemoryBudgetError(msg)


class ImageInfo(NamedTuple):
    """Metadata describing a decoded high depth image."""

    format: str
    width: int
    height: int
    mode: str
    bit_depth: int
    channels: int
    palette: bytes | None
    palette_size: int
    transparency: bytes | None
    interlace: bool
    file_row_order: str
    frame_count: int
    frame_index: int
    icc_profile: bytes | None = None

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def has_alpha(self) -> bool:
        return self.channels in (2, 4)


class DecodedImage:
    """A decoded image plus full precision sample planes.

    ``image`` is the regular Pillow image, identical to the result of
    :py:func:`PIL.Image.open` for the same bytes.  ``planes`` holds one plane per
    channel.  Planes store 16-bit samples big-endian and 8-bit samples one
    byte each; sub-8-bit samples keep their raw value in one byte per sample.
    """

    __slots__ = ("image", "info", "planes")

    def __init__(
        self, image: Image.Image, info: ImageInfo, planes: tuple[bytes, ...]
    ) -> None:
        self.image = image
        self.info = info
        self.planes = planes

    def __repr__(self) -> str:
        return (
            f"<DecodedImage {self.info.format} {self.info.mode} "
            f"{self.info.width}x{self.info.height} {self.info.bit_depth}-bit>"
        )

    def sample(self, x: int, y: int, channel: int = 0) -> int:
        """Return the raw sample value at a pixel.

        Sub-8-bit samples are not scaled; 16-bit samples are returned in the
        range 0..65535.
        """

        info = self.info
        if not 0 <= x < info.width or not 0 <= y < info.height:
            msg = f"pixel ({x}, {y}) is outside the image"
            raise IndexError(msg)
        if not 0 <= channel < len(self.planes):
            msg = f"channel {channel} is out of range"
            raise IndexError(msg)
        plane = self.planes[channel]
        bits = info.bit_depth
        if bits == 16:
            index = (y * info.width + x) * 2
            return (plane[index] << 8) | plane[index + 1]
        value = plane[y * info.width + x]
        return value if bits == 8 else value & ((1 << bits) - 1)

    def rescale(self, bits: int = 8) -> DecodedImage:
        """Return a copy whose samples are rescaled to ``bits`` bits."""

        if bits not in (1, 2, 4, 8, 16):
            msg = f"unsupported bit depth {bits}"
            raise ValueError(msg)
        source_bits = self.info.bit_depth
        if bits == source_bits:
            return self
        planes = tuple(
            rescale_plane(plane, source_bits, bits, self.info.width, self.info.height)
            for plane in self.planes
        )
        return DecodedImage(self.image, self.info._replace(bit_depth=bits), planes)


def _check_bytes(chunk: object) -> bytes:
    if isinstance(chunk, bytes):
        return chunk
    if isinstance(chunk, (bytearray, memoryview)):
        return bytes(chunk)
    msg = f"image chunks must be bytes, not {type(chunk).__name__}"
    raise TypeError(msg)


def _read_source(source: object) -> bytes:
    """Read a bytes object, path, file object or chunk iterator into bytes."""

    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if is_path(source):
        with open(os.fspath(source), "rb") as fp:
            return fp.read()
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            msg = "image file must be opened in binary mode"
            raise TypeError(msg)
        return _check_bytes(data)
    if isinstance(source, Iterable) and not isinstance(source, (str, dict)):
        return b"".join(_check_bytes(chunk) for chunk in source)
    msg = f"unsupported image source {type(source).__name__}"
    raise TypeError(msg)


# --------------------------------------------------------------------
# Sample scaling


def sample_to_unit(value: int, bits: int) -> float:
    """Map an integer sample of ``bits`` bits to the unit interval."""

    max_value = (1 << bits) - 1
    return max(0, min(max_value, int(value))) / max_value


def unit_to_sample(value: float, bits: int) -> int:
    """Map a unit interval value to an integer sample, rounding half up.

    NaN and infinities have deterministic results: NaN maps to 0, negative
    infinity to 0 and positive infinity to the maximum sample.
    """

    max_value = (1 << bits) - 1
    if value != value:
        return 0
    if value == float("inf"):
        return max_value
    if value == float("-inf"):
        return 0
    scaled = value * max_value
    if scaled <= 0:
        return 0
    if scaled >= max_value:
        return max_value
    return int(scaled + 0.5)


def _iter_samples(plane: bytes, bits: int, width: int, height: int) -> Iterator[int]:
    if bits == 8:
        return iter(plane[: width * height])
    if bits == 16:
        return ((plane[i] << 8) | plane[i + 1] for i in range(0, width * height * 2, 2))
    # Sub-8-bit planes store one raw sample per byte.
    mask = (1 << bits) - 1
    return (value & mask for value in plane[: width * height])


def rescale_plane(
    plane: bytes, from_bits: int, to_bits: int, width: int, height: int
) -> bytes:
    """Rescale samples in ``plane`` from ``from_bits`` to ``to_bits``.

    Planes store 8-bit and 16-bit samples densely (16-bit big-endian); samples
    narrower than 8 bits use one byte each.
    """

    if from_bits == to_bits:
        return bytes(plane)
    if from_bits not in (1, 2, 4, 8, 16) or to_bits not in (1, 2, 4, 8, 16):
        msg = f"cannot rescale {from_bits}-bit samples to {to_bits} bits"
        raise ValueError(msg)
    source_max = (1 << from_bits) - 1
    target_max = (1 << to_bits) - 1
    scaled = [
        (value * target_max + source_max // 2) // source_max
        for value in _iter_samples(plane, from_bits, width, height)
    ]
    if to_bits in (1, 2, 4, 8):
        return bytes(scaled)
    values = bytearray(width * height * 2)
    for index, value in enumerate(scaled):
        values[index * 2] = value >> 8
        values[index * 2 + 1] = value & 0xFF
    return bytes(values)


# --------------------------------------------------------------------
# PNG


class _PNGChunk(NamedTuple):
    cid: bytes
    data: bytes


def _png_chunks(data: bytes) -> list[_PNGChunk]:
    if not data.startswith(_PNG_MAGIC):
        msg = "not a PNG file"
        raise DecodeError(msg)
    chunks: list[_PNGChunk] = []
    pos = len(_PNG_MAGIC)
    size = len(data)
    while pos < size:
        if pos + 8 > size:
            msg = "truncated PNG chunk header"
            raise DecodeError(msg)
        length = i32(data, pos)
        cid = data[pos + 4 : pos + 8]
        if len(cid) != 4 or not all(65 <= c <= 90 or 97 <= c <= 122 for c in cid):
            msg = f"broken PNG file (chunk {cid!r})"
            raise DecodeError(msg)
        data_start = pos + 8
        data_end = data_start + length
        if data_end + 4 > size:
            msg = (
                f"PNG chunk {cid.decode('latin-1')} declares {length} bytes "
                "but the file is truncated"
            )
            raise DecodeError(msg)
        payload = data[data_start:data_end]
        expected_crc = i32(data, data_end)
        crc = zlib.crc32(payload, zlib.crc32(cid)) & 0xFFFFFFFF
        if crc != expected_crc:
            msg = f"broken PNG file (bad header checksum in {cid!r})"
            raise DecodeError(msg)
        chunks.append(_PNGChunk(cid, payload))
        pos = data_end + 4
        if cid == b"IEND":
            if pos != size:
                msg = f"PNG file has {size - pos} trailing bytes after IEND"
                raise DecodeError(msg)
            break
    if not chunks or chunks[0].cid != b"IHDR":
        msg = "PNG file does not start with IHDR"
        raise DecodeError(msg)
    if chunks[-1].cid != b"IEND":
        msg = "PNG file has no IEND chunk"
        raise DecodeError(msg)
    return chunks


class _PNGHeader(NamedTuple):
    width: int
    height: int
    bit_depth: int
    color_type: int
    interlace: bool

    @property
    def channels(self) -> int:
        return _PNG_CHANNELS[self.color_type]


def _png_header(chunks: list[_PNGChunk]) -> _PNGHeader:
    chunk = chunks[0]
    if len(chunk.data) != 13:
        msg = "PNG IHDR chunk must be 13 bytes"
        raise DecodeError(msg)
    width = i32(chunk.data, 0)
    height = i32(chunk.data, 4)
    bit_depth = chunk.data[8]
    color_type = chunk.data[9]
    compression = chunk.data[10]
    filter_method = chunk.data[11]
    interlace = chunk.data[12]
    if width <= 0 or height <= 0:
        msg = f"invalid PNG image size {width}x{height}"
        raise DecodeError(msg)
    if (bit_depth, color_type) not in _PNG_COMBINATIONS:
        msg = f"invalid PNG bit depth {bit_depth} for colour type {color_type}"
        raise DecodeError(msg)
    if compression != 0:
        msg = f"unknown PNG compression method {compression}"
        raise DecodeError(msg)
    if filter_method != 0:
        msg = f"unknown PNG filter method {filter_method}"
        raise DecodeError(msg)
    if interlace not in (0, 1):
        msg = f"unknown PNG interlace method {interlace}"
        raise DecodeError(msg)
    return _PNGHeader(width, height, bit_depth, color_type, interlace == 1)


def _png_mode(header: _PNGHeader) -> str:
    bit_depth, color_type = header.bit_depth, header.color_type
    if color_type == 0:
        if bit_depth == 16:
            return "I;16"
        return "1" if bit_depth == 1 else "L"
    if color_type == 3:
        return "P"
    if color_type == 2:
        return "RGB"
    if color_type == 4:
        return "LA" if bit_depth == 8 else "RGBA"
    return "RGBA"


def _png_row_bytes(width: int, bit_depth: int, channels: int) -> int:
    return (width * bit_depth * channels + 7) // 8


def _png_expected_size(header: _PNGHeader) -> int:
    channels = header.channels
    if not header.interlace:
        return header.height * (
            1 + _png_row_bytes(header.width, header.bit_depth, channels)
        )
    total = 0
    for x0, y0, dx, dy in PNG_PASSES:
        pass_width = (header.width - x0 + dx - 1) // dx if header.width > x0 else 0
        pass_height = (header.height - y0 + dy - 1) // dy if header.height > y0 else 0
        if pass_width and pass_height:
            total += pass_height * (
                1 + _png_row_bytes(pass_width, header.bit_depth, channels)
            )
    return total


class _PNGStructure(NamedTuple):
    header: _PNGHeader
    palette: bytes | None
    palette_size: int
    transparency: bytes | None
    icc_profile: bytes | None
    idat: bytes


def _png_structure(chunks: list[_PNGChunk]) -> _PNGStructure:
    header = _png_header(chunks)
    palette: bytes | None = None
    palette_size = 0
    transparency: bytes | None = None
    icc: bytes | None = None
    idat: list[bytes] = []
    seen_idat = False
    idat_closed = False
    for chunk in chunks[1:]:
        cid = chunk.cid
        if cid == b"IHDR":
            msg = "PNG file has more than one IHDR chunk"
            raise DecodeError(msg)
        if cid == b"PLTE":
            if palette is not None:
                msg = "PNG file has more than one PLTE chunk"
                raise DecodeError(msg)
            if header.color_type in (0, 4):
                msg = "PLTE chunk is not allowed for greyscale PNG images"
                raise DecodeError(msg)
            if len(chunk.data) == 0 or len(chunk.data) % 3:
                msg = f"invalid palette size {len(chunk.data)}"
                raise DecodeError(msg)
            palette_size = len(chunk.data) // 3
            max_entries = 256 if header.color_type != 3 else 1 << header.bit_depth
            if palette_size > max_entries:
                msg = f"invalid palette size {palette_size}"
                raise DecodeError(msg)
            palette = chunk.data
        elif cid == b"tRNS":
            if header.color_type in (4, 6):
                msg = "tRNS chunk is not allowed for images with alpha"
                raise DecodeError(msg)
            if transparency is not None:
                msg = "PNG file has more than one tRNS chunk"
                raise DecodeError(msg)
            if header.color_type == 3:
                if palette is None:
                    msg = "tRNS chunk appears before PLTE"
                    raise DecodeError(msg)
                if len(chunk.data) > palette_size:
                    msg = (
                        f"tRNS chunk has {len(chunk.data)} entries, "
                        f"but the palette has {palette_size}"
                    )
                    raise DecodeError(msg)
            else:
                required = 2 if header.color_type == 0 else 6
                if len(chunk.data) != required:
                    msg = (
                        f"tRNS chunk must be {required} bytes, "
                        f"got {len(chunk.data)}"
                    )
                    raise DecodeError(msg)
            transparency = chunk.data
        elif cid == b"IDAT":
            if idat_closed:
                msg = "PNG IDAT chunks must be consecutive"
                raise DecodeError(msg)
            seen_idat = True
            idat.append(chunk.data)
        else:
            if seen_idat:
                idat_closed = True
            if cid == b"acTL":
                msg = "animated PNG is not supported by high depth decoding"
                raise DecodeError(msg)
            if cid == b"iCCP":
                index = chunk.data.find(b"\0")
                if index < 0 or index + 2 > len(chunk.data):
                    msg = "invalid iCCP chunk"
                    raise DecodeError(msg)
                if chunk.data[index + 1] != 0:
                    msg = "unknown iCCP compression method"
                    raise DecodeError(msg)
                try:
                    icc = zlib.decompress(chunk.data[index + 2 :])
                except zlib.error as e:
                    msg = "invalid iCCP profile data"
                    raise DecodeError(msg) from e
            elif cid[:1].isupper() and cid not in {
                b"IEND",
            }:
                msg = f"unknown critical PNG chunk {cid!r}"
                raise DecodeError(msg)
    if header.color_type == 3 and palette is None:
        msg = "indexed PNG file has no PLTE chunk"
        raise DecodeError(msg)
    if not seen_idat:
        msg = "PNG file has no IDAT chunk"
        raise DecodeError(msg)
    return _PNGStructure(
        header, palette, palette_size, transparency, icc, b"".join(idat)
    )


def _png_planes(data: bytes, structure: _PNGStructure) -> tuple[bytes, ...]:
    header = structure.header
    expected = _png_expected_size(header)
    try:
        raw = zlib.decompress(structure.idat)
    except zlib.error as e:
        msg = "broken PNG compressed data"
        raise DecodeError(msg) from e
    if len(raw) < expected:
        msg = f"truncated PNG image data: expected {expected} bytes, got {len(raw)}"
        raise DecodeError(msg)
    if len(raw) > expected:
        msg = f"PNG image data has {len(raw) - expected} extra bytes"
        raise DecodeError(msg)

    channels = header.channels
    bit_depth = header.bit_depth
    bpp = max(1, (bit_depth * channels + 7) // 8)
    rows: list[bytearray] = [
        bytearray(_png_row_bytes(header.width, bit_depth, channels))
        for _ in range(header.height)
    ]
    position = 0

    def decode_pass(
        width: int, height: int, x0: int, y0: int, dx: int, dy: int
    ) -> None:
        nonlocal position
        row_bytes = _png_row_bytes(width, bit_depth, channels)
        previous = bytes(row_bytes)
        current = bytearray(row_bytes)
        for row_index in range(height):
            if position + 1 + row_bytes > len(raw):
                msg = "truncated PNG image data"
                raise DecodeError(msg)
            filter_type = raw[position]
            row = raw[position + 1 : position + 1 + row_bytes]
            position += 1 + row_bytes
            if filter_type == 0:
                current[:] = row
            elif filter_type == 1:
                left = bytearray(row_bytes)
                for i in range(row_bytes):
                    left[i] = (row[i] + (left[i - bpp] if i >= bpp else 0)) & 0xFF
                current[:] = left
            elif filter_type == 2:
                for i in range(row_bytes):
                    current[i] = (row[i] + previous[i]) & 0xFF
            elif filter_type == 3:
                for i in range(row_bytes):
                    left_value = current[i - bpp] if i >= bpp else 0
                    current[i] = (row[i] + ((left_value + previous[i]) >> 1)) & 0xFF
            elif filter_type == 4:
                for i in range(row_bytes):
                    if i >= bpp:
                        left_value = current[i - bpp]
                        upper_left = previous[i - bpp]
                    else:
                        left_value = upper_left = 0
                    upper = previous[i]
                    estimate = left_value + upper - upper_left
                    pa = abs(estimate - left_value)
                    pb = abs(estimate - upper)
                    pc = abs(estimate - upper_left)
                    if pa <= pb and pa <= pc:
                        predictor = left_value
                    elif pb <= pc:
                        predictor = upper
                    else:
                        predictor = upper_left
                    current[i] = (row[i] + predictor) & 0xFF
            else:
                msg = f"unknown PNG filter type {filter_type}"
                raise DecodeError(msg)
            previous = bytes(current)
            y = y0 + row_index * dy
            if dx == 1 and x0 == 0:
                rows[y][:] = current
            else:
                _merge_pass_row(rows[y], current, width, bit_depth * channels, x0, dx)

    if not header.interlace:
        decode_pass(header.width, header.height, 0, 0, 1, 1)
    else:
        for x0, y0, dx, dy in PNG_PASSES:
            pass_width = (header.width - x0 + dx - 1) // dx if header.width > x0 else 0
            pass_height = (
                (header.height - y0 + dy - 1) // dy if header.height > y0 else 0
            )
            if pass_width and pass_height:
                decode_pass(pass_width, pass_height, x0, y0, dx, dy)

    return _split_planes(rows, header)


def _merge_pass_row(
    target: bytearray,
    row: bytearray,
    pass_width: int,
    bits_per_pixel: int,
    x0: int,
    dx: int,
) -> None:
    if bits_per_pixel % 8 == 0:
        pixel_bytes = bits_per_pixel // 8
        for index in range(pass_width):
            x = x0 + index * dx
            target_start = x * pixel_bytes
            source_start = index * pixel_bytes
            target[target_start : target_start + pixel_bytes] = row[
                source_start : source_start + pixel_bytes
            ]
        return
    mask = (1 << bits_per_pixel) - 1
    for index in range(pass_width):
        x = x0 + index * dx
        source_byte = row[(index * bits_per_pixel) // 8]
        shift = 8 - bits_per_pixel - (index * bits_per_pixel) % 8
        value = (source_byte >> shift) & mask
        target_byte = x * bits_per_pixel // 8
        target_shift = 8 - bits_per_pixel - (x * bits_per_pixel) % 8
        target[target_byte] = (target[target_byte] & ~(mask << target_shift)) | (
            value << target_shift
        )


def _split_planes(rows: list[bytearray], header: _PNGHeader) -> tuple[bytes, ...]:
    bit_depth = header.bit_depth
    channels = header.channels
    width = header.width
    if bit_depth < 8:
        # Unpack the packed samples; one raw sample per byte.
        mask = (1 << bit_depth) - 1
        out = bytearray(width * len(rows))
        for y, row in enumerate(rows):
            for x in range(width):
                value = row[(x * bit_depth) // 8]
                shift = 8 - bit_depth - (x * bit_depth) % 8
                out[y * width + x] = (value >> shift) & mask
        return (bytes(out),)
    if channels == 1:
        return (b"".join(bytes(row) for row in rows),)
    sample_bytes = 2 if bit_depth == 16 else 1
    planes = [bytearray(width * len(rows) * sample_bytes) for _ in range(channels)]
    for y, row in enumerate(rows):
        for x in range(width):
            base = x * channels * sample_bytes
            index = (y * width + x) * sample_bytes
            for channel in range(channels):
                planes[channel][index : index + sample_bytes] = row[
                    base + channel * sample_bytes : base + (channel + 1) * sample_bytes
                ]
    return tuple(bytes(plane) for plane in planes)


# --------------------------------------------------------------------
# Format detection


def _identify(data: bytes) -> str:
    if data.startswith(_PNG_MAGIC):
        return "PNG"
    if data.startswith(b"BM"):
        return "BMP"
    if data[:2] in (b"II", b"MM"):
        return "TIFF"
    msg = "cannot identify the high depth image format"
    raise DecodeError(msg)


def _png_info(structure: _PNGStructure) -> ImageInfo:
    header = structure.header
    return ImageInfo(
        format="PNG",
        width=header.width,
        height=header.height,
        mode=_png_mode(header),
        bit_depth=header.bit_depth,
        channels=header.channels,
        palette=structure.palette,
        palette_size=structure.palette_size,
        transparency=structure.transparency,
        interlace=header.interlace,
        file_row_order="top-down",
        frame_count=1,
        frame_index=0,
        icc_profile=structure.icc_profile,
    )


# --------------------------------------------------------------------
# BMP


class _BMPHeader(NamedTuple):
    header_size: int
    width: int
    height: int
    bit_depth: int
    compression: int
    palette_size: int
    data_offset: int
    row_order: str


def _bmp_header(data: bytes) -> _BMPHeader:
    if len(data) < 14 or not data.startswith(b"BM"):
        msg = "not a BMP file"
        raise DecodeError(msg)
    file_size = i32le(data, 2)
    if file_size and file_size > len(data):
        msg = f"BMP file declares {file_size} bytes but only {len(data)} are present"
        raise DecodeError(msg)
    data_offset = i32le(data, 10)
    if data_offset < 18 or data_offset > len(data):
        msg = f"invalid BMP pixel data offset {data_offset}"
        raise DecodeError(msg)
    if len(data) < 18:
        msg = "truncated BMP header"
        raise DecodeError(msg)
    header_size = i32le(data, 14)
    if header_size not in _BMP_HEADER_SIZES:
        msg = f"unsupported BMP header type ({header_size})"
        raise DecodeError(msg)
    if len(data) < 14 + header_size:
        msg = "truncated BMP header"
        raise DecodeError(msg)
    if header_size == 12:
        width = i16le(data, 18)
        height = i16le(data, 20)
        planes = i16le(data, 22)
        bits = i16le(data, 24)
        compression = 0
        colors_used = 0
        y_flip = False
        palette_padding = 3
    else:
        width = i32le(data, 18)
        height_raw = i32le(data, 22)
        if height_raw >= 0x80000000:
            # The height field is a signed 32-bit integer; a negative value
            # marks a top-down bitmap.
            height_raw -= 0x100000000
        y_flip = height_raw < 0
        height = -height_raw if y_flip else height_raw
        planes = i16le(data, 26)
        bits = i16le(data, 28)
        compression = i32le(data, 30)
        colors_used = i32le(data, 46) if header_size >= 40 else 0
        palette_padding = 4
    if width <= 0 or height <= 0:
        msg = f"invalid BMP image size {width}x{height}"
        raise DecodeError(msg)
    if planes != 1:
        msg = f"invalid BMP plane count {planes}"
        raise DecodeError(msg)
    if bits not in _BMP_BIT_DEPTHS:
        msg = f"unsupported BMP pixel depth ({bits})"
        raise DecodeError(msg)
    if compression not in _BMP_COMPRESSIONS:
        msg = f"unsupported BMP compression ({compression})"
        raise DecodeError(msg)
    if compression == 1 and bits != 8:
        msg = "BMP RLE8 compression requires 8-bit pixels"
        raise DecodeError(msg)
    if compression == 2 and bits != 4:
        msg = "BMP RLE4 compression requires 4-bit pixels"
        raise DecodeError(msg)
    if compression == 3 and bits not in (16, 32):
        msg = f"BMP bitfields compression is not valid for {bits}-bit pixels"
        raise DecodeError(msg)
    palette_size = 0
    if bits <= 8:
        palette_size = colors_used or (1 << bits)
        if not 0 < palette_size <= 256:
            msg = f"unsupported BMP palette size ({palette_size})"
            raise DecodeError(msg)
        if palette_size > (1 << bits):
            msg = f"BMP palette has {palette_size} entries for {bits}-bit pixels"
            raise DecodeError(msg)
        palette_end = 14 + header_size + palette_size * palette_padding
        if palette_end > data_offset:
            msg = (
                f"BMP palette of {palette_size} entries does not fit before "
                f"the pixel data at offset {data_offset}"
            )
            raise DecodeError(msg)
        if palette_end > len(data):
            msg = "BMP palette is truncated"
            raise DecodeError(msg)
    elif colors_used:
        palette_end = 14 + header_size + colors_used * palette_padding
        if palette_end > data_offset or palette_end > len(data):
            msg = "BMP palette is truncated"
            raise DecodeError(msg)
    if compression == 0:
        stride = ((width * bits + 31) >> 3) & ~3
        needed = stride * height
        if data_offset + needed > len(data):
            msg = (
                f"truncated BMP pixel data: row stride {stride} requires "
                f"{needed} bytes, only {len(data) - data_offset} present"
            )
            raise DecodeError(msg)
    return _BMPHeader(
        header_size=header_size,
        width=width,
        height=height,
        bit_depth=bits,
        compression=compression,
        palette_size=palette_size,
        data_offset=data_offset,
        row_order="top-down" if y_flip else "bottom-up",
    )


def _bmp_palette(data: bytes, header: _BMPHeader) -> bytes | None:
    if header.palette_size == 0:
        return None
    padding = 3 if header.header_size == 12 else 4
    start = 14 + header.header_size
    raw = data[start : start + header.palette_size * padding]
    out = bytearray(header.palette_size * 4)
    for index in range(header.palette_size):
        base = index * padding
        out[index * 4 : index * 4 + 4] = bytes(
            (raw[base + 2], raw[base + 1], raw[base], 255)
        )
    return bytes(out)


def _bmp_info(data: bytes, header: _BMPHeader) -> ImageInfo:
    return ImageInfo(
        format="BMP",
        width=header.width,
        height=header.height,
        mode="P" if header.bit_depth <= 8 else "RGB",
        bit_depth=header.bit_depth,
        channels=1 if header.bit_depth <= 8 else (4 if header.bit_depth == 32 else 3),
        palette=_bmp_palette(data, header),
        palette_size=header.palette_size,
        transparency=None,
        interlace=False,
        file_row_order=header.row_order,
        frame_count=1,
        frame_index=0,
    )


# --------------------------------------------------------------------
# TIFF


_TIFF_TYPE_SIZES = {
    1: 1,  # BYTE
    2: 1,  # ASCII
    3: 2,  # SHORT
    4: 4,  # LONG
    5: 8,  # RATIONAL
    6: 1,  # SBYTE
    7: 1,  # UNDEFINED
    8: 2,  # SSHORT
    9: 4,  # SLONG
    10: 8,  # SRATIONAL
    11: 4,  # FLOAT
    12: 8,  # DOUBLE
}

_TIFF_WIDTH = 256
_TIFF_HEIGHT = 257
_TIFF_BITS = 258
_TIFF_COMPRESSION = 259
_TIFF_PHOTOMETRIC = 262
_TIFF_STRIP_OFFSETS = 273
_TIFF_SAMPLES = 277
_TIFF_ROWS_PER_STRIP = 278
_TIFF_STRIP_COUNTS = 279
_TIFF_PLANAR = 284
_TIFF_ORIENTATION = 274
_TIFF_TILE_WIDTH = 322
_TIFF_TILE_HEIGHT = 323
_TIFF_TILE_OFFSETS = 324
_TIFF_TILE_COUNTS = 325

_TIFF_COMPRESSIONS = {
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    32773,
    32946,
    34676,
    34677,
    34712,
    34925,
    50000,
    50001,
}

_TIFF_BIT_DEPTHS = {1, 2, 4, 8, 12, 16, 32}
_MAX_TIFF_IFDS = 1 << 16
_MAX_IFD_ENTRIES = 1 << 12


def _tiff_endian(data: bytes) -> str:
    if len(data) < 8:
        msg = "truncated TIFF header"
        raise DecodeError(msg)
    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        msg = "not a TIFF file"
        raise DecodeError(msg)
    if struct.unpack(endian + "H", data[2:4])[0] != 42:
        msg = "invalid TIFF magic number"
        raise DecodeError(msg)
    first = struct.unpack(endian + "I", data[4:8])[0]
    if first < 8 or first + 2 > len(data):
        msg = f"invalid TIFF IFD offset {first}"
        raise DecodeError(msg)
    return endian


def _tiff_read(data: bytes, endian: str, offset: int, size: int) -> bytes:
    if offset < 0 or size < 0 or offset + size > len(data):
        msg = f"TIFF offset {offset}+{size} is outside the file"
        raise DecodeError(msg)
    return data[offset : offset + size]


def _tiff_ifd_offsets(data: bytes, endian: str) -> tuple[int, ...]:
    offsets: list[int] = []
    seen: set[int] = set()
    offset = struct.unpack(endian + "I", data[4:8])[0]
    while offset:
        if offset in seen or len(offsets) >= _MAX_TIFF_IFDS:
            msg = "TIFF IFD chain is cyclic or too long"
            raise DecodeError(msg)
        seen.add(offset)
        entries = struct.unpack(endian + "H", _tiff_read(data, endian, offset, 2))[0]
        if entries > _MAX_IFD_ENTRIES:
            msg = f"TIFF IFD has too many entries ({entries})"
            raise DecodeError(msg)
        ifd_end = offset + 2 + entries * 12 + 4
        if ifd_end > len(data):
            msg = "truncated TIFF IFD"
            raise DecodeError(msg)
        offsets.append(offset)
        offset = struct.unpack(endian + "I", data[offset + 2 + entries * 12 : ifd_end])[
            0
        ]
        if offset and (offset < 8 or offset + 2 > len(data)):
            msg = f"invalid TIFF IFD offset {offset}"
            raise DecodeError(msg)
    return tuple(offsets)


def _tiff_tags(
    data: bytes, endian: str, offset: int
) -> tuple[dict[int, list[int] | bytes], int]:
    entries = struct.unpack(endian + "H", _tiff_read(data, endian, offset, 2))[0]
    if entries > _MAX_IFD_ENTRIES:
        msg = f"TIFF IFD has too many entries ({entries})"
        raise DecodeError(msg)
    result: dict[int, list[int] | bytes] = {}
    for index in range(entries):
        entry_offset = offset + 2 + index * 12
        entry = _tiff_read(data, endian, entry_offset, 12)
        tag, type_id, count = struct.unpack(endian + "HHI", entry[:8])
        size = _TIFF_TYPE_SIZES.get(type_id)
        if size is None:
            continue
        value = entry[8:12]
        total = size * count
        if total > 4:
            value_offset = struct.unpack(endian + "I", value)[0]
            value = _tiff_read(data, endian, value_offset, total)
        else:
            value = value[:total]
        if type_id in (1, 6, 7, 2):
            result[tag] = bytes(value)
        elif type_id in (3, 8):
            result[tag] = list(struct.unpack(endian + "H" * count, value[: 2 * count]))
        elif type_id in (4, 9):
            result[tag] = list(struct.unpack(endian + "I" * count, value[: 4 * count]))
        elif type_id in (5, 10):
            values: list[int] = []
            for i in range(count):
                numerator, denominator = struct.unpack(
                    endian + "II", value[i * 8 : i * 8 + 8]
                )
                values.append(numerator // denominator if denominator else 0)
            result[tag] = values
        elif type_id in (11, 12):
            result[tag] = list(struct.unpack(endian + "f" * count, value[: 4 * count]))
    next_offset = struct.unpack(
        endian + "I", _tiff_read(data, endian, offset + 2 + entries * 12, 4)
    )[0]
    return result, next_offset


class _TIFFIFD(NamedTuple):
    width: int
    height: int
    bits: int
    samples: int
    compression: int
    photometric: int | None
    planar: int
    orientation: int
    strip_offsets: tuple[int, ...]
    strip_counts: tuple[int, ...]
    rows_per_strip: int
    tile_width: int | None
    tile_height: int | None
    tile_offsets: tuple[int, ...]
    tile_counts: tuple[int, ...]


def _tiff_ifd(data: bytes, endian: str, offset: int) -> _TIFFIFD:
    tags, _ = _tiff_tags(data, endian, offset)
    try:
        width = cast("list[int]", tags[_TIFF_WIDTH])[0]
        height = cast("list[int]", tags[_TIFF_HEIGHT])[0]
    except KeyError as e:
        msg = "Missing TIFF dimensions"
        raise DecodeError(msg) from e
    if width <= 0 or height <= 0:
        msg = f"invalid TIFF image size {width}x{height}"
        raise DecodeError(msg)
    bits_list = cast("list[int]", tags.get(_TIFF_BITS, [1]))
    samples = cast("list[int]", tags.get(_TIFF_SAMPLES, [1]))[0]
    if not 1 <= samples <= 64:
        msg = f"invalid TIFF samples per pixel {samples}"
        raise DecodeError(msg)
    for bits in bits_list:
        if bits not in _TIFF_BIT_DEPTHS:
            msg = f"unsupported TIFF bit depth {bits}"
            raise DecodeError(msg)
    bits = bits_list[0] if len(bits_list) == 1 else max(bits_list)
    compression = cast("list[int]", tags.get(_TIFF_COMPRESSION, [1]))[0]
    if compression not in _TIFF_COMPRESSIONS:
        msg = f"unsupported TIFF compression {compression}"
        raise DecodeError(msg)
    planar = cast("list[int]", tags.get(_TIFF_PLANAR, [1]))[0]
    if planar not in (1, 2):
        msg = f"invalid TIFF planar configuration {planar}"
        raise DecodeError(msg)
    orientation = cast("list[int]", tags.get(_TIFF_ORIENTATION, [1]))[0]
    strip_offsets = tuple(cast("list[int]", tags.get(_TIFF_STRIP_OFFSETS, [])))
    strip_counts = tuple(cast("list[int]", tags.get(_TIFF_STRIP_COUNTS, [])))
    rows_per_strip = cast("list[int]", tags.get(_TIFF_ROWS_PER_STRIP, [height]))[0]
    if rows_per_strip <= 0:
        msg = f"invalid TIFF rows per strip {rows_per_strip}"
        raise DecodeError(msg)
    tile_width = cast("list[int]", tags.get(_TIFF_TILE_WIDTH, [None]))[0]
    tile_height = cast("list[int]", tags.get(_TIFF_TILE_HEIGHT, [None]))[0]
    tile_offsets = tuple(cast("list[int]", tags.get(_TIFF_TILE_OFFSETS, [])))
    tile_counts = tuple(cast("list[int]", tags.get(_TIFF_TILE_COUNTS, [])))
    if strip_offsets:
        if len(strip_offsets) != len(strip_counts):
            msg = "TIFF strip offsets and byte counts differ in length"
            raise DecodeError(msg)
        for strip_offset, strip_count in zip(strip_offsets, strip_counts):
            if strip_offset < 0 or strip_count < 0:
                msg = "negative TIFF strip offset or byte count"
                raise DecodeError(msg)
            if strip_offset + strip_count > len(data):
                msg = "TIFF strip data is outside the file"
                raise DecodeError(msg)
    elif tile_offsets:
        if len(tile_offsets) != len(tile_counts):
            msg = "TIFF tile offsets and byte counts differ in length"
            raise DecodeError(msg)
        if not tile_width or not tile_height:
            msg = "TIFF tiled image has no tile dimensions"
            raise DecodeError(msg)
        for tile_offset, tile_count in zip(tile_offsets, tile_counts):
            if tile_offset < 0 or tile_count < 0:
                msg = "negative TIFF tile offset or byte count"
                raise DecodeError(msg)
            if tile_offset + tile_count > len(data):
                msg = "TIFF tile data is outside the file"
                raise DecodeError(msg)
    else:
        msg = "TIFF image has no strip or tile data"
        raise DecodeError(msg)
    photometric_values = tags.get(_TIFF_PHOTOMETRIC)
    photometric = None
    if isinstance(photometric_values, list) and photometric_values:
        photometric = photometric_values[0]
    return _TIFFIFD(
        width=width,
        height=height,
        bits=bits,
        samples=samples,
        compression=compression,
        photometric=photometric,
        planar=planar,
        orientation=orientation,
        strip_offsets=strip_offsets,
        strip_counts=strip_counts,
        rows_per_strip=rows_per_strip,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_offsets=tile_offsets,
        tile_counts=tile_counts,
    )


def _tiff_header(data: bytes) -> tuple[str, tuple[int, ...]]:
    endian = _tiff_endian(data)
    return endian, _tiff_ifd_offsets(data, endian)


def _tiff_info(
    data: bytes, frame: int, endian: str, offsets: tuple[int, ...]
) -> ImageInfo:
    if not 0 <= frame < len(offsets):
        msg = f"TIFF frame {frame} is out of range"
        raise IndexError(msg)
    ifd = _tiff_ifd(data, endian, offsets[frame])
    samples, bits = ifd.samples, ifd.bits
    if samples == 1:
        mode = "I;16" if bits == 16 else ("1" if bits == 1 else "L")
    elif samples == 2:
        mode = "LA"
    elif samples == 4:
        mode = "RGBA"
    else:
        mode = "RGB"
    return ImageInfo(
        format="TIFF",
        width=ifd.width,
        height=ifd.height,
        mode=mode,
        bit_depth=bits,
        channels=samples,
        palette=None,
        palette_size=0,
        transparency=None,
        interlace=False,
        file_row_order="top-down",
        frame_count=len(offsets),
        frame_index=frame,
    )


# --------------------------------------------------------------------
# Plane extraction


def _planes_from_image(image: Image.Image, info: ImageInfo) -> tuple[bytes, ...]:
    """Derive sample planes from a decoded Pillow image.

    Used for formats whose strip/tile decoding is handled by the C layer
    (TIFF and BMP).  Planes keep the pixel values that ``Image.open``
    reports, normalised to 16-bit big-endian samples for high depth modes.
    """

    mode = image.mode
    data = image.tobytes()
    if mode.startswith("I;16"):
        if mode in ("I;16", "I;16L", "I;16N") and sys.byteorder == "little":
            # The image stores little-endian samples; planes are big-endian.
            swapped = bytearray(data)
            swapped[0::2], swapped[1::2] = data[1::2], data[0::2]
            return (bytes(swapped),)
        if mode == "I;16B":
            return (data,)
        if mode in ("I;16", "I;16L") and sys.byteorder == "big":
            swapped = bytearray(data)
            swapped[0::2], swapped[1::2] = data[1::2], data[0::2]
            return (bytes(swapped),)
        return (data,)
    if mode in ("RGB", "RGBA", "LA", "RGBX"):
        channels = len(mode)
        planes = [bytearray(image.width * image.height) for _ in range(channels)]
        for index in range(channels):
            planes[index][:] = data[index::channels]
        return tuple(bytes(plane) for plane in planes)
    if mode == "L" and info.bit_depth < 8:
        # Unpack the 8-bit values back to their original sample width.
        width, height = image.size
        bits = info.bit_depth
        row_bytes = (width * bits + 7) // 8
        out = bytearray(row_bytes * height)
        samples = data[: width * height]
        max_value = (1 << bits) - 1
        for index, value in enumerate(samples):
            sample = (value * max_value + 127) // 255
            x = index % width
            y = index // width
            target_byte = y * row_bytes + (x * bits) // 8
            shift = 8 - bits - (x * bits) % 8
            out[target_byte] |= sample << shift
        return (bytes(out),)
    return (data,)


def _tiff_planes(
    data: bytes, endian: str, ifd: _TIFFIFD, image: Image.Image
) -> tuple[bytes, ...]:
    """Return raw sample planes for a TIFF frame.

    Only uncompressed, contiguous strip layouts are decoded here; anything
    else (tiles, planar storage, compression, multiple strips) falls back to
    the plane values that Pillow produced for the decoded image.
    """

    width, height, samples, bits = ifd.width, ifd.height, ifd.samples, ifd.bits
    if (
        ifd.compression != 1
        or ifd.planar != 1
        or ifd.tile_offsets
        or len(ifd.strip_offsets) != 1
        or ifd.strip_counts[0] < width * height * samples * max(1, bits // 8)
    ):
        return _planes_from_image(image, _tiff_image_info(image, ifd))
    raw = data[ifd.strip_offsets[0] : ifd.strip_offsets[0] + ifd.strip_counts[0]]
    row_bytes = (width * samples * bits + 7) // 8
    if len(raw) < row_bytes * height:
        return _planes_from_image(image, _tiff_image_info(image, ifd))
    planes: list[bytearray] = []
    sample_bytes = 2 if bits == 16 else 1
    if bits in (8, 16):
        planes = [bytearray(width * height * sample_bytes) for _ in range(samples)]
        for y in range(height):
            row_start = y * row_bytes
            for x in range(width):
                index = (y * width + x) * sample_bytes
                base = row_start + x * samples * sample_bytes
                for channel in range(samples):
                    sample = raw[
                        base
                        + channel * sample_bytes : base
                        + (channel + 1) * sample_bytes
                    ]
                    if sample_bytes == 2 and endian == "<":
                        sample = sample[::-1]
                    planes[channel][index : index + sample_bytes] = sample
        return tuple(bytes(plane) for plane in planes)
    if bits < 8 and samples == 1:
        # One raw sample per byte, as with the PNG sub-8-bit planes.
        out = bytearray(width * height)
        mask = (1 << bits) - 1
        for y in range(height):
            row_start = y * row_bytes
            for x in range(width):
                byte = raw[row_start + (x * bits) // 8]
                shift = 8 - bits - (x * bits) % 8
                out[y * width + x] = (byte >> shift) & mask
        return (bytes(out),)
    return _planes_from_image(image, _tiff_image_info(image, ifd))


def _tiff_image_info(image: Image.Image, ifd: _TIFFIFD) -> ImageInfo:
    return ImageInfo(
        format="TIFF",
        width=image.size[0],
        height=image.size[1],
        mode=image.mode,
        bit_depth=ifd.bits,
        channels=ifd.samples,
        palette=None,
        palette_size=0,
        transparency=None,
        interlace=False,
        file_row_order="top-down",
        frame_count=1,
        frame_index=0,
    )


# --------------------------------------------------------------------
# Public API


def probe(
    source: object, *, frame: int = 0, budget: MemoryBudget | None = None
) -> ImageInfo:
    """Return metadata for a high depth image without decoding its pixels.

    :param source: bytes, path, file object or iterator of byte chunks.
    :param frame: TIFF frame number.
    :param budget: optional :class:`MemoryBudget` checked against the decoded
        size before any pixel buffer is allocated.
    """

    data = _read_source(source)
    kind = _identify(data)
    info = _info(data, kind, frame)
    if budget is not None:
        budget.check(
            info.size,
            channels=info.channels,
            bytes_per_sample=2 if info.bit_depth > 8 else 1,
        )
    return info


def _info(data: bytes, kind: str, frame: int) -> ImageInfo:
    if kind == "PNG":
        return _png_info(_png_structure(_png_chunks(data)))
    if kind == "BMP":
        return _bmp_info(data, _bmp_header(data))
    endian, offsets = _tiff_header(data)
    return _tiff_info(data, frame, endian, offsets)


def decode_planes(
    source: object, *, frame: int = 0, budget: MemoryBudget | None = None
) -> tuple[tuple[bytes, ...], ImageInfo]:
    """Decode the raw sample planes and metadata of a high depth image.

    :param source: bytes, path, file object or iterator of byte chunks.
    :param frame: TIFF frame number.
    :param budget: optional :class:`MemoryBudget`.
    :returns: ``(planes, info)``.
    """

    data = _read_source(source)
    return _decode_planes(data, frame, budget)


def _decode_planes(
    data: bytes, frame: int, budget: MemoryBudget | None
) -> tuple[tuple[bytes, ...], ImageInfo]:
    kind = _identify(data)
    info = _info(data, kind, frame)
    _check_budget(info, budget)
    image = _open(data, frame)
    if kind == "PNG":
        planes = _png_planes(data, _png_structure(_png_chunks(data)))
    elif kind == "TIFF":
        endian, offsets = _tiff_header(data)
        ifd = _tiff_ifd(data, endian, offsets[frame])
        planes = _tiff_planes(data, endian, ifd, image)
    else:
        planes = _planes_from_image(image, info)
    info = _align_info(info, image)
    return planes, info


def _check_budget(info: ImageInfo, budget: MemoryBudget | None) -> None:
    if budget is not None:
        budget.check(
            info.size,
            channels=info.channels,
            bytes_per_sample=2 if info.bit_depth > 8 else 1,
        )


def _align_info(info: ImageInfo, image: Image.Image) -> ImageInfo:
    return info._replace(
        width=image.size[0],
        height=image.size[1],
        mode=image.mode,
    )


def _open(data: bytes, frame: int) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    try:
        if frame:
            frames = getattr(image, "n_frames", 1)
            if frame >= frames:
                msg = f"frame {frame} is out of range"
                raise IndexError(msg)
            image.seek(frame)
        image.load()
    except BaseException:
        image.close()
        raise
    return image


def decode(
    source: object,
    *,
    frame: int = 0,
    budget: MemoryBudget | None = None,
) -> DecodedImage:
    """Decode a PNG, TIFF or BMP source into a :class:`DecodedImage`.

    All input methods return identical pixels and metadata for the same data.
    Structural errors are raised before pixels are allocated.

    :param source: bytes, path, file object or iterator of byte chunks.
    :param frame: TIFF frame number.
    :param budget: optional :class:`MemoryBudget`.
    """

    data = _read_source(source)
    planes, info = _decode_planes(data, frame, budget)
    image = _open(data, frame)
    return DecodedImage(image, info, planes)


def read(
    source: object,
    *,
    frame: int = 0,
    budget: MemoryBudget | None = None,
) -> Image.Image:
    """Decode a source and return its regular Pillow image.

    Equivalent to ``decode(source, ...).image`` but skips plane extraction.
    """

    data = _read_source(source)
    kind = _identify(data)
    info = _info(data, kind, frame)
    _check_budget(info, budget)
    return _open(data, frame)


class StreamReader:
    """Incrementally feed byte chunks and decode the completed image.

    The received bytes are retained so the result has the same planes and
    metadata as :func:`decode` on the concatenated chunks.

    :param budget: optional :class:`MemoryBudget` checked once the header is
        available.
    """

    def __init__(self, budget: MemoryBudget | None = None) -> None:
        self._data = bytearray()
        self._budget = budget

    def feed(self, chunk: bytes | bytearray | memoryview) -> None:
        self._data += _check_bytes(chunk)

    def close(self) -> DecodedImage:
        """Finish parsing and return the decoded image."""

        data = bytes(self._data)
        planes, info = _decode_planes(data, 0, self._budget)
        image = _open(data, 0)
        return DecodedImage(image, info, planes)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
