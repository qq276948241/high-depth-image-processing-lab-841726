#
# The Python Imaging Library.
#
# Strict TIFF decoder used by :mod:`PIL.highdepth`.
#
# See the README file for information on usage and redistribution.
#
"""TIFF decoder.

Supports baseline and extension TIFF: little and big endian byte orders,
uncompressed, LZW, Deflate and PackBits strips/tiles, chunky and planar
sample layouts, 8 and 16 bits per sample, palettes and RGBA, and EXIF
orientation.

The IFD is parsed and every offset is range-checked before any pixel buffer is
allocated, so a file that lies about its dimensions, strip offsets or tag
counts fails with a stable error instead of reading out of bounds.  Only one
page is decoded by :func:`decode_tiff_page`; :func:`PIL.highdepth.read` and
:func:`PIL.highdepth.read_frames` iterate pages.
"""

from __future__ import annotations

import struct
from array import array
from typing import TYPE_CHECKING

from .codecs import (
    apply_predictor,
    deflate_decode,
    lzw_decode,
    packbits_decode,
    unpack_bits,
)
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
    scale_sample,
    storage_bytes,
)
from .source import Source, open_source

if TYPE_CHECKING:
    from .budget import CancellationToken, MemoryBudget

_TIFF_MAGIC = b"II\x2a\x00"
_TIFF_MAGIC_BIG = b"MM\x00\x2a"
_BIGTIFF_MAGIC = b"II\x2b\x00"
_BIGTIFF_MAGIC_BIG = b"MM\x00\x2b"

# Tag numbers
_IMAGE_WIDTH = 256
_IMAGE_LENGTH = 257
_BITS_PER_SAMPLE = 258
_COMPRESSION = 259
_PHOTOMETRIC = 262
_FILL_ORDER = 266
_STRIP_OFFSETS = 273
_ORIENTATION = 274
_SAMPLES_PER_PIXEL = 277
_ROWS_PER_STRIP = 278
_STRIP_BYTE_COUNTS = 279
_PLANAR_CONFIG = 284
_PREDICTOR = 317
_COLORMAP = 320
_TILE_WIDTH = 322
_TILE_LENGTH = 323
_TILE_OFFSETS = 324
_TILE_BYTE_COUNTS = 325
_EXTRA_SAMPLES = 338
_SAMPLE_FORMAT = 339

_SAMPLE_FORMAT_UINT = 1

# TIFF field types: code -> (struct format character, byte size)
_TYPES = {
    1: ("B", 1),  # BYTE
    2: ("B", 1),  # ASCII
    3: ("H", 2),  # SHORT
    4: ("I", 4),  # LONG
    5: ("I", 8),  # RATIONAL (two LONGs; numeric reinterpretation is 8 bytes)
    6: ("b", 1),  # SBYTE
    7: ("B", 1),  # UNDEFINED
    8: ("h", 2),  # SSHORT
    9: ("i", 4),  # SLONG
    10: ("i", 8),  # SRATIONAL
    11: ("f", 4),  # FLOAT
    12: ("d", 8),  # DOUBLE
    13: ("I", 4),  # IFD
    16: ("Q", 8),  # LONG8
    17: ("q", 8),  # SLONG8
    18: ("Q", 8),  # IFD8
}

_COMPRESSIONS = {
    1: "raw",
    5: "lzw",
    8: "deflate",
    32773: "packbits",
    32946: "deflate",
}
_SUPPORTED_PHOTOMETRIC = {0, 1, 2, 3, 4}

# Tags whose value is an offset list; SHORT and LONG both occur in the wild.
_OFFSET_TAGS = {_STRIP_OFFSETS, _STRIP_BYTE_COUNTS, _TILE_OFFSETS, _TILE_BYTE_COUNTS}


class _Tiff:
    """Byte-order aware reader for a TIFF byte stream."""

    def __init__(self, source: Source, big_endian: bool, bigtiff: bool) -> None:
        self.source = source
        self.byte_order = ">" if big_endian else "<"
        self.bigtiff = bigtiff
        self.entry_size = 20 if bigtiff else 12
        self.count_size = 8 if bigtiff else 2
        self.offset_size = 8 if bigtiff else 4
        self.inline_size = self.entry_size - (12 if bigtiff else 8)
        # ``layout`` is set per-page in ``_decode_page``; ``extent`` records
        # the end of the last structure read so trailing bytes can be detected.
        self.layout: dict[int, tuple[int, int, bytes]] = {}
        self.extent = 0

    def ifd_count(self, offset: int) -> int:
        if self.bigtiff:
            (count,) = struct.unpack(
                self.byte_order + "Q", self.source.read_exact(offset, 8, "IFD count")
            )
        else:
            (count,) = struct.unpack(
                self.byte_order + "H", self.source.read_exact(offset, 2, "IFD count")
            )
        return count

    def entry(self, ifd_offset: int, index: int) -> tuple[int, int, int, bytes]:
        base = ifd_offset + self.count_size + index * self.entry_size
        data = self.source.read_exact(base, self.entry_size, "IFD entry")
        tag, type_id = struct.unpack(self.byte_order + "HH", data[:4])
        if type_id not in _TYPES:
            msg = f"unsupported TIFF field type {type_id} for tag {tag}"
            raise UnsupportedFeatureError(msg)
        if self.bigtiff:
            (count,) = struct.unpack(self.byte_order + "Q", data[4:12])
        else:
            (count,) = struct.unpack(self.byte_order + "I", data[4:8])
        value_field = data[self.entry_size - self.inline_size :]
        return tag, type_id, count, value_field

    def raw_values(
        self, tag: int, type_id: int, count: int, value_field: bytes
    ) -> bytes:
        code, unit = _TYPES[type_id]
        total = checked_mul(count, unit, "TIFF tag size")
        if total <= self.inline_size:
            return value_field[:total]
        (offset,) = struct.unpack(
            self.byte_order + ("Q" if self.bigtiff else "I"), value_field
        )
        return self.source.read_exact(offset, total, f"TIFF tag {tag}")

    def next_ifd(self, ifd_offset: int, count: int) -> tuple[int, int]:
        """Return ``(next_ifd_offset, end_of_this_ifd)``."""
        position = ifd_offset + self.count_size + count * self.entry_size
        end = position + self.offset_size
        return (
            struct.unpack(
                self.byte_order + ("Q" if self.bigtiff else "I"),
                self.source.read_exact(position, self.offset_size, "next IFD"),
            )[0],
            end,
        )

    def numeric(
        self, tag: int, type_id: int, count: int, value_field: bytes
    ) -> list[int]:
        raw = self.raw_values(tag, type_id, count, value_field)
        if type_id in (5, 10):
            # Rationals: expose the numerator only, which is what callers of
            # the pipeline ever ask for (resolution etc. is carried in info).
            fmt = self.byte_order + "I" * (len(raw) // 4)
            numbers = list(struct.unpack(fmt, raw))
            return numbers[0::2] if type_id == 5 else list(numbers)
        code = _TYPES[type_id][0]
        size = struct.calcsize(code)
        return list(struct.unpack(self.byte_order + code * (len(raw) // size), raw))

    def numeric_values(self, tag: int) -> list[int] | None:
        """Return the values of ``tag`` across all IFDs, or ``None``."""
        if tag not in self.layout:
            return None
        type_id, count, value_field = self.layout[tag]
        return self.numeric(tag, type_id, count, value_field)

    def scalar(self, tag: int, default: int | None = None) -> int:
        values = self.numeric_values(tag)
        if not values:
            if default is None:
                msg = f"TIFF is missing required tag {tag}"
                raise FormatError(msg)
            return default
        return values[0]


def _parse_layout(tiff: _Tiff, ifd_offset: int) -> dict[int, tuple[int, int, bytes]]:
    count = tiff.ifd_count(ifd_offset)
    if count > 4096:
        msg = f"IFD declares {count} entries, more than any sane file"
        raise ChunkLengthError(msg)
    table_end = ifd_offset + tiff.count_size + count * tiff.entry_size
    if checked_add(table_end, tiff.offset_size, "IFD") > tiff.source.size:
        msg = "IFD extends past the end of the file"
        raise TruncatedDataError(msg)
    layout: dict[int, tuple[int, int, bytes]] = {}
    for index in range(count):
        tag, type_id, item_count, value_field = tiff.entry(ifd_offset, index)
        if (
            checked_mul(item_count, _TYPES[type_id][1], "TIFF tag size")
            > tiff.source.size
        ):
            msg = f"tag {tag} declares {item_count} values, larger than the file"
            raise ChunkLengthError(msg)
        layout[tag] = (type_id, item_count, value_field)
    return layout


def _ifd_offsets(source: Source) -> tuple[_Tiff, list[int]]:
    """Validate the header and return the reader plus every IFD offset."""
    if source.size < 8:
        msg = "truncated TIFF header"
        raise TruncatedDataError(msg)
    magic = source.read_exact(0, 4, "TIFF magic")
    if magic in (_TIFF_MAGIC, _BIGTIFF_MAGIC):
        big_endian = False
    elif magic in (_TIFF_MAGIC_BIG, _BIGTIFF_MAGIC_BIG):
        big_endian = True
    else:
        msg = "not a TIFF stream"
        raise FormatError(msg)
    bigtiff = magic in (_BIGTIFF_MAGIC, _BIGTIFF_MAGIC_BIG)
    tiff = _Tiff(source, big_endian, bigtiff)
    if bigtiff:
        if source.size < 16:
            msg = "truncated BigTIFF header"
            raise TruncatedDataError(msg)
        offset_size, _reserved = struct.unpack(
            tiff.byte_order + "HH", source.read_exact(4, 4)
        )
        if offset_size != 8:
            msg = f"unsupported BigTIFF offset size {offset_size}"
            raise UnsupportedFeatureError(msg)
        (first_ifd,) = struct.unpack(tiff.byte_order + "Q", source.read_exact(8, 8))
        tiff.extent = 16
    else:
        (first_ifd,) = struct.unpack(tiff.byte_order + "I", source.read_exact(4, 4))
        tiff.extent = 8
    if first_ifd == 0:
        msg = "TIFF has no image file directory"
        raise FormatError(msg)

    offsets: list[int] = []
    seen: set[int] = set()
    offset = first_ifd
    while offset:
        if offset in seen:
            msg = "TIFF IFD chain contains a cycle"
            raise FormatError(msg)
        if len(offsets) >= 65536:
            msg = "TIFF has more than 65536 directories"
            raise ChunkLengthError(msg)
        if offset + tiff.count_size > source.size:
            msg = f"IFD offset {offset} is outside the file"
            raise FormatError(msg)
        seen.add(offset)
        offsets.append(offset)
        count = tiff.ifd_count(offset)
        next_offset, end = tiff.next_ifd(offset, count)
        tiff.extent = max(tiff.extent, end)
        offset = next_offset
    return tiff, offsets


def decode_tiff_page(
    source: object,
    page: int = 0,
    *,
    budget: MemoryBudget | None = None,
    token: CancellationToken | None = None,
) -> ImageData:
    """Decode one page of a TIFF from any supported source."""
    src = open_source(source, budget=budget, token=token)
    owns = src is not source
    try:
        tiff, offsets = _ifd_offsets(src)
        if not 0 <= page < len(offsets):
            msg = f"TIFF page {page} out of range ({len(offsets)} pages)"
            raise IndexError(msg)
        return _decode_page(tiff, offsets[page], page, budget, token)
    finally:
        if owns:
            src.close()


def tiff_page_count(source: object) -> int:
    """Return the number of directories in a TIFF stream."""
    src = open_source(source)
    owns = src is not source
    try:
        _tiff, offsets = _ifd_offsets(src)
        return len(offsets)
    finally:
        if owns:
            src.close()


def _decode_page(
    tiff: _Tiff,
    ifd_offset: int,
    page_index: int,
    budget: MemoryBudget | None,
    token: CancellationToken | None,
) -> ImageData:
    tiff.layout = _parse_layout(tiff, ifd_offset)

    width = tiff.scalar(_IMAGE_WIDTH)
    height = tiff.scalar(_IMAGE_LENGTH)
    if width <= 0 or height <= 0:
        msg = f"invalid TIFF size {(width, height)}"
        raise FormatError(msg)
    check_image_size(width, height)
    samples_per_pixel = tiff.scalar(_SAMPLES_PER_PIXEL, 1)
    bits_values = tiff.numeric_values(_BITS_PER_SAMPLE)
    photometric_default = 1
    if bits_values is None:
        bits_per_sample = (
            1 if tiff.scalar(_PHOTOMETRIC, photometric_default) in (0, 1) else 8
        )
    else:
        if len(bits_values) not in (1, samples_per_pixel):
            msg = "bits per sample count does not match samples per pixel"
            raise FormatError(msg)
        bits_per_sample = bits_values[0]
        if any(bits != bits_per_sample for bits in bits_values):
            msg = "mixed bits per sample are not supported"
            raise UnsupportedFeatureError(msg)
    compression = tiff.scalar(_COMPRESSION, 1)
    photometric = tiff.scalar(_PHOTOMETRIC, photometric_default)
    planar = tiff.scalar(_PLANAR_CONFIG, 1)
    predictor = tiff.scalar(_PREDICTOR, 1)
    orientation = tiff.scalar(_ORIENTATION, 1)
    sample_format = tiff.scalar(_SAMPLE_FORMAT, _SAMPLE_FORMAT_UINT)
    fill_order = tiff.scalar(_FILL_ORDER, 1)
    extra = tiff.numeric_values(_EXTRA_SAMPLES) or []

    if sample_format != _SAMPLE_FORMAT_UINT:
        msg = f"unsupported TIFF sample format {sample_format}"
        raise UnsupportedFeatureError(msg)
    if bits_per_sample not in (1, 2, 4, 8, 16):
        msg = f"unsupported TIFF bit depth {bits_per_sample}"
        raise UnsupportedFeatureError(msg)
    if fill_order != 1:
        msg = "TIFF fill order 2 is not supported"
        raise UnsupportedFeatureError(msg)
    if compression not in _COMPRESSIONS:
        msg = f"unsupported TIFF compression {compression}"
        raise UnsupportedFeatureError(msg)
    if photometric not in _SUPPORTED_PHOTOMETRIC:
        msg = f"unsupported TIFF photometric interpretation {photometric}"
        raise UnsupportedFeatureError(msg)
    if planar not in (1, 2):
        msg = f"unsupported TIFF planar configuration {planar}"
        raise UnsupportedFeatureError(msg)
    if predictor not in (1, 2):
        msg = f"unsupported TIFF predictor {predictor}"
        raise UnsupportedFeatureError(msg)
    if orientation not in range(1, 9):
        msg = f"invalid TIFF orientation {orientation}"
        raise FormatError(msg)

    if photometric == 3:
        channels = 1
        mode = "P"
    elif photometric == 2:
        if samples_per_pixel not in (3, 4):
            msg = f"RGB TIFF with {samples_per_pixel} samples is not supported"
            raise UnsupportedFeatureError(msg)
        channels = samples_per_pixel
        mode = "RGBA" if samples_per_pixel == 4 else "RGB"
    else:  # 0, 1, 4
        if samples_per_pixel not in (1, 2):
            msg = f"grayscale TIFF with {samples_per_pixel} samples is not supported"
            raise UnsupportedFeatureError(msg)
        channels = samples_per_pixel
        mode = "LA" if samples_per_pixel == 2 else "L"

    storage_bits = 8 if bits_per_sample <= 8 else 16
    if budget is not None:
        budget.check(storage_bytes(width, height, channels, storage_bits), "TIFF image")

    palette = None
    if photometric == 3:
        colormap = tiff.numeric_values(_COLORMAP)
        if colormap is None:
            msg = "palette TIFF has no colour map"
            raise IllegalPaletteError(msg)
        palette = _build_colormap(colormap, samples_per_pixel)

    pixel_count = checked_mul(width, height, "TIFF pixel count")
    decoded: list[int] = [0] * (pixel_count * channels)

    tile_width_tag = tiff.numeric_values(_TILE_WIDTH)
    if tile_width_tag is not None:
        _decode_tiles(
            tiff,
            decoded,
            width,
            height,
            channels,
            bits_per_sample,
            planar,
            predictor,
            photometric,
            mode,
            token,
        )
    else:
        _decode_strips(
            tiff,
            decoded,
            width,
            height,
            channels,
            bits_per_sample,
            planar,
            predictor,
            photometric,
            mode,
            token,
        )

    if photometric == 0 and bits_per_sample == 1 and samples_per_pixel == 1:
        # min-is-white: a zero bit is white, so invert the bilevel samples.
        decoded = [1 - value for value in decoded]

    # Samples are returned in their stored (file) order and the orientation tag
    # is carried in ``ImageData.orientation``; :func:`normalize_orientation`
    # applies it on request.  Rotating here would make an orientation-preserving
    # re-encode rotate the pixels a second time.

    info: dict[str, object] = {
        "page": page_index,
        "ifd_offset": ifd_offset,
        "photometric": photometric,
        "bits_per_sample": bits_per_sample,
        "compression": compression,
        "orientation": orientation,
    }
    if extra:
        info["extra_samples"] = extra

    if mode == "P":
        assert palette is not None
        max_index = max(decoded) if decoded else 0
        indices = bytes(decoded)
        bit_depth = max(1, max_index.bit_length())
        return ImageData(
            format="TIFF",
            mode="P",
            width=width,
            height=height,
            bit_depth=min(bit_depth, 8),
            storage_bits=8,
            channels=1,
            samples=indices,
            palette=palette,
            row_stride=row_bytes(width, 1, 8),
            orientation=orientation,
            info=info,
        )

    if bits_per_sample == 1:
        output = [255 if sample else 0 for sample in decoded]
        bit_depth = 8
        storage_bits = 8
    elif bits_per_sample < 8:
        output = [scale_sample(sample, bits_per_sample, 8) for sample in decoded]
        bit_depth = 8
        storage_bits = 8
    else:
        output = decoded
        bit_depth = bits_per_sample

    samples: bytes | array[int] = (
        array("H", output) if storage_bits == 16 else bytes(output)
    )
    return ImageData(
        format="TIFF",
        mode=mode,
        width=width,
        height=height,
        bit_depth=bit_depth,
        storage_bits=storage_bits,
        channels=channels,
        samples=samples,
        row_stride=row_bytes(width, channels, storage_bits),
        orientation=orientation,
        info=info,
    )


def _blocks_layout(
    tiff: _Tiff,
    width: int,
    height: int,
    channels: int,
    planar: int,
    is_tiled: bool,
) -> tuple[int, int, int]:
    if is_tiled:
        tile_width = tiff.scalar(_TILE_WIDTH)
        tile_length = tiff.scalar(_TILE_LENGTH)
        if tile_width <= 0 or tile_length <= 0:
            msg = "invalid TIFF tile size"
            raise FormatError(msg)
        across = (width + tile_width - 1) // tile_width
        down = (height + tile_length - 1) // tile_length
        per_plane = across * down
        return per_plane * (channels if planar == 2 else 1), tile_width, tile_length
    rows = tiff.scalar(_ROWS_PER_STRIP, height)
    if rows <= 0:
        msg = f"invalid TIFF rows per strip {rows}"
        raise FormatError(msg)
    per_plane = (height + rows - 1) // rows
    return per_plane * (channels if planar == 2 else 1), width, rows


def _block_table(
    tiff: _Tiff, expected: int, offsets_tag: int, counts_tag: int
) -> tuple[list[int], list[int]]:
    offsets = tiff.numeric_values(offsets_tag)
    counts = tiff.numeric_values(counts_tag)
    if offsets is None:
        msg = f"TIFF is missing the {offsets_tag} offset table"
        raise FormatError(msg)
    if counts is None:
        msg = f"TIFF is missing the {counts_tag} byte count table"
        raise FormatError(msg)
    if len(offsets) < expected:
        msg = f"TIFF declares {len(offsets)} blocks but the layout needs {expected}"
        raise ChunkLengthError(msg)
    if len(counts) < expected:
        msg = f"TIFF has {len(counts)} byte counts for {expected} blocks"
        raise ChunkLengthError(msg)
    return offsets[:expected], counts[:expected]


def _decode_strips(
    tiff: _Tiff,
    decoded: list[int],
    width: int,
    height: int,
    channels: int,
    bits_per_sample: int,
    planar: int,
    predictor: int,
    photometric: int,
    mode: str,
    token: CancellationToken | None,
) -> None:
    expected, _block_width, rows_per_strip = _blocks_layout(
        tiff, width, height, channels, planar, False
    )
    offsets, counts = _block_table(tiff, expected, _STRIP_OFFSETS, _STRIP_BYTE_COUNTS)
    for strip_index in range(expected):
        if token is not None:
            token.raise_if_cancelled()
        plane, index = (
            (strip_index % channels, strip_index // channels)
            if planar == 2
            else (0, strip_index)
        )
        y0 = index * rows_per_strip
        rows = min(rows_per_strip, height - y0)
        data = _decode_block(
            tiff,
            offsets[strip_index],
            counts[strip_index],
            width,
            rows,
            bits_per_sample,
            1 if planar == 2 else channels,
            predictor,
        )
        _blit(
            decoded,
            data,
            width,
            height,
            channels,
            0,
            y0,
            width,
            rows,
            plane,
            planar,
        )


def _decode_tiles(
    tiff: _Tiff,
    decoded: list[int],
    width: int,
    height: int,
    channels: int,
    bits_per_sample: int,
    planar: int,
    predictor: int,
    photometric: int,
    mode: str,
    token: CancellationToken | None,
) -> None:
    expected, tile_width, tile_length = _blocks_layout(
        tiff, width, height, channels, planar, True
    )
    offsets, counts = _block_table(tiff, expected, _TILE_OFFSETS, _TILE_BYTE_COUNTS)
    across = (width + tile_width - 1) // tile_width
    for tile_index in range(expected):
        if token is not None:
            token.raise_if_cancelled()
        plane, index = (
            (tile_index % channels, tile_index // channels)
            if planar == 2
            else (0, tile_index)
        )
        row = index // across
        column = index % across
        x0 = column * tile_width
        y0 = row * tile_length
        rows = min(tile_length, height - y0)
        # Tiles are always full size on disk, even at the right/bottom edge.
        data = _decode_block(
            tiff,
            offsets[tile_index],
            counts[tile_index],
            tile_width,
            tile_length,
            bits_per_sample,
            1 if planar == 2 else channels,
            predictor,
        )
        _blit(
            decoded,
            data,
            width,
            height,
            channels,
            x0,
            y0,
            tile_width,
            tile_length,
            plane,
            planar,
            copy_width=min(tile_width, width - x0),
            copy_rows=rows,
        )


def _decode_block(
    tiff: _Tiff,
    offset: int,
    byte_count: int,
    block_width: int,
    rows: int,
    bits_per_sample: int,
    samples_per_pixel: int,
    predictor: int,
) -> list[int]:
    source = tiff.source
    if offset < 0 or byte_count < 0:
        msg = "negative TIFF block offset or byte count"
        raise FormatError(msg)
    if checked_add(offset, byte_count, "TIFF block") > source.size:
        msg = f"TIFF block at {offset} with {byte_count} bytes is outside the file"
        raise TruncatedDataError(msg)
    codec = _COMPRESSIONS[tiff.scalar(_COMPRESSION, 1)]
    big_endian = tiff.byte_order == ">"
    row_length = row_bytes(block_width, samples_per_pixel, bits_per_sample)
    expected = checked_mul(row_length, rows, "TIFF block size")
    raw = source.read_exact(offset, byte_count, "TIFF block")
    if codec == "raw":
        if byte_count < expected:
            msg = f"raw TIFF block has {byte_count} bytes, expected {expected}"
            raise TruncatedDataError(msg)
        decompressed = raw[:expected]
    elif byte_count == 0:
        msg = "compressed TIFF block has zero bytes"
        raise TruncatedDataError(msg)
    elif codec == "lzw":
        decompressed = lzw_decode(raw, expected)
    elif codec == "packbits":
        decompressed = packbits_decode(raw, expected)
    else:
        decompressed = deflate_decode(raw, expected)
    if predictor == 2:
        decompressed = apply_predictor(
            decompressed,
            rows,
            row_length,
            bits_per_sample,
            samples_per_pixel,
            predictor,
            big_endian=big_endian,
        )
    return unpack_bits(
        decompressed,
        bits_per_sample,
        block_width * rows * samples_per_pixel,
        big_endian=big_endian,
    )


def _blit(
    decoded: list[int],
    data: list[int],
    width: int,
    height: int,
    channels: int,
    x0: int,
    y0: int,
    block_width: int,
    block_rows: int,
    plane: int,
    planar: int,
    *,
    copy_width: int | None = None,
    copy_rows: int | None = None,
) -> None:
    if copy_width is None:
        copy_width = min(block_width, width - x0)
    if copy_rows is None:
        copy_rows = min(block_rows, height - y0)
    source_samples = 1 if planar == 2 else channels
    for row in range(copy_rows):
        source_base = row * block_width * source_samples
        target_base = ((y0 + row) * width + x0) * channels
        if source_samples == 1 and channels == 1:
            decoded[target_base : target_base + copy_width] = data[
                source_base : source_base + copy_width
            ]
            continue
        for column in range(copy_width):
            for band in range(source_samples):
                target_band = plane if planar == 2 else band
                value = data[source_base + column * source_samples + band]
                decoded[target_base + column * channels + target_band] = value


def _build_colormap(values: list[int], samples_per_pixel: int) -> Palette:
    if samples_per_pixel != 1:
        msg = "colour map requires one sample per pixel"
        raise UnsupportedFeatureError(msg)
    if len(values) % 3 or not values:
        msg = "colour map must have three equally sized tables"
        raise IllegalPaletteError(msg)
    entries = len(values) // 3
    if entries > 65536:
        msg = f"colour map has {entries} entries, maximum is 65536"
        raise IllegalPaletteError(msg)
    reds, greens, blues = (
        values[:entries],
        values[entries : 2 * entries],
        values[2 * entries :],
    )
    return Palette(
        tuple(
            (reds[i] >> 8, greens[i] >> 8, blues[i] >> 8, 255) for i in range(entries)
        )
    )


__all__ = ["decode_tiff_page", "tiff_page_count"]
