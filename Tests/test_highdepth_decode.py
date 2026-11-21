"""Bit-fidelity and error-path tests for the PNG, TIFF and BMP decoders."""

from __future__ import annotations

import io
import struct
import zlib
from array import array

import pytest

from PIL import Image
from PIL.highdepth import (
    ChecksumError,
    ChunkLengthError,
    FormatError,
    IllegalPaletteError,
    TrailingDataError,
    TruncatedDataError,
    UnsupportedFeatureError,
    read,
    read_header,
)
from PIL.highdepth import fixtures as fx
from PIL.highdepth.model import ImageData

from .highdepth_fixtures.helpers import make_palette, make_rgb, save

IMAGE_SOURCES = [
    pytest.param("bytes", id="bytes"),
    pytest.param("file", id="file"),
    pytest.param("chunks", id="chunks"),
]


def _via(source_kind: str, payload: bytes):
    if source_kind == "bytes":
        return payload
    if source_kind == "file":
        return io.BytesIO(payload)
    return iter([payload[i : i + 3] for i in range(0, len(payload), 3)])


@pytest.mark.parametrize("source_kind", IMAGE_SOURCES)
@pytest.mark.parametrize("format", ("PNG", "TIFF", "BMP"))
def test_three_input_styles_agree(source_kind: str, format: str) -> None:
    payload = save(make_rgb(), format)
    data = read(_via(source_kind, payload), format=format)
    expected = read(payload, format=format)
    assert data.sha256() == expected.sha256()
    assert data.describe() == expected.describe()


# ------------------------------------------------------------------------- PNG


def test_png_16bit_gray_preserves_samples() -> None:
    values = [0, 1, 255, 256, 1000, 0x8000, 0xFFFE, 0xFFFF]
    payload = fx.grayscale16_png(values, 4, 2)
    data = read(payload)
    assert data.mode == "L"
    assert data.bit_depth == 16
    assert data.storage_bits == 16
    assert list(data.samples) == values
    assert read_header(payload)["bit_depth"] == 16


def test_png_16bit_rgb_preserves_samples() -> None:
    values = [0, 1, 2, 65535, 65534, 65533, 10, 20, 30, 40, 50, 60]
    payload = fx.rgb16_png(values, 2, 2)
    data = read(payload)
    assert data.mode == "RGB"
    assert data.storage_bits == 16
    assert data.row_samples(0) == tuple(values[:6])
    assert data.row_samples(1) == tuple(values[6:])


@pytest.mark.parametrize("depth", (1, 2, 4, 8))
def test_png_palette_depths(depth: int) -> None:
    count = min(1 << depth, 8)
    palette = make_palette(count)
    indices = [(x + y) % count for y in range(3) for x in range(5)]
    payload = fx.palette_png(indices, 5, 3, palette, depth=depth)
    data = read(payload)
    assert data.mode == "P"
    assert data.bit_depth == depth
    assert list(data.samples) == indices
    assert data.palette == palette


def test_png_palette_alpha_from_trns() -> None:
    palette = make_palette(4)
    payload = fx.palette_png(
        [0, 1, 2, 3], 4, 1, palette, depth=2, transparency=bytes((0, 255, 128))
    )
    data = read(payload)
    assert data.palette is not None
    assert [entry[3] for entry in data.palette.entries] == [0, 255, 128, 255]


def test_png_palette_depth_not_stored_as_8() -> None:
    # A 2-bit palette must not be reported as an 8-bit image.
    palette = make_palette(4)
    payload = fx.palette_png([0, 1, 2, 3], 4, 1, palette, depth=2)
    data = read(payload)
    assert data.bit_depth == 2
    assert data.storage_bits == 8


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            fx.truncated_png(fx.grayscale16_png([1, 2, 3, 4], 2, 2), 30), id="truncated"
        ),
        pytest.param(
            fx.png_with_bad_chunk_length(fx.grayscale16_png([1, 2, 3, 4], 2, 2)),
            id="bad-length",
        ),
        pytest.param(
            fx.png_with_trailing_bytes(fx.grayscale16_png([1, 2, 3, 4], 2, 2)),
            id="trailing",
        ),
        pytest.param(fx.png_without_palette([0, 1, 2, 3], 2, 2), id="no-palette"),
    ],
)
def test_png_errors_are_stable(payload: bytes) -> None:
    with pytest.raises(
        (TruncatedDataError, ChunkLengthError, TrailingDataError, IllegalPaletteError)
    ):
        read(payload)


def test_png_bad_crc() -> None:
    payload = bytearray(fx.grayscale16_png([1, 2, 3, 4], 2, 2))
    payload[30] ^= 0xFF
    with pytest.raises(ChecksumError):
        read(bytes(payload))


def test_png_truncated_row_reported_before_pixel_allocation() -> None:
    payload = fx.png_with_truncated_row([0] * 8, 2, 4)
    with pytest.raises(TruncatedDataError):
        read(payload)


def test_png_deep_chunk_length_bounds() -> None:
    # An IDAT declaring 2**31-1 bytes must fail on length, not attempt a read.
    payload = fx.png_with_bad_chunk_length(fx.grayscale16_png([1, 2, 3, 4], 2, 2))
    with pytest.raises(TruncatedDataError):
        read(payload)


def test_png_unknown_critical_chunk_rejected() -> None:
    png = fx.grayscale16_png([1, 2, 3, 4], 2, 2)
    header_end = 8 + 25
    payload = png[:header_end] + fx.png_chunk(b"ZZZZ", b"x") + png[header_end:]
    with pytest.raises(UnsupportedFeatureError):
        read(payload)


def test_png_unknown_ancillary_chunk_ignored() -> None:
    png = fx.grayscale16_png([1, 2, 3, 4], 2, 2)
    header_end = 8 + 25
    payload = png[:header_end] + fx.png_chunk(b"zzZz", b"x") + png[header_end:]
    data = read(payload)
    assert list(data.samples) == [1, 2, 3, 4]


def test_png_interlaced_matches_progressive() -> None:
    # An Adam7-encoded image must decode to the same samples as the plain one.
    width, height = 9, 7
    plain = bytearray()
    values = []
    for y in range(height):
        values.extend([(x * 7 + y * 11) % 256 for x in range(width)])
    for y in range(height):
        plain.append(0)
        plain += bytes(values[y * width : (y + 1) * width])
    (
        b"\x89PNG\r\n\x1a\n"
        + fx.png_ihdr(width, height, 8, 0)
        + fx.png_chunk(b"IDAT", zlib.compress(bytes(plain)))
        + fx.png_chunk(b"IEND", b"")
    )
    adam = [
        (0, 0, 8, 8),
        (4, 0, 8, 8),
        (0, 4, 4, 8),
        (2, 0, 4, 4),
        (0, 2, 2, 4),
        (1, 0, 2, 2),
        (0, 1, 1, 2),
    ]
    interlaced = bytearray()
    for x0, y0, dx, dy in adam:
        for y in range(y0, height, dy):
            xs = range(x0, width, dx)
            if not xs:
                continue
            interlaced.append(0)
            interlaced += bytes(values[y * width + x] for x in xs)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + fx.png_ihdr(width, height, 8, 0, interlace=1)
        + fx.png_chunk(b"IDAT", zlib.compress(bytes(interlaced)))
        + fx.png_chunk(b"IEND", b"")
    )
    assert list(read(payload).samples) == values


def test_png_16bit_tiff_cross_check_with_pillow() -> None:
    values = [0, 1000, 2000, 3000, 65535, 12345, 54321, 7]
    payload = fx.grayscale16_png(values, 4, 2)
    data = read(payload)
    with Image.open(io.BytesIO(payload)) as im:
        assert list(im.get_flattened_data()) == values
    assert list(data.samples) == values


# ------------------------------------------------------------------------- BMP


def test_bmp_565_bitfields() -> None:
    # 565: red 0xF800, green 0x07E0, blue 0x001F
    values = [0xF800, 0x07E0, 0x001F, 0xFFFF]
    payload = fx.bmp_rgb565(values, 2, 2)
    data = read(payload)
    assert data.mode == "RGB"
    assert data.row_samples(0) == (255, 0, 0, 0, 255, 0)
    assert data.row_samples(1) == (0, 0, 255, 255, 255, 255)


def test_bmp_bottom_up_flip() -> None:
    # The same logical rows, declared top-down and bottom-up, must decode to
    # identical pixels with the first row on top.
    values = [0xF800, 0x001F, 0x07E0, 0xFFFF]
    top_down = read(fx.bmp_rgb565(values, 2, 2, top_down=True))
    bottom_up = read(fx.bmp_rgb565(values, 2, 2))
    assert top_down.tobytes() == bottom_up.tobytes()
    assert top_down.row_samples(0) == (255, 0, 0, 0, 0, 255)
    assert top_down.row_samples(1) == (0, 255, 0, 255, 255, 255)


def test_bmp_palette_indices() -> None:
    palette = make_palette(4)
    payload = fx.bmp_palette8([0, 1, 2, 3, 3, 2, 1, 0], 4, 2, palette)
    data = read(payload)
    assert data.mode == "P"
    assert list(data.samples)[:4] == [0, 1, 2, 3]
    assert data.palette.entries[:4] == palette.entries


def test_bmp_index_outside_palette() -> None:
    palette = make_palette(2)
    payload = bytearray(fx.bmp_palette8([0, 1, 0, 1], 2, 2, palette))
    # Patch the pixel data to reference index 5, which is outside the table.
    payload[-4:] = bytes((5, 0, 0, 0))
    with pytest.raises(IllegalPaletteError):
        read(bytes(payload))


def test_bmp_truncated_data() -> None:
    payload = fx.bmp_rgb565([0xF800] * 16, 4, 4)
    with pytest.raises(TruncatedDataError):
        read(fx.bmp_truncated(payload, len(payload) - 10))


def test_bmp_palette_overlaps_pixels() -> None:
    payload = bytearray(fx.bmp_palette8([0, 1, 0, 1], 2, 2, make_palette(4)))
    # Claim a palette far larger than the gap before the pixel data.
    struct.pack_into("<I", payload, 46, 256)
    with pytest.raises((ChunkLengthError, TruncatedDataError, IllegalPaletteError)):
        read(bytes(payload))


def test_bmp_rle8_roundtrip() -> None:
    width, height = 4, 2
    palette = make_palette(4)
    pixels = [1] * width + [2] * width
    rle = bytearray()
    # BMP RLE streams store rows bottom first.
    for y in range(height - 1, -1, -1):
        rle += bytes((width, pixels[y * width]))
        rle += b"\x00\x00"  # end of line
    rle += b"\x00\x01"  # end of bitmap
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
        1,
        len(rle),
        2835,
        2835,
        len(palette),
        0,
    )
    data_offset = 14 + 40 + len(colors)
    payload = (
        b"BM"
        + struct.pack("<IHHI", data_offset + len(rle), 0, 0, data_offset)
        + dib
        + bytes(colors)
        + bytes(rle)
    )
    data = read(payload)
    assert data.mode == "P"
    assert list(data.samples) == pixels


def test_bmp_rle_absolute_mode() -> None:
    width, height = 4, 1
    palette = make_palette(4)
    pixels = [0, 1, 2, 3]
    rle = bytearray(b"\x00\x04") + bytes(pixels) + b"\x00\x01"
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
        1,
        len(rle),
        2835,
        2835,
        len(palette),
        0,
    )
    data_offset = 14 + 40 + len(colors)
    payload = (
        b"BM"
        + struct.pack("<IHHI", data_offset + len(rle), 0, 0, data_offset)
        + dib
        + bytes(colors)
        + bytes(rle)
    )
    assert list(read(payload).samples) == pixels


def test_bmp_rle_without_end_marker_is_truncated() -> None:
    width, height = 2, 1
    palette = make_palette(4)
    rle = bytearray(b"\x02\x01")
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
        1,
        len(rle),
        2835,
        2835,
        len(palette),
        0,
    )
    data_offset = 14 + 40 + len(colors)
    payload = (
        b"BM"
        + struct.pack("<IHHI", data_offset + len(rle), 0, 0, data_offset)
        + dib
        + bytes(colors)
        + bytes(rle)
    )
    with pytest.raises(TruncatedDataError):
        read(payload)


# ------------------------------------------------------------------------- TIFF


def _write_fixture_format(data: ImageData, format: str) -> bytes:
    from PIL.highdepth.encode import encode_png, encode_tiff

    out = io.BytesIO()
    if format == "PNG":
        encode_png([data], out)
    else:
        encode_tiff([data], out)
    return out.getvalue()


def test_tiff_16bit_gray_roundtrip() -> None:
    values = [0, 1, 40000, 65535]
    data = ImageData("TIFF", "L", 2, 2, 16, 16, 1, array("H", values))
    payload = _write_fixture_format(data, "TIFF")
    loaded = read(payload)
    assert loaded.storage_bits == 16
    assert list(loaded.samples) == values


def test_tiff_16bit_rgb_keeps_depth() -> None:
    values = array("H", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    data = ImageData("TIFF", "RGB", 2, 2, 16, 16, 3, values)
    payload = _write_fixture_format(data, "TIFF")
    loaded = read(payload)
    assert loaded.storage_bits == 16
    assert loaded.tobytes() == data.tobytes()


def test_tiff_big_endian() -> None:
    # A hand-built big-endian little image (2x1, 8-bit gray) with one strip.
    width, height = 2, 1
    bits = struct.pack(">H", 8)
    entries = [
        (256, 4, 1, struct.pack(">I", width)),
        (257, 4, 1, struct.pack(">I", height)),
        (258, 3, 1, bits + b"\x00\x00"),
        (259, 3, 1, struct.pack(">H", 1) + b"\x00\x00"),
        (262, 3, 1, struct.pack(">H", 1) + b"\x00\x00"),
        (273, 4, 1, struct.pack(">I", 8)),
        (277, 3, 1, struct.pack(">H", 1) + b"\x00\x00"),
        (279, 4, 1, struct.pack(">I", 2)),
    ]
    ifd = struct.pack(">H", len(entries))
    for tag, type_id, count, value in entries:
        ifd += struct.pack(">HHI", tag, type_id, count) + value
    ifd += struct.pack(">I", 0)
    # Header, pixel bytes at offset 8, then the IFD at offset 10.
    payload = b"MM\x00\x2a" + struct.pack(">I", 10) + bytes((10, 20)) + ifd
    data = read(payload)
    assert list(data.samples) == [10, 20]


def test_tiff_strip_offsets_outside_file() -> None:
    data = ImageData("TIFF", "L", 2, 2, 8, 8, 1, bytes((1, 2, 3, 4)))
    payload = bytearray(_write_fixture_format(data, "TIFF"))
    # Point the strip offset past the end of the file.
    ifd = struct.unpack_from("<I", payload, 4)[0]
    count = struct.unpack_from("<H", payload, ifd)[0]
    for index in range(count):
        base = ifd + 2 + 12 * index
        tag = struct.unpack_from("<H", payload, base)[0]
        if tag == 273:
            struct.pack_into("<I", payload, base + 8, len(payload) + 100)
    with pytest.raises(TruncatedDataError):
        read(bytes(payload))


def test_tiff_missing_strip_table() -> None:
    data = ImageData("TIFF", "L", 2, 2, 8, 8, 1, bytes((1, 2, 3, 4)))
    payload = bytearray(_write_fixture_format(data, "TIFF"))
    # Change the strip-offset tag number so the required table disappears.
    ifd = struct.unpack_from("<I", payload, 4)[0]
    count = struct.unpack_from("<H", payload, ifd)[0]
    for index in range(count):
        base = ifd + 2 + 12 * index
        tag = struct.unpack_from("<H", payload, base)[0]
        if tag == 273:
            struct.pack_into("<H", payload, base, 999)
            break
    with pytest.raises(FormatError):
        read(bytes(payload))


def test_tiff_page_count_and_frames() -> None:
    frames = [
        ImageData("TIFF", "L", 3, 2, 8, 8, 1, bytes((v, v, v, v, v, v))) for v in (7, 9)
    ]
    out = io.BytesIO()
    from PIL.highdepth.encode import encode_tiff

    encode_tiff(frames, out)
    payload = out.getvalue()
    from PIL.highdepth import read_frames

    loaded = read_frames(payload)
    assert len(loaded) == 2
    assert [frame.row_samples(0)[0] for frame in loaded] == [7, 9]


def test_tiff_pillow_cross_check() -> None:
    im = make_rgb()
    payload = save(im, "TIFF")
    data = read(payload)
    assert data.tobytes() == im.tobytes()


def test_detect_unknown_stream() -> None:
    with pytest.raises(FormatError):
        read(b"not an image at all")
