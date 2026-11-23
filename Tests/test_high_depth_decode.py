from __future__ import annotations

import io
import struct
import zlib
from typing import Any

import pytest

from PIL import HighDepth, Image, ImageFile

TYPE_CHECKING = False
if TYPE_CHECKING:
    from pathlib import Path


def chunk(cid: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + cid
        + data
        + struct.pack(">I", zlib.crc32(cid + data) & 0xFFFFFFFF)
    )


MAGIC = b"\x89PNG\r\n\x1a\n"


def png(
    width: int,
    height: int,
    bit_depth: int,
    color_type: int,
    raw: bytes,
    *,
    extra: bytes = b"",
    interlace: int = 0,
) -> bytes:
    header = chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, interlace),
    )
    return (
        MAGIC
        + header
        + extra
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def rgb16(pixels: list[tuple[int, int, int]], width: int) -> bytes:
    height = len(pixels) // width
    rows = b""
    for y in range(height):
        rows += b"\x00"
        for pixel in pixels[y * width : (y + 1) * width]:
            rows += struct.pack(">3H", *pixel)
    return png(width, height, 16, 2, rows)


def gray16(pixels: list[int], width: int) -> bytes:
    height = len(pixels) // width
    rows = b""
    for y in range(height):
        rows += b"\x00"
        for value in pixels[y * width : (y + 1) * width]:
            rows += struct.pack(">H", value)
    return png(width, height, 16, 0, rows)


def sources(data: bytes, tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / "high-depth.png"
    path.write_bytes(data)
    return {
        "bytes": data,
        "path": path,
        "file": io.BytesIO(data),
        "chunks": [data[i : i + 7] for i in range(0, len(data), 7)],
        "bytearray": bytearray(data),
    }


def test_input_methods_agree(tmp_path: Path) -> None:
    # Arrange
    data = rgb16(
        [(0x1234, 0x5678, 0x9ABC), (0xFFFF, 0x8000, 0x0001), (1, 2, 3), (4, 5, 6)],
        2,
    )
    expected = HighDepth.decode(data)

    # Act / Assert
    for name, source in sources(data, tmp_path).items():
        image = HighDepth.decode(source)
        assert image.info == expected.info, name
        assert image.planes == expected.planes, name
        assert image.image.tobytes() == expected.image.tobytes(), name


def test_16_bit_rgb_samples() -> None:
    # Arrange
    data = rgb16([(0x1234, 0x5678, 0x9ABC), (0xFFFF, 0x8000, 0x0001)], 2)

    # Act
    image = HighDepth.decode(data)

    # Assert
    assert image.info.mode == "RGB"
    assert image.info.bit_depth == 16
    assert [
        tuple(image.sample(x, 0, channel) for channel in range(3)) for x in range(2)
    ] == [
        (0x1234, 0x5678, 0x9ABC),
        (0xFFFF, 0x8000, 0x0001),
    ]


def test_16_bit_grayscale_is_i16() -> None:
    # Arrange
    data = gray16([0, 1, 0x7FFF, 0xFFFF], 2)

    # Act
    image = HighDepth.decode(data)

    # Assert
    assert image.info.mode == "I;16"
    assert image.image.getpixel((1, 1)) == 0xFFFF
    assert [image.sample(x, 0) for x in range(2)] == [0, 1]


@pytest.mark.parametrize(
    "bit_depth,color_type,raw,expected",
    (
        (1, 0, b"\x00\x00", "1"),
        (2, 0, b"\x00\x1b", "L"),
        (4, 0, b"\x00\x12", "L"),
        (8, 0, b"\x00\x42\x43", "L"),
        (16, 0, b"\x00\x12\x34\x56\x78", "I;16"),
    ),
)
def test_gray_bit_depths(
    bit_depth: int, color_type: int, raw: bytes, expected: str
) -> None:
    # Arrange
    data = png(2, 1, bit_depth, color_type, raw)

    # Act
    image = HighDepth.decode(data)

    # Assert
    assert image.info.mode == expected
    assert image.info.bit_depth == bit_depth


def test_sub_byte_samples_are_not_scaled() -> None:
    # Arrange
    data = png(4, 1, 2, 0, b"\x00\x1b")

    # Act
    image = HighDepth.decode(data)

    # Assert
    assert [image.sample(x, 0) for x in range(4)] == [0, 1, 2, 3]
    rescaled = image.rescale(8)
    assert [rescaled.sample(x, 0) for x in range(4)] == [0, 85, 170, 255]


def test_palette_and_transparency() -> None:
    # Arrange
    palette = chunk(b"PLTE", bytes((255, 0, 0, 0, 255, 0, 0, 0, 255)))
    data = png(3, 1, 8, 3, b"\x00\x00\x01\x02", extra=palette)

    # Act
    image = HighDepth.decode(data)

    # Assert
    assert image.info.mode == "P"
    assert image.info.palette == bytes((255, 0, 0, 0, 255, 0, 0, 0, 255))
    assert image.info.palette_size == 3
    assert image.info.transparency is None


def test_palette_transparency_index() -> None:
    # Arrange
    palette = chunk(b"PLTE", bytes((255, 0, 0, 0, 255, 0)))
    transparency = chunk(b"tRNS", b"\x80")
    data = png(2, 1, 8, 3, b"\x00\x00\x01", extra=palette + transparency)

    # Act
    image = HighDepth.decode(data)

    # Assert
    assert image.info.transparency == b"\x80"
    rgba = image.image.convert("RGBA")
    assert rgba.getpixel((0, 0)) == (255, 0, 0, 128)
    assert rgba.getpixel((1, 0)) == (0, 255, 0, 255)


def test_16_bit_rgb_transparency() -> None:
    # Arrange
    transparent = (0x1234, 0x5678, 0x9ABC)
    data = rgb16([transparent, (0xFFFF, 0x8000, 0x0001)], 2)
    trns = chunk(b"tRNS", struct.pack(">3H", *transparent))
    index = data.index(b"IDAT") - 4
    data = data[:index] + trns + data[index:]

    # Act
    image = HighDepth.decode(data)

    # Assert
    rgba = image.image.convert("RGBA")
    assert rgba.getpixel((0, 0)) == (0x12, 0x56, 0x9A, 0)
    assert rgba.getpixel((1, 0))[3] == 255


def test_16_bit_grayscale_transparency() -> None:
    # Arrange
    data = gray16([0x1234, 0xABCD], 2)
    trns = chunk(b"tRNS", struct.pack(">H", 0x1234))
    index = data.index(b"IDAT") - 4
    data = data[:index] + trns + data[index:]

    # Act
    image = HighDepth.decode(data)

    # Assert
    assert image.info.transparency == struct.pack(">H", 0x1234)
    rgba = image.image.convert("RGBA")
    assert rgba.getpixel((0, 0))[3] == 0
    assert rgba.getpixel((1, 0))[3] == 255


def test_row_stride_and_orientation() -> None:
    # Arrange: distinct rows so a stride mistake swaps them.
    data = rgb16(
        [
            (0x0100, 0x0100, 0x0100),
            (0x0200, 0x0200, 0x0200),
            (0x0300, 0x0300, 0x0300),
            (0x0400, 0x0400, 0x0400),
            (0x0500, 0x0500, 0x0500),
            (0x0600, 0x0600, 0x0600),
        ],
        3,
    )

    # Act
    image = HighDepth.decode(data)

    # Assert
    assert image.info.file_row_order == "top-down"
    assert image.image.getpixel((0, 0)) == (1, 1, 1)
    assert [image.sample(x, 1, 0) >> 8 for x in range(3)] == [4, 5, 6]


def test_adam7_matches_core_decoder() -> None:
    # Arrange
    for name in ("Tests/images/pil123rgba.png", "Tests/images/pil123p.png"):
        data = open(name, "rb").read()

        # Act
        decoded = HighDepth.decode(data)
        reference = Image.open(io.BytesIO(data))
        reference.load()

        # Assert
        assert decoded.info.interlace
        assert decoded.image.tobytes() == reference.tobytes()


def test_animation_is_rejected() -> None:
    # Arrange
    actl = chunk(b"acTL", struct.pack(">II", 2, 0))
    data = png(1, 1, 8, 2, b"\x00\x01\x02\x03", extra=actl)

    # Act / Assert
    with pytest.raises(HighDepth.DecodeError, match="animated PNG"):
        HighDepth.decode(data)


def test_multipage_tiff_frame_consistency(tmp_path: Path) -> None:
    # Arrange
    name = "Tests/images/multipage.tiff"
    data = open(name, "rb").read()

    # Act / Assert
    probe = HighDepth.probe(name)
    assert probe.frame_count == 3
    for frame in range(probe.frame_count):
        expected = HighDepth.decode(name, frame=frame)
        for source in (
            data,
            io.BytesIO(data),
            [data[i : i + 13] for i in range(0, len(data), 13)],
        ):
            decoded = HighDepth.decode(source, frame=frame)
            assert decoded.info == expected.info
            assert decoded.image.tobytes() == expected.image.tobytes()
        with Image.open(io.BytesIO(data)) as reference:
            reference.seek(frame)
            reference.load()
            assert expected.image.tobytes() == reference.tobytes()


def test_frame_out_of_range() -> None:
    # Arrange
    name = "Tests/images/multipage.tiff"

    # Act / Assert
    with pytest.raises(IndexError):
        HighDepth.decode(name, frame=99)


def test_bmp_bottom_up_and_top_down() -> None:
    # Arrange: two 24-bit rows, file order bottom-up.
    rows = bytes((3, 2, 1)) * 2 + bytes((6, 5, 4)) * 2

    # Act
    image = HighDepth.decode(_bmp(2, 2, 24, rows))

    # Assert
    assert image.info.file_row_order == "bottom-up"
    assert image.image.getpixel((0, 0)) == (4, 5, 6)
    top_down = HighDepth.decode(_bmp(2, 2, 24, rows, top_down=True))
    assert top_down.info.file_row_order == "top-down"
    assert top_down.image.getpixel((0, 0)) == (1, 2, 3)


def test_bmp_palette() -> None:
    # Arrange
    palette = bytes((0, 0, 255, 0, 0, 255, 0, 0))  # BGRX red, green
    data = _bmp(2, 1, 8, b"\x00\x01", palette=palette, colors=2)

    # Act
    image = HighDepth.decode(data)

    # Assert
    assert image.info.palette == bytes((255, 0, 0, 255, 0, 255, 0, 255))
    assert image.image.getpixel((0, 0)) == 0
    assert image.image.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def _bmp(
    width: int,
    height: int,
    bits: int,
    pixels: bytes,
    *,
    palette: bytes = b"",
    colors: int = 0,
    top_down: bool = False,
    header_size: int = 40,
) -> bytes:
    stride = ((width * bits + 31) >> 3) & ~3
    padded = b"".join(
        pixels[y * ((width * bits + 7) // 8) : (y + 1) * ((width * bits + 7) // 8)]
        + b"\0" * (stride - (width * bits + 7) // 8)
        for y in range(height)
    )
    file_size = 14 + header_size + len(palette) + len(padded)
    signed_height = -height if top_down else height
    return (
        b"BM"
        + struct.pack("<IHHI", file_size, 0, 0, 14 + header_size + len(palette))
        + struct.pack(
            "<IiiHHIIiiII",
            header_size,
            width,
            signed_height,
            1,
            bits,
            0,
            len(padded),
            2835,
            2835,
            colors,
            0,
        )
        + palette
        + padded
    )


def test_truncated_row_is_rejected_before_decode() -> None:
    # Arrange
    data = rgb16([(1, 2, 3), (4, 5, 6)], 2)
    truncated = bytearray(data)
    # Drop the final byte of the last decompressed row by recompressing a short
    # payload with a valid CRC.
    idat_index = truncated.index(b"IDAT") - 4
    raw = zlib.compress(b"\x00" + struct.pack(">6H", 1, 2, 3, 4, 5, 7))[:-2]
    tampered = bytes(truncated[:idat_index]) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")

    # Act / Assert
    with pytest.raises(HighDepth.DecodeError, match="broken PNG compressed data"):
        HighDepth.decode(tampered)


def test_short_decompressed_data_is_rejected() -> None:
    # Arrange: only one row of two is present.
    raw = b"\x00" + struct.pack(">3H", 1, 2, 3)
    data = png(1, 2, 16, 2, raw)

    # Act / Assert
    with pytest.raises(HighDepth.DecodeError, match="truncated PNG image data"):
        HighDepth.decode(data)


def test_contradictory_chunk_length() -> None:
    # Arrange
    header = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    lying = struct.pack(">I", 1000) + b"tEXt" + b"abc"
    data = MAGIC + header + lying + chunk(b"IDAT", zlib.compress(b"\x00\x01\x02\x03"))

    # Act / Assert
    with pytest.raises(HighDepth.DecodeError, match="truncated"):
        HighDepth.decode(data)


def test_bad_chunk_crc() -> None:
    # Arrange
    data = bytearray(png(1, 1, 8, 2, b"\x00\x01\x02\x03"))
    # Corrupt a byte inside the IDAT payload, leaving the chunk framing intact.
    index = data.index(b"IDAT") + 4
    data[index] ^= 0xFF

    # Act / Assert
    with pytest.raises(HighDepth.DecodeError, match="checksum"):
        HighDepth.decode(bytes(data))


def test_trailing_bytes_after_iend() -> None:
    # Arrange
    data = png(1, 1, 8, 2, b"\x00\x01\x02\x03") + b"junk"

    # Act / Assert
    with pytest.raises(HighDepth.DecodeError, match="trailing bytes"):
        HighDepth.decode(data)


def test_extra_decompressed_rows() -> None:
    # Arrange
    data = png(1, 1, 8, 2, b"\x00\x01\x02\x03\x00\x04\x05\x06")

    # Act / Assert
    with pytest.raises(HighDepth.DecodeError, match="extra bytes"):
        HighDepth.decode(data)


@pytest.mark.parametrize(
    "palette,message",
    (
        (bytes((255, 0)), "invalid palette size"),
        (bytes(value % 256 for value in range(3 * 257)), "invalid palette size"),
    ),
)
def test_invalid_palette(palette: bytes, message: str) -> None:
    # Arrange
    extra = chunk(b"PLTE", palette)
    data = png(1, 1, 8, 3, b"\x00\x00", extra=extra)

    # Act / Assert
    with pytest.raises(HighDepth.DecodeError, match=message):
        HighDepth.decode(data)


def test_palette_smaller_than_transparency() -> None:
    # Arrange
    extra = chunk(b"PLTE", bytes((255, 0, 0))) + chunk(b"tRNS", b"\x80\x80")
    data = png(1, 1, 8, 3, b"\x00\x00", extra=extra)

    # Act / Assert
    with pytest.raises(HighDepth.DecodeError, match="tRNS chunk has 2 entries"):
        HighDepth.decode(data)


@pytest.mark.parametrize(
    "bit_depth,color_type",
    ((3, 0), (16, 3), (8, 5), (4, 2)),
)
def test_invalid_bit_depth_combination(bit_depth: int, color_type: int) -> None:
    # Arrange
    header = chunk(
        b"IHDR", struct.pack(">IIBBBBB", 1, 1, bit_depth, color_type, 0, 0, 0)
    )
    data = MAGIC + header + chunk(b"IDAT", zlib.compress(b"\x00")) + chunk(b"IEND", b"")

    # Act / Assert
    with pytest.raises(HighDepth.DecodeError, match="invalid PNG bit depth"):
        HighDepth.decode(data)


def test_size_overflow_is_rejected_before_allocation() -> None:
    # Arrange
    header = chunk(
        b"IHDR", struct.pack(">IIBBBBB", 0x7FFFFFFF, 0x7FFFFFFF, 16, 6, 0, 0, 0)
    )
    data = MAGIC + header + chunk(b"IDAT", zlib.compress(b"")) + chunk(b"IEND", b"")

    # Act / Assert
    with pytest.raises(HighDepth.DecodeError, match="overflows"):
        HighDepth.decode(data, budget=HighDepth.MemoryBudget(1 << 20))
    with pytest.raises(HighDepth.DecodeError, match="overflows"):
        HighDepth.MemoryBudget().size((0x7FFFFFFF, 2), channels=4, bytes_per_sample=2)


def test_budget_is_checked_before_pixels() -> None:
    # Arrange
    data = rgb16([(1, 2, 3)] * 4, 2)

    # Act / Assert
    with pytest.raises(MemoryError):
        HighDepth.decode(data, budget=HighDepth.MemoryBudget(2))
    assert HighDepth.decode(data, budget=HighDepth.MemoryBudget(1024)).info.size == (
        2,
        2,
    )


def test_probe_does_not_decode() -> None:
    # Arrange
    data = rgb16([(1, 2, 3), (4, 5, 6)], 2)

    # Act
    info = HighDepth.probe(data)

    # Assert
    assert info.format == "PNG"
    assert info.size == (2, 1)
    assert info.channels == 3


def test_rescale_roundtrip() -> None:
    # Arrange
    data = gray16([0, 0x1234, 0x8000, 0xFFFF], 4)
    image = HighDepth.decode(data)

    # Act
    down = image.rescale(8)
    up = down.rescale(16)

    # Assert
    assert down.info.bit_depth == 8
    assert [down.sample(x, 0) for x in range(4)] == [0, 18, 128, 255]
    assert [up.sample(x, 0) for x in range(4)] == [0, 4626, 32896, 65535]


def test_sample_to_unit_and_back() -> None:
    # Arrange
    values = (0, 1, 0x1234, 0x7FFF, 0xFFFF)

    # Act / Assert
    for value in values:
        unit = HighDepth.sample_to_unit(value, 16)
        assert HighDepth.unit_to_sample(unit, 16) == value
    assert HighDepth.unit_to_sample(float("nan"), 8) == 0
    assert HighDepth.unit_to_sample(float("inf"), 8) == 255
    assert HighDepth.unit_to_sample(float("-inf"), 8) == 0
    assert HighDepth.unit_to_sample(2.0, 8) == 255
    assert HighDepth.unit_to_sample(-1.0, 8) == 0


def test_imagefile_parser_matches_decode() -> None:
    # Arrange
    data = open("Tests/images/pil123rgba.png", "rb").read()
    expected = HighDepth.decode(data)

    # Act
    parser = ImageFile.Parser()
    for index in range(0, len(data), 37):
        parser.feed(data[index : index + 37])
    parsed = parser.close()

    # Assert
    assert parsed.tobytes() == expected.image.tobytes()
    assert ImageFile.decode_high_depth(data).tobytes() == parsed.tobytes()


def test_stream_reader_matches_decode() -> None:
    # Arrange
    data = rgb16([(1, 2, 3), (4, 5, 6)], 2)
    expected = HighDepth.decode(data)

    # Act
    reader = HighDepth.StreamReader()
    for index in range(0, len(data), 5):
        reader.feed(data[index : index + 5])
    result = reader.close()

    # Assert
    assert result.info == expected.info
    assert result.planes == expected.planes
    assert result.image.tobytes() == expected.image.tobytes()


def test_stream_reader_budget() -> None:
    # Arrange
    data = rgb16([(1, 2, 3)] * 4, 2)
    reader = HighDepth.StreamReader(HighDepth.MemoryBudget(1))
    reader.feed(data)

    # Act / Assert
    with pytest.raises(MemoryError):
        reader.close()


def test_decode_does_not_modify_source() -> None:
    # Arrange
    data = rgb16([(1, 2, 3), (4, 5, 6)], 2)
    original = bytes(data)

    # Act
    HighDepth.decode(data)
    HighDepth.probe(data)

    # Assert
    assert data == original


def test_public_image_entry_points() -> None:
    # Arrange
    data = rgb16([(1, 2, 3), (4, 5, 6)], 2)
    with Image.open(io.BytesIO(data)) as image:
        image.load()

        # Act
        quantized = Image.quantize_high_depth(image, 2)

    # Assert
    assert quantized.mode == "P"
    with Image.open(io.BytesIO(data)) as image:
        reloaded = Image.open_high_depth(io.BytesIO(data))
        assert reloaded.mode == image.mode
        assert reloaded.tobytes() == image.tobytes()
