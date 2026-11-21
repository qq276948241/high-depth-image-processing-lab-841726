"""Encoders, multi-frame round trips, atomic output and cancellation."""

from __future__ import annotations

import io
import os
import struct
from array import array

import pytest

from PIL import Image
from PIL.highdepth import (
    CancellationToken,
    MemoryBudget,
    UnsupportedFeatureError,
    WriteVerificationError,
    encode,
    read,
    read_frames,
)
from PIL.highdepth.budget import BudgetExceededError
from PIL.highdepth.model import ImageData, Palette

from .highdepth_fixtures.helpers import make_palette, make_rgb, tiff_tag_values


def _gray(v: int, size: tuple[int, int] = (4, 3)) -> ImageData:
    return ImageData.from_pil(Image.new("L", size, v))


def _rgb() -> ImageData:
    return ImageData.from_pil(make_rgb())


# ---------------------------------------------------------------- single frame


@pytest.mark.parametrize("format", ("PNG", "TIFF"))
def test_single_frame_roundtrip(format: str) -> None:
    data = _rgb()
    out = io.BytesIO()
    encode.write_frames(None, [data], format, fp=out)
    loaded = read(out.getvalue(), format=format)
    assert loaded.tobytes() == data.tobytes()
    assert loaded.mode == data.mode


def test_png_16bit_roundtrip() -> None:
    values = array("H", [0, 1, 300, 4000, 50000, 65535])
    data = ImageData("PNG", "L", 3, 2, 16, 16, 1, values)
    out = io.BytesIO()
    encode.write_frames(None, [data], "PNG", fp=out)
    loaded = read(out.getvalue())
    assert loaded.storage_bits == 16
    assert loaded.tobytes() == data.tobytes()


def test_png_16bit_rgb_roundtrip() -> None:
    values = array("H", list(range(12)))
    data = ImageData("PNG", "RGB", 2, 2, 16, 16, 3, values)
    out = io.BytesIO()
    encode.write_frames(None, [data], "PNG", fp=out)
    loaded = read(out.getvalue())
    assert loaded.storage_bits == 16
    assert loaded.tobytes() == data.tobytes()


def test_png_palette_transparency_roundtrip() -> None:
    palette = make_palette(4, with_alpha=True)
    data = ImageData("PNG", "P", 2, 2, 2, 8, 1, bytes((0, 1, 2, 3)), palette=palette)
    out = io.BytesIO()
    encode.write_frames(None, [data], "PNG", fp=out)
    loaded = read(out.getvalue())
    assert loaded.palette is not None
    assert [entry[3] for entry in loaded.palette.entries] == [0, 255, 255, 255]


def test_bmp_rgb_and_rgba_roundtrip() -> None:
    for mode in ("RGB", "RGBA"):
        im = Image.new(mode, (5, 3))
        im.putdata(
            [
                tuple(
                    (x * 40 + y, y * 70, x * 10 + 5)[: len(mode)],
                )
                + ((200,) if mode == "RGBA" else ())
                for y in range(3)
                for x in range(5)
            ]
        )
        data = ImageData.from_pil(im)
        out = io.BytesIO()
        encode.write_frames(None, [data], "BMP", fp=out)
        loaded = read(out.getvalue())
        assert loaded.mode == mode
        assert loaded.tobytes() == data.tobytes()


def test_bmp_palette_roundtrip() -> None:
    palette = make_palette(4)
    data = ImageData(
        "BMP", "P", 4, 2, 2, 8, 1, bytes((0, 1, 2, 3, 3, 2, 1, 0)), palette=palette
    )
    out = io.BytesIO()
    encode.write_frames(None, [data], "BMP", fp=out)
    loaded = read(out.getvalue())
    assert loaded.mode == "P"
    assert list(loaded.samples) == [0, 1, 2, 3, 3, 2, 1, 0]
    assert loaded.palette.entries[:4] == palette.entries


def test_bmp_rejects_multiple_frames() -> None:
    with pytest.raises(UnsupportedFeatureError):
        encode.write_frames(None, [_gray(1), _gray(2)], "BMP", fp=io.BytesIO())


def test_empty_frame_rejected() -> None:
    empty = ImageData("PNG", "L", 0, 0, 8, 8, 1, b"")
    with pytest.raises(Exception):
        encode.write_frames(None, [empty], "PNG", fp=io.BytesIO())


# ---------------------------------------------------------------- multi frame


@pytest.mark.parametrize("format", ("TIFF", "PNG"))
def test_multiframe_preserves_order(format: str) -> None:
    frames = [_gray(value) for value in (10, 30, 20, 200)]
    out = io.BytesIO()
    encode.write_frames(None, frames, format, fp=out)
    loaded = read_frames(out.getvalue())
    assert len(loaded) == len(frames)
    assert [frame.row_samples(0)[0] for frame in loaded] == [10, 30, 20, 200]
    assert [frame.sha256() for frame in loaded] == [frame.sha256() for frame in frames]


def test_multiframe_tiff_metadata_orientation() -> None:
    frame = _gray(5)
    rotated = ImageData(
        frame.format,
        frame.mode,
        frame.width,
        frame.height,
        frame.bit_depth,
        frame.storage_bits,
        frame.channels,
        frame.samples,
        orientation=6,
    )
    out = io.BytesIO()
    encode.write_frames(None, [rotated, _gray(6)], "TIFF", fp=out)
    payload = out.getvalue()
    assert tiff_tag_values(payload, 0)[274] == 6


def test_multiframe_png_rejects_mixed_modes() -> None:
    gray = _gray(4, (6, 4))
    rgb = _rgb()
    assert gray.width == rgb.width
    assert gray.height == rgb.height
    with pytest.raises(UnsupportedFeatureError):
        encode.write_frames(None, [gray, rgb], "PNG", fp=io.BytesIO())


def test_multiframe_size_mismatch() -> None:
    from PIL.highdepth.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        encode.write_frames(
            None, [_gray(1, (2, 2)), _gray(2, (3, 3))], "TIFF", fp=io.BytesIO()
        )


# ---------------------------------------------------------------- atomicity


def test_atomic_write_creates_file(tmp_path: object) -> None:
    from pathlib import Path

    directory = Path(str(tmp_path))
    target = directory / "image.png"
    encode.write_frames(target, [_rgb()], "PNG")
    assert target.exists()
    payload = target.read_bytes()
    assert read(payload).tobytes() == _rgb().tobytes()


def test_failed_write_leaves_no_partial_file(tmp_path: object) -> None:
    from pathlib import Path

    directory = Path(str(tmp_path))
    target = directory / "image.png"
    token = CancellationToken()
    token.cancel()
    with pytest.raises(Exception):
        encode.write_frames(target, [_rgb()], "PNG", token=token)
    assert not target.exists()
    assert not [name for name in os.listdir(directory) if name.startswith(".highdepth")]


def test_cancellation_stops_after_first_block() -> None:
    data = ImageData("PNG", "L", 64, 128, 8, 8, 1, bytes(64 * 128))
    token = CancellationToken()

    class CancelAfterOneWrite(io.BytesIO):
        writes = 0

        def write(self, payload: bytes) -> int:  # type: ignore[override]
            self.writes += 1
            if self.writes == 3:
                token.cancel()
            return super().write(payload)

    out = CancelAfterOneWrite()
    with pytest.raises(Exception):
        encode.write_frames(None, [data], "PNG", fp=out, token=token)
    # Once cancelled, no further block was written.
    before = out.writes
    with pytest.raises(Exception):
        encode.write_frames(None, [data], "PNG", fp=out, token=token)
    assert out.writes == before


def test_budget_rejection_before_large_allocation() -> None:
    data = ImageData("PNG", "RGB", 200, 200, 8, 8, 3, bytes(200 * 200 * 3))
    with pytest.raises(BudgetExceededError):
        encode.write_frames(
            None, [data], "PNG", fp=io.BytesIO(), budget=MemoryBudget(1024)
        )


def test_verify_roundtrip_helper(tmp_path: object) -> None:
    from pathlib import Path

    target = Path(str(tmp_path)) / "check.tiff"
    frames = [_gray(1), _gray(2)]
    encode.write_frames(target, frames, "TIFF")
    encode.verify_roundtrip(target, frames, "TIFF")


def test_verify_roundtrip_detects_mismatch(tmp_path: object) -> None:
    from pathlib import Path

    target = Path(str(tmp_path)) / "check.png"
    encode.write_frames(target, [_gray(1)], "PNG")
    with pytest.raises(WriteVerificationError):
        encode.verify_roundtrip(target, [_gray(2)], "PNG")


def test_tiff_deflate_roundtrip() -> None:
    data = _rgb()
    out = io.BytesIO()
    encode.write_frames(None, [data], "TIFF", fp=out, compression="deflate")
    assert read(out.getvalue()).tobytes() == data.tobytes()


def test_tiff_inline_and_out_of_line_strip_tables() -> None:
    small = ImageData("TIFF", "L", 2, 2, 8, 8, 1, bytes((1, 2, 3, 4)))
    large = ImageData("TIFF", "L", 4, 9000, 8, 8, 1, bytes([7]) * 4 * 9000)
    for data in (small, large):
        out = io.BytesIO()
        encode.write_frames(None, [data], "TIFF", fp=out)
        assert read(out.getvalue()).tobytes() == data.tobytes()


def test_png_dpi_and_icc_metadata() -> None:
    data = _rgb()
    icc = b"\x00" * 200
    out = io.BytesIO()
    encode.write_frames(None, [data], "PNG", fp=out, dpi=(300, 300), icc_profile=icc)
    payload = out.getvalue()
    assert b"pHYs" in payload
    assert b"iCCP" in payload
    assert read(payload).tobytes() == data.tobytes()


def test_bmp_header_fields() -> None:
    data = _rgb()
    out = io.BytesIO()
    encode.write_frames(None, [data], "BMP", fp=out)
    payload = out.getvalue()
    assert payload[:2] == b"BM"
    (file_size,) = struct.unpack_from("<I", payload, 2)
    assert file_size == len(payload)
    (offset,) = struct.unpack_from("<I", payload, 10)
    assert 14 < offset < len(payload)


@pytest.mark.parametrize(
    "orientation, size, expected",
    (
        (1, (3, 2), [1, 2, 3, 4, 5, 6]),
        (2, (3, 2), [3, 2, 1, 6, 5, 4]),
        (3, (3, 2), [6, 5, 4, 3, 2, 1]),
        (4, (3, 2), [4, 5, 6, 1, 2, 3]),
        (5, (2, 3), [1, 4, 2, 5, 3, 6]),
        (6, (2, 3), [4, 1, 5, 2, 6, 3]),
        (7, (2, 3), [6, 3, 5, 2, 4, 1]),
        (8, (2, 3), [3, 6, 2, 5, 1, 4]),
    ),
)
def test_orientation_roundtrip_and_normalization(
    orientation: int, size: tuple[int, int], expected: list[int]
) -> None:
    base = ImageData("TIFF", "L", 3, 2, 8, 8, 1, bytes((1, 2, 3, 4, 5, 6)))
    tagged = ImageData(
        "TIFF",
        "L",
        3,
        2,
        8,
        8,
        1,
        base.samples,
        orientation=orientation,
    )
    out = io.BytesIO()
    encode.write_frames(None, [tagged], "TIFF", fp=out)
    loaded = read(out.getvalue())
    # The tag survives the write-back and the samples are not rotated twice.
    assert loaded.orientation == orientation
    assert loaded.tobytes() == tagged.tobytes()
    normalized = loaded.normalize_orientation()
    assert normalized.orientation == 1
    assert (normalized.width, normalized.height) == size
    assert list(normalized.samples) == expected
    # Idempotent: normalizing a normalized image changes nothing.
    assert normalized.normalize_orientation() is normalized


@pytest.mark.parametrize(
    "mode, channels", (("L", 1), ("LA", 2), ("RGB", 3), ("RGBA", 4))
)
@pytest.mark.parametrize("bits", (8, 16))
def test_roundtrip_matrix_tiff(mode: str, channels: int, bits: int) -> None:
    width, height = 4, 3
    if bits == 8:
        samples: bytes | array = bytes(range(width * height * channels))
    else:
        samples = array(
            "H", [(i * 977) % 65536 for i in range(width * height * channels)]
        )
    data = ImageData("", mode, width, height, bits, bits, channels, samples)
    out = io.BytesIO()
    encode.write_frames(None, [data], "TIFF", fp=out)
    loaded = read(out.getvalue())
    assert (loaded.mode, loaded.storage_bits) == (mode, bits)
    assert loaded.tobytes() == data.tobytes()


@pytest.mark.parametrize(
    "mode, channels", (("L", 1), ("LA", 2), ("RGB", 3), ("RGBA", 4))
)
@pytest.mark.parametrize("bits", (8, 16))
def test_roundtrip_matrix_png(mode: str, channels: int, bits: int) -> None:
    width, height = 4, 3
    if bits == 8:
        samples: bytes | array = bytes(range(width * height * channels))
    else:
        samples = array(
            "H", [(i * 977) % 65536 for i in range(width * height * channels)]
        )
    data = ImageData("", mode, width, height, bits, bits, channels, samples)
    out = io.BytesIO()
    encode.write_frames(None, [data], "PNG", fp=out)
    loaded = read(out.getvalue())
    assert (loaded.mode, loaded.storage_bits) == (mode, bits)
    assert loaded.tobytes() == data.tobytes()


@pytest.mark.parametrize("bits", (1, 2, 4, 8))
@pytest.mark.parametrize("format", ("PNG", "TIFF"))
def test_palette_roundtrip_at_every_depth(bits: int, format: str) -> None:
    entries = min(8, 1 << bits)
    palette = Palette(
        tuple(
            (i * 9 % 256, i * 17 % 256, i * 31 % 256, 0 if i == 0 else 255)
            for i in range(entries)
        )
    )
    indices = bytes((i * 7) % entries for i in range(5 * 4))
    data = ImageData("", "P", 5, 4, bits, 8, 1, indices, palette=palette)
    out = io.BytesIO()
    encode.write_frames(None, [data], format, fp=out)
    loaded = read(out.getvalue())
    if format == "TIFF":
        # TIFF has no palette alpha; the frame is promoted to RGBA losslessly.
        assert loaded.mode == "RGBA"
        assert loaded.tobytes() == data.to_rgba().tobytes()
    else:
        assert loaded.mode == "P"
        assert loaded.samples == indices
        assert loaded.palette == palette


def test_bmp_rejects_gray_modes() -> None:
    gray = ImageData("", "L", 2, 2, 8, 8, 1, bytes((1, 2, 3, 4)))
    la = ImageData("", "LA", 2, 1, 8, 8, 2, bytes((10, 0, 200, 255)))
    for data in (gray, la):
        with pytest.raises(UnsupportedFeatureError):
            encode.write_frames(None, [data], "BMP", fp=io.BytesIO())


def test_tiff_palette_alpha_promotes_to_rgba() -> None:
    palette = Palette(((10, 20, 30, 0), (40, 50, 60, 255)))
    data = ImageData("", "P", 2, 2, 8, 8, 1, bytes((0, 1, 0, 1)), palette=palette)
    out = io.BytesIO()
    encode.write_frames(None, [data], "TIFF", fp=out)
    loaded = read(out.getvalue())
    assert loaded.mode == "RGBA"
    assert loaded.samples == data.to_rgba().samples
