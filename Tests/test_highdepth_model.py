"""Unit tests for the normalized model, budgets and byte sources."""

from __future__ import annotations

import io
from array import array

import pytest

from PIL import highdepth as hd
from PIL.highdepth.errors import (
    BitDepthError,
    BudgetExceededError,
    DimensionOverflowError,
    IllegalPaletteError,
    InvalidParameterError,
)
from PIL.highdepth.model import ImageData, Palette, checked_mul, row_bytes
from PIL.highdepth.source import BytesSource, open_source

from .highdepth_fixtures.helpers import make_palette, make_rgb, save


def test_palette_requires_entries() -> None:
    with pytest.raises(IllegalPaletteError):
        Palette(())
    with pytest.raises(IllegalPaletteError):
        Palette(tuple((0, 0, 0, 255) for _ in range(257)))
    with pytest.raises(IllegalPaletteError):
        Palette(((0, 0, 0),))


def test_palette_conversions_roundtrip() -> None:
    rgb = Palette.from_rgb(bytes(range(9)))
    assert rgb.entries == ((0, 1, 2, 255), (3, 4, 5, 255), (6, 7, 8, 255))
    assert rgb.rgb_bytes() == bytes(range(9))
    rgba = Palette.from_rgba(bytes(range(12)))
    assert rgba.has_alpha
    assert rgba.rgba_bytes() == bytes(range(12))
    assert rgba.alpha_bytes() == bytes((3, 7, 11))


def test_palette_with_entry_is_immutable() -> None:
    original = make_palette()
    updated = original.with_entry(0, (1, 2, 3, 4))
    assert original.entries[0] != updated.entries[0]
    assert updated.entries[1:] == original.entries[1:]


def test_row_bytes_rounding() -> None:
    assert row_bytes(3, 1, 1) == 1
    assert row_bytes(9, 1, 1) == 2
    assert row_bytes(3, 3, 8) == 9
    assert row_bytes(3, 3, 16) == 18
    with pytest.raises(InvalidParameterError):
        row_bytes(0, 1, 8)


def test_checked_mul_overflow() -> None:
    assert checked_mul(0, 10) == 0
    with pytest.raises(DimensionOverflowError):
        checked_mul(-1, 2)
    with pytest.raises(DimensionOverflowError):
        checked_mul(2**62, 4)


def test_image_data_defaults_stride() -> None:
    data = ImageData("PNG", "RGB", 3, 2, 8, 8, 3, bytes(18))
    assert data.row_stride == 9
    assert data.row_length_bytes == 9


def test_image_data_rejects_bad_buffer() -> None:
    with pytest.raises(InvalidParameterError):
        ImageData("PNG", "RGB", 3, 2, 8, 8, 3, bytes(17))


def test_palette_image_requires_palette() -> None:
    with pytest.raises(IllegalPaletteError):
        ImageData("PNG", "P", 2, 2, 8, 8, 1, bytes(4))


def test_palette_index_checked_against_depth() -> None:
    palette = make_palette(4)
    with pytest.raises(IllegalPaletteError):
        ImageData("PNG", "P", 2, 1, 2, 8, 1, bytes((0, 5)), palette=palette)


def test_transparency_tuple_length() -> None:
    with pytest.raises(InvalidParameterError):
        ImageData("PNG", "RGB", 1, 1, 8, 8, 3, bytes(3), transparency=(1, 2))


def test_row_and_sample_access() -> None:
    data = ImageData.from_pil(make_rgb(4, 2))
    assert data.row_samples(0) == tuple(data.samples[:12])
    assert data.sample_at(1, 1) == data.row_samples(1)[3:6]
    with pytest.raises(InvalidParameterError):
        data.sample_at(99, 0)
    with pytest.raises(InvalidParameterError):
        data.row_samples(2)


def test_from_pil_preserves_gray_and_rgba_modes() -> None:
    from PIL import Image

    for im in (
        Image.new("L", (2, 2), 7),
        Image.new("LA", (2, 2), (7, 128)),
        Image.new("RGB", (2, 2), (1, 2, 3)),
        Image.new("RGBA", (2, 2), (1, 2, 3, 4)),
    ):
        data = ImageData.from_pil(im)
        assert data.mode == im.mode
        assert data.tobytes() == im.tobytes()


def test_from_pil_palette_keeps_indices() -> None:
    from PIL import Image

    im = Image.new("P", (3, 1))
    im.putpalette(make_palette(4).rgb_bytes())
    im.putdata([0, 1, 2])
    data = ImageData.from_pil(im)
    assert data.mode == "P"
    assert data.samples == bytes((0, 1, 2))


def test_to_pil_gray16_roundtrip() -> None:
    values = [0, 1, 1000, 65535]
    data = ImageData("PNG", "L", 2, 2, 16, 16, 1, array("H", values))
    im = data.to_pil()
    assert im.mode == "I;16"
    # Pillow's I;16 raw bytes are host endian; compare the samples instead.
    assert list(im.get_flattened_data()) == values
    assert ImageData.from_pil(im).tobytes() == data.tobytes()


def test_to_pil_rgb16_requires_explicit_reduction() -> None:
    data = ImageData("PNG", "RGB", 1, 1, 16, 16, 3, array("H", [0, 0x8000, 0xFFFF]))
    with pytest.raises(BitDepthError):
        data.to_pil()
    im = data.to_pil(allow_reduce=True)
    assert im.getpixel((0, 0)) == (0, 128, 255)


def test_to_rgba_does_not_blacken_transparent_entries() -> None:
    palette = Palette(((10, 20, 30, 0), (200, 100, 50, 255)))
    data = ImageData("PNG", "P", 2, 1, 1, 8, 1, bytes((0, 1)), palette=palette)
    rgba = data.to_rgba()
    assert rgba.sample_at(0, 0) == (10, 20, 30, 0)


def test_flatten_rounds_half_up() -> None:
    data = ImageData("PNG", "LA", 2, 1, 8, 8, 2, bytes((255, 0, 0, 255)))
    flattened = data.flatten((255, 255, 255))
    assert flattened.sample_at(0, 0) == (255,)
    assert flattened.sample_at(1, 0) == (0,)


def test_sha256_distinguishes_metadata() -> None:
    base = ImageData.from_pil(make_rgb())
    other = ImageData.from_pil(make_rgb())
    assert base.sha256() == other.sha256()
    changed = base.with_info(comment="x")
    assert changed.sha256() == base.sha256()


# ------------------------------------------------------------------ budget


def test_budget_rejects_before_allocating() -> None:
    budget = hd.MemoryBudget(10)
    with pytest.raises(BudgetExceededError):
        budget.reserve(11, "row")
    assert budget.used == 0
    budget.reserve(4)
    assert budget.used == 4
    budget.release(4)
    assert budget.used == 0


def test_budget_peak_tracking() -> None:
    budget = hd.MemoryBudget(100)
    budget.reserve(40)
    budget.reserve(50)
    budget.release(50)
    budget.reserve(20)
    assert budget.peak == 90


def test_budget_rejects_non_positive_limit() -> None:
    with pytest.raises(InvalidParameterError):
        hd.MemoryBudget(0)


def test_iter_tiles_covers_image() -> None:
    tiles = list(hd.iter_tiles(7, 5, 2))
    covered = {
        (x, y)
        for tile in tiles
        for y in range(tile.y0, tile.y1)
        for x in range(tile.x0, tile.x1)
    }
    assert covered == {(x, y) for y in range(5) for x in range(7)}


def test_iter_tiles_cancels() -> None:
    token = hd.CancellationToken()
    iterator = hd.iter_tiles(10, 10, 2, token=token)
    next(iterator)
    token.cancel()
    with pytest.raises(hd.CancelledError):
        next(iterator)


# ------------------------------------------------------------------ sources


def test_source_forms_are_equivalent() -> None:
    payload = save(make_rgb(), "PNG")
    direct = hd.read(payload)
    chunked = hd.read(iter([payload[i : i + 5] for i in range(0, len(payload), 5)]))
    from_bytes_io = hd.read(io.BytesIO(payload))
    assert direct.sha256() == chunked.sha256() == from_bytes_io.sha256()


def test_bytes_source_does_not_copy() -> None:
    source = BytesSource(bytearray(b"hello"))
    assert isinstance(source, BytesSource)
    assert source.size == 5
    assert source.read_at(1, 3) == b"ell"


def test_spooling_source_rejects_non_bytes() -> None:
    with pytest.raises(hd.SourceError):
        open_source(iter([b"ok", 5]), budget=hd.MemoryBudget(100))


def test_spooling_source_budget() -> None:
    with pytest.raises(BudgetExceededError):
        open_source(iter([b"x" * 50, b"y" * 50]), budget=hd.MemoryBudget(60))


def test_source_read_exact_raises_on_short_read() -> None:
    source = BytesSource(b"abc")
    assert source.read_exact(0, 3) == b"abc"
    with pytest.raises(hd.TruncatedDataError):
        source.read_exact(1, 3)


def test_unsupported_source_type() -> None:
    with pytest.raises(hd.SourceError):
        open_source(42)  # type: ignore[arg-type]
