"""Determinism, tie-breaks and alpha handling for palette quantization."""

from __future__ import annotations

import pytest

from PIL import Image
from PIL.highdepth import (
    BitDepthError,
    EmptyImageError,
    InvalidParameterError,
    Palette,
    build_palette,
    map_pixels,
    quantize,
)
from PIL.highdepth.model import ImageData
from PIL.highdepth.quantize import ALPHA_WEIGHT

from .highdepth_fixtures.helpers import make_rgb, make_rgba


def _rgb(rgb: tuple[int, ...]) -> ImageData:
    return ImageData.from_pil(Image.new("RGB", (len(rgb) // 3, 1), rgb))


def _pixels(rgb: tuple[int, ...], alpha: tuple[int, ...] | None = None) -> ImageData:
    count = len(rgb) // 3
    data = bytearray()
    for index in range(count):
        data += bytes(rgb[index * 3 : index * 3 + 3])
        data.append(255 if alpha is None else alpha[index])
    return ImageData("PNG", "RGBA", count, 1, 8, 8, 4, bytes(data))


# ------------------------------------------------------------ determinism


@pytest.mark.parametrize("seed", (0, 1, 12345))
@pytest.mark.parametrize("colors", (2, 4, 16))
def test_quantize_is_byte_reproducible(seed: int, colors: int) -> None:
    data = ImageData.from_pil(make_rgb(16, 8))
    first = quantize(data, colors, dither=True, seed=seed)
    second = quantize(data, colors, dither=True, seed=seed)
    assert first.tobytes() == second.tobytes()
    assert first.palette == second.palette
    # A fresh decode from bytes must be identical too.
    third = quantize(
        ImageData.from_pil(make_rgb(16, 8)), colors, dither=True, seed=seed
    )
    assert third.tobytes() == first.tobytes()


def test_seed_changes_dither_but_not_palette() -> None:
    data = ImageData.from_pil(make_rgb(16, 8))
    a = quantize(data, 4, dither=True, seed=0)
    b = quantize(data, 4, dither=True, seed=9)
    assert a.palette == b.palette


def test_no_dither_is_pure_nearest() -> None:
    data = _pixels((0, 0, 0, 64, 64, 64, 200, 200, 200))
    palette = Palette(((0, 0, 0, 255), (255, 255, 255, 255)))
    result = map_pixels(data, palette, dither=False)
    assert list(result.samples) == [0, 0, 1]


# ------------------------------------------------------------ tie-breaks


def test_tie_break_prefers_lower_index() -> None:
    # A pixel exactly halfway between two palette entries must pick index 0.
    palette = Palette(((0, 0, 0, 255), (200, 0, 0, 255)))
    data = _pixels((100, 0, 0))
    result = map_pixels(data, palette, dither=False)
    assert result.samples == bytes((0,))


def test_tie_break_in_build_palette_is_sorted() -> None:
    data = ImageData.from_pil(make_rgb(8, 2))
    palette = build_palette(data, 4)
    keys = [(entry[3], entry[0], entry[1], entry[2]) for entry in palette.entries]
    assert keys == sorted(keys)


def test_median_cut_channel_tie_breaks_to_red() -> None:
    # Two colours differ equally in R, G and B; the split must be on R.
    data = _pixels((0, 0, 0, 10, 10, 10))
    palette = build_palette(data, 2)
    assert len(palette) == 2
    assert palette.entries[0][0] < palette.entries[1][0]


def test_exact_histogram_when_colours_fit() -> None:
    rgb = (255, 0, 0, 0, 255, 0, 0, 0, 255)
    palette = build_palette(_pixels(rgb), 8)
    assert palette == Palette(((0, 0, 255, 255), (0, 255, 0, 255), (255, 0, 0, 255)))


# ------------------------------------------------------------ alpha


def test_transparent_pixels_share_one_entry() -> None:
    alpha = [0, 0, 255, 255]
    data = _pixels((0, 0, 0, 255, 255, 255, 10, 10, 10, 200, 200, 200), alpha)
    result = quantize(data, 4, dither=False)
    assert result.palette is not None
    assert result.palette.entries[0][3] == 0
    assert result.samples[0] == result.samples[1] == 0


def test_transparent_edges_are_not_blackened() -> None:
    # A transparent black pixel adjacent to opaque red and green must not bleed
    # error into either neighbour.
    data = bytearray()
    colours = [(0, 0, 0, 0), (255, 0, 0, 255), (0, 255, 0, 255)] * 3
    for pixel in colours:
        data += bytes(pixel)
    image = ImageData("PNG", "RGBA", 3, 3, 8, 8, 4, bytes(data))
    result = quantize(image, 4, dither=True)
    assert result.palette is not None
    # The opaque reds and greens keep their exact palette entries.
    red_index = result.palette.entries.index((255, 0, 0, 255))
    green_index = result.palette.entries.index((0, 255, 0, 255))
    row = 3
    for y in range(3):
        assert result.samples[y * row + 1] == red_index
        assert result.samples[y * row + 2] == green_index


def test_dither_never_emits_from_transparent() -> None:
    data = _pixels(
        (255, 0, 0, 0, 0, 0, 255, 0, 0),
        (255, 0, 255),
    )
    palette = Palette(((0, 0, 0, 0), (255, 0, 0, 255), (0, 0, 255, 255)))
    result = map_pixels(data, palette, dither=True)
    # The middle transparent pixel cannot receive residual from its left
    # neighbour; it stays on the transparent entry.
    assert result.samples[1] == 0


def test_partial_alpha_does_not_emit_error() -> None:
    data = _pixels((255, 127, 0), (128,))
    palette = Palette(((255, 0, 0, 255), (0, 255, 0, 255)))
    result = map_pixels(data, palette, dither=True)
    assert len(result.samples) == 1


def test_alpha_weight_affects_distance() -> None:
    from PIL.highdepth.quantize import _distance

    assert _distance((0, 0, 0, 255), (0, 0, 0, 0)) == pytest.approx(
        ALPHA_WEIGHT * 255**2
    )
    assert _distance((0, 0, 0, 0), (255, 255, 255, 0)) == 0.0


# ------------------------------------------------------------ guards


def test_empty_image_rejected() -> None:
    data = ImageData("PNG", "RGB", 0, 0, 8, 8, 3, b"")
    with pytest.raises(EmptyImageError):
        quantize(data)


def test_solid_colour_image() -> None:
    data = _pixels((17, 34, 51) * 10)
    result = quantize(data, 8)
    assert result.palette == Palette(((17, 34, 51, 255),))
    assert set(result.samples) == {0}


def test_odd_dimensions() -> None:
    data = ImageData.from_pil(make_rgb(7, 5))
    result = quantize(data, 4)
    assert (result.width, result.height) == (7, 5)
    assert len(result.samples) == 35


def test_sixteen_bit_input_refused() -> None:
    from array import array

    data = ImageData("PNG", "L", 2, 1, 16, 16, 1, array("H", [0, 65535]))
    with pytest.raises(BitDepthError):
        quantize(data)


def test_colours_out_of_range() -> None:
    data = ImageData.from_pil(make_rgb())
    for colors in (0, 257):
        with pytest.raises(InvalidParameterError):
            build_palette(data, colors)


def test_quantize_outputs_valid_palette_mode() -> None:
    data = ImageData.from_pil(make_rgba())
    result = quantize(data, 8, dither=False)
    assert result.mode == "P"
    assert result.storage_bits == 8
    assert result.bit_depth <= 8
    result.validate()


def test_transparent_index_recorded() -> None:
    data = _pixels((1, 2, 3, 4, 5, 6), (0, 255))
    result = quantize(data, 4)
    assert result.transparency == 0


def test_quantize_to_pillow_palette_is_valid() -> None:
    data = ImageData.from_pil(make_rgba())
    result = quantize(data, 8)
    im = result.to_pil()
    assert im.mode == "P"
    palette = im.getpalette("RGBA")
    assert palette is not None
    assert len(palette) >= len(result.palette) * 4


def test_large_colour_count_is_capped_by_distinct_colours() -> None:
    data = _pixels((0, 0, 0, 0, 0, 0, 255, 255, 255))
    result = quantize(data, 256)
    assert len(result.palette) == 2
