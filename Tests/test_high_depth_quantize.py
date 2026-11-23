from __future__ import annotations

import pytest

from PIL import HighDepth, Image, Quantize

TYPE_CHECKING = False


def _i16(values: tuple[int, ...]) -> Image.Image:
    image = Image.new("I;16", (len(values), 1))
    image.putdata(values)
    return image


def test_high_depth_modes_use_python_path() -> None:
    # Arrange
    image = _i16((0, 1000, 20000, 65535))

    # Act
    quantized = image.quantize(2)

    # Assert
    assert quantized.mode == "P"
    assert Quantize.HIGH_DEPTH_MODES >= {"I;16", "I", "LA", "PA"}


def test_quantize_keeps_16_bit_decisions() -> None:
    # Arrange: the two 8-bit values would collide, but the 16-bit values differ.
    image = _i16((0x0100, 0x01FF))

    # Act
    quantized = image.quantize(2, dither=Image.Dither.NONE)

    # Assert
    assert len(set(quantized.tobytes())) == 2


def test_quantize_is_byte_identical() -> None:
    # Arrange
    image = Image.new("I;16", (8, 4))
    image.putdata(tuple((index * 2731) % 65536 for index in range(32)))

    # Act
    results = [image.quantize(4).tobytes() for _ in range(3)]

    # Assert
    assert len(set(results)) == 1


def test_quantize_palette_is_sorted() -> None:
    # Arrange
    image = Image.new("I;16", (4, 1))
    image.putdata((65535, 0, 32768, 16384))

    # Act
    quantized = image.quantize(4, dither=Image.Dither.NONE)
    palette = quantized.getpalette("RGB")

    # Assert
    entries = [tuple(palette[i : i + 3]) for i in range(0, len(palette), 3)]
    assert entries == sorted(entries)


def test_nearest_tie_breaks_to_lowest_index() -> None:
    # Arrange: a mid grey is exactly between two palette entries.
    image = Image.new("LA", (1, 1), (128, 255))

    # Act
    quantized = Quantize.quantize(image, 2, dither=False)

    # Assert
    palette = quantized.getpalette("RGBA")
    entries = [palette[i : i + 4] for i in range(0, len(palette), 4)]
    assert quantized.getpixel((0, 0)) == entries.index(entries[0])


def test_alpha_palette_is_rgba() -> None:
    # Arrange
    image = Image.new("LA", (2, 1))
    image.putdata(((0, 255), (255, 0)))

    # Act
    quantized = Quantize.quantize(image, 2, dither=False)

    # Assert
    assert quantized.palette.mode == "RGBA"
    assert quantized.convert("RGBA").getpixel((1, 0))[3] == 0


def test_transparent_edge_is_not_darkened() -> None:
    # Arrange: a fully transparent black pixel next to opaque red.
    image = Image.new("RGBA", (3, 1))
    image.putdata(((0, 0, 0, 0), (255, 0, 0, 255), (255, 0, 0, 255)))

    # Act
    quantized = Quantize.quantize(image, 2)
    result = quantized.convert("RGBA")

    # Assert: the opaque red pixels stay red, not darkened towards black.
    assert result.getpixel((1, 0)) == (255, 0, 0, 255)
    assert result.getpixel((2, 0)) == (255, 0, 0, 255)
    assert result.getpixel((0, 0)) == (0, 0, 0, 0)


def test_dither_deterministic_for_pa() -> None:
    # Arrange
    image = Image.new("LA", (4, 2))
    image.putdata([(value * 16, 255 - value * 8) for value in range(8)])
    quantized = image.quantize(2)
    palette = image.convert("P").getpalette()

    # Act
    results = [
        Quantize.quantize_to_palette(image, image.quantize(2)).tobytes()
        for _ in range(3)
    ]

    # Assert
    assert palette is not None
    assert len(set(results)) == 1
    assert quantized.mode == "P"


def test_empty_image() -> None:
    # Arrange
    image = Image.new("I;16", (0, 0))

    # Act
    quantized = Quantize.quantize(image, 4)

    # Assert
    assert quantized.size == (0, 0)
    assert quantized.getpalette() == []


def test_colors_out_of_range() -> None:
    # Arrange
    image = _i16((1, 2, 3))

    # Act / Assert
    for colors in (0, 257):
        with pytest.raises(ValueError, match="colors must be between"):
            Quantize.quantize(image, colors)


def test_web_palette_matches_core() -> None:
    # Arrange
    image = Image.new("RGB", (4, 1))
    image.putdata(((0, 0, 0), (255, 255, 255), (51, 51, 51), (128, 0, 0)))
    core = image.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    core_web = image.convert("P")

    # Act
    high = Quantize.quantize_to_web_palette(_i16((0, 65535)))

    # Assert
    assert core_web.getpalette("RGB")[:9] == [0] * 9
    assert len(high.getpalette("RGB")) == 226 * 3
    assert core.mode == "P"


def test_reference_palette_is_used() -> None:
    # Arrange
    image = _i16((0, 0xFFFF))
    reference = Image.new("P", (1, 1))
    reference.putpalette((255, 0, 0, 0, 0, 255))

    # Act
    quantized = image.quantize(palette=reference)

    # Assert
    assert quantized.getpalette() == list(reference.getpalette())


def test_reference_palette_rejects_non_p() -> None:
    # Arrange
    image = _i16((0, 1))
    reference = Image.new("RGB", (1, 1))

    # Act / Assert
    with pytest.raises(ValueError, match="bad mode for palette image"):
        Quantize.quantize_to_palette(image, reference)


def test_budget_is_checked() -> None:
    # Arrange
    image = _i16(tuple(range(64)))

    # Act / Assert
    with pytest.raises(MemoryError):
        Quantize.quantize(image, 4, budget=HighDepth.MemoryBudget(8))


def test_quantize_high_depth_entry_point() -> None:
    # Arrange
    image = _i16((0, 1000, 20000, 65535))

    # Act
    quantized = Image.quantize_high_depth(image, 4)

    # Assert
    assert quantized.mode == "P"
    assert len(quantized.getcolors() or []) <= 4


def test_source_is_unchanged() -> None:
    # Arrange
    image = Image.new("RGBA", (2, 1), (1, 2, 3, 4))
    original = image.tobytes()

    # Act
    Quantize.quantize(image, 2)

    # Assert
    assert image.tobytes() == original
