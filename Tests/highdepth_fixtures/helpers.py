"""Shared helpers for the PIL.highdepth test-suite."""

from __future__ import annotations

import io
import os
import struct

import pytest

from PIL import Image
from PIL.highdepth.model import ImageData, Palette

# Re-exported to keep individual test modules small.
ICC_SRGB = os.path.join("Tests", "icc", "sRGB_IEC61966-2-1_black_scaled.icc")
ICC_GREY = os.path.join("Tests", "icc", "sGrey-v2-nano.icc")
ICC_DISPLAY = os.path.join("Tests", "icc", "sRGB_v4_ICC_preference.icc")


def make_rgb(width: int = 6, height: int = 4) -> Image.Image:
    im = Image.new("RGB", (width, height))
    im.putdata(
        [
            (x * 31 % 256, y * 67 % 256, (x * 13 + y * 29) % 256)
            for y in range(height)
            for x in range(width)
        ]
    )
    return im


def make_gray(width: int = 5, height: int = 3) -> Image.Image:
    im = Image.new("L", (width, height))
    im.putdata([(x * 41 + y * 17) % 256 for y in range(height) for x in range(width)])
    return im


def make_rgba(width: int = 4, height: int = 4) -> Image.Image:
    im = Image.new("RGBA", (width, height))
    im.putdata(
        [
            (x * 60 % 256, y * 60 % 256, 128, 255 if (x + y) % 2 else 0)
            for y in range(height)
            for x in range(width)
        ]
    )
    return im


def make_palette(entries: int = 8, with_alpha: bool = False) -> Palette:
    values = []
    for i in range(entries):
        alpha = 0 if with_alpha and i == 0 else 255
        values.append((i * 32 % 256, (i * 48 + 7) % 256, (i * 96 + 3) % 256, alpha))
    return Palette(tuple(values))


def save(im: Image.Image, format: str, **options: object) -> bytes:
    out = io.BytesIO()
    im.save(out, format, **options)
    return out.getvalue()


@pytest.fixture
def rgb_image() -> Image.Image:
    return make_rgb()


@pytest.fixture
def gray_image() -> Image.Image:
    return make_gray()


@pytest.fixture
def rgba_image() -> Image.Image:
    return make_rgba()


def tiff_tag_values(payload: bytes, page: int = 0) -> dict[int, int]:
    """Read scalar SHORT/LONG tag values from a little-endian TIFF page."""
    assert payload[:4] == b"II\x2a\x00"
    ifd = struct.unpack_from("<I", payload, 4)[0]
    for _ in range(page):
        count = struct.unpack_from("<H", payload, ifd)[0]
        ifd = struct.unpack_from("<I", payload, ifd + 2 + 12 * count)[0]
    count = struct.unpack_from("<H", payload, ifd)[0]
    out: dict[int, int] = {}
    for index in range(count):
        tag, type_id, item_count, value = struct.unpack_from(
            "<HHII", payload, ifd + 2 + 12 * index
        )
        if type_id == 3 and item_count == 1:
            out[tag] = value & 0xFFFF
        elif type_id == 4 and item_count == 1:
            out[tag] = value
    return out


def row_tiles(data: ImageData, height: int) -> list[ImageData]:
    from PIL.highdepth.read import iter_image_tiles

    return [tile for _bounds, tile in iter_image_tiles(data, height)]


def assert_samples_equal(a: ImageData, b: ImageData) -> None:
    assert a.mode == b.mode
    assert (a.width, a.height) == (b.width, b.height)
    assert a.storage_bits == b.storage_bits
    assert a.tobytes() == b.tobytes()
    assert a.palette == b.palette


__all__ = [
    "ICC_DISPLAY",
    "ICC_GREY",
    "ICC_SRGB",
    "assert_samples_equal",
    "make_gray",
    "make_palette",
    "make_rgb",
    "make_rgba",
    "row_tiles",
    "save",
    "tiff_tag_values",
]
