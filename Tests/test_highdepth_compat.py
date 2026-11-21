"""Compatibility with Image, ImageFile, ImageCms and the classic plugins."""

from __future__ import annotations

import io
import os

import pytest

from PIL import Image, ImageCms, features
from PIL.highdepth import BitDepthError, ImageData, ProfileError, compat, read

from .highdepth_fixtures.helpers import (
    ICC_SRGB,
    make_palette,
    make_rgb,
    save,
)


def test_open_normalized_matches_pillow() -> None:
    payload = save(make_rgb(), "PNG")
    data = compat.open_normalized(io.BytesIO(payload))
    with Image.open(io.BytesIO(payload)) as im:
        assert data.mode == im.mode
        assert data.tobytes() == im.tobytes()


def test_open_normalized_palette_depth() -> None:
    im = Image.new("P", (4, 2))
    im.putpalette(make_palette(4).rgb_bytes())
    im.putdata([0, 1, 2, 3] * 2)
    payload = save(im, "PNG")
    data = compat.open_normalized(io.BytesIO(payload), bit_depth=8)
    assert data.mode == "P"
    assert data.bit_depth == 8


def test_save_normalized_roundtrip() -> None:
    data = ImageData.from_pil(make_rgb())
    out = io.BytesIO()
    compat.save_normalized(data, out, "TIFF")
    assert read(out.getvalue()).tobytes() == data.tobytes()


def test_to_and_from_high_depth() -> None:
    im = make_rgb()
    data = compat.to_high_depth(im)
    restored = compat.from_high_depth(data)
    assert restored.tobytes() == im.tobytes()


def test_quantize_image_returns_palette_image() -> None:
    result = compat.quantize_image(make_rgb(16, 8), 4)
    assert result.mode == "P"
    assert result.palette is not None
    colors = result.getcolors()
    assert colors is not None
    assert len(colors) <= 4


def test_quantize_image_with_fixed_palette() -> None:
    palette = Image.new("P", (1, 1))
    palette.putpalette([0, 0, 0, 255, 0, 0])
    original = make_rgb(8, 4)
    result = compat.quantize_image(original, palette=palette)
    assert result.mode == "P"
    again = compat.quantize_image(original, palette=palette)
    assert result.tobytes() == again.tobytes()


def test_quantize_image_rejects_non_palette() -> None:
    with pytest.raises(Exception):
        compat.quantize_image(make_rgb(), palette=make_rgb())


def test_quantize_image_16bit_needs_explicit_depth() -> None:
    im = Image.new("I;16", (2, 2), 1000)
    with pytest.raises(BitDepthError):
        compat.quantize_image(im)


def test_sixteen_bit_pillow_image_normalizes_and_roundtrips() -> None:
    im = Image.new("I;16", (3, 2))
    im.putdata([0, 1, 1000, 40000, 65535, 7])
    data = compat.to_high_depth(im)
    assert data.storage_bits == 16
    restored = compat.from_high_depth(data)
    assert list(restored.get_flattened_data()) == [0, 1, 1000, 40000, 65535, 7]


def test_read_pillow_frames_matches_image_sequence() -> None:
    from PIL import ImageSequence

    frames = [Image.new("L", (4, 3), value) for value in (10, 20, 30)]
    out = io.BytesIO()
    frames[0].save(out, "TIFF", save_all=True, append_images=frames[1:])
    payload = out.getvalue()
    normalized = compat.read_pillow_frames(payload)
    with Image.open(io.BytesIO(payload)) as im:
        pillow_bytes = [frame.tobytes() for frame in ImageSequence.Iterator(im)]
    assert [frame.tobytes() for frame in normalized] == pillow_bytes


def test_apply_icc_leaves_input_on_failure() -> None:
    pytest.importorskip("PIL._imagingcms")
    if not os.path.exists(ICC_SRGB):
        pytest.skip("sRGB profile not available")
    profile = ImageCms.ImageCmsProfile(ICC_SRGB)
    transform = ImageCms.buildTransform(profile, profile, "RGB", "RGB")
    im = make_rgb()
    fingerprint = im.tobytes()
    # A mode-mismatched image must fail before any mutation.
    gray = im.convert("L")
    with pytest.raises(ProfileError):
        compat.apply_icc(gray, transform, in_place=True)
    assert im.tobytes() == fingerprint


def test_apply_icc_in_place_success() -> None:
    pytest.importorskip("PIL._imagingcms")
    if not os.path.exists(ICC_SRGB):
        pytest.skip("sRGB profile not available")
    profile = ImageCms.ImageCmsProfile(ICC_SRGB)
    transform = ImageCms.buildTransform(profile, profile, "RGB", "RGB")
    im = make_rgb()
    expected = transform.apply(im).tobytes()
    returned = compat.apply_icc(im, transform, in_place=True)
    assert returned is im
    assert im.tobytes() == expected


def test_apply_pipeline_returns_pillow_image() -> None:
    from PIL.highdepth import ColorPipeline

    im = make_rgb()
    result = compat.apply_pipeline(im, ColorPipeline("srgb", "linear", output_bits=16))
    assert result.mode == "RGB"
    assert isinstance(result, Image.Image)


def test_stream_to_pillow_tiles() -> None:
    data = ImageData.from_pil(make_rgb(6, 4))
    tiles = compat.stream_to_pillow(data, tile_height=2)
    assert len(tiles) == 2
    assert tiles[0].size == (6, 2)


def test_save_path_atomic(tmp_path: object) -> None:
    from pathlib import Path

    target = Path(str(tmp_path)) / "out.bmp"
    data = ImageData.from_pil(make_rgb())
    compat.save_path(data, target, "BMP")
    assert target.exists()


@pytest.mark.skipif(not features.check("littlecms2"), reason="littleCMS not available")
def test_imagecms_transform_still_works_alongside() -> None:
    if not os.path.exists(ICC_SRGB):
        pytest.skip("sRGB profile not available")
    im = make_rgb()
    out = ImageCms.profileToProfile(im, ICC_SRGB, ICC_SRGB)
    assert out is not None
    assert out.size == im.size
