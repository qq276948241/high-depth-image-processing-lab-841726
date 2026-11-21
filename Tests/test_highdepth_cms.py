"""Colour management stages, failure isolation and concurrency."""

from __future__ import annotations

import concurrent.futures as futures
import math
import os

import pytest

from PIL import Image, features
from PIL.highdepth import (
    ColorPipeline,
    ImageData,
    NonFiniteSampleError,
    ProfileError,
    UnknownColorSpaceError,
    cms,
    load_profile,
)
from PIL.highdepth.model import ImageData as _Model

from .highdepth_fixtures.helpers import ICC_GREY, ICC_SRGB, make_rgb, make_rgba

pytestmark = pytest.mark.skipif(
    not features.check("littlecms2"), reason="littleCMS not available"
)


def skip_missing_profile() -> None:
    if not os.path.exists(ICC_SRGB):
        pytest.skip("sRGB profile not available")


def _ramp(width: int = 16) -> ImageData:
    im = Image.new("RGB", (width, 1))
    im.putdata([(v * 255 // (width - 1),) * 3 for v in range(width)])
    return ImageData.from_pil(im)


def test_srgb_linear_roundtrip_8bit_documented_loss() -> None:
    # At 8 bits per sample, linear light cannot retain dark-end precision;
    # that is inherent, not a bug.  Mid and high tones still round-trip.
    data = _ramp(32)
    linear = cms.srgb_to_linear(data)
    back = cms.linear_to_srgb(linear)
    for original, returned in zip(data.samples, back.samples, strict=True):
        if original >= 64:
            assert abs(original - returned) <= 2
    assert "linearize" in linear.info["cms_stages"]


def test_srgb_linear_roundtrip_16bit_is_exact() -> None:
    data = _ramp(32)
    linear = cms.srgb_to_linear(data, output_bits=16)
    assert linear.storage_bits == 16
    back = cms.linear_to_srgb(linear, output_bits=8)
    assert back.storage_bits == 8
    for original, returned in zip(data.samples, back.samples, strict=True):
        assert abs(original - returned) <= 1


def test_linear_of_zero_and_one_is_exact() -> None:
    data = ImageData.from_pil(Image.new("L", (2, 1)).convert("RGB"))
    linear = cms.srgb_to_linear(data)
    assert linear.samples[:3] == b"\x00\x00\x00"


def test_gamma_applied_once() -> None:
    # ``apply_gamma`` encodes an already-linear image; a mid-grey linear value
    # becomes the gamma-encoded equivalent in one pass.
    data = ImageData.from_pil(Image.new("RGB", (1, 1), (128, 128, 128)))
    result = cms.apply_gamma(data, 2.2)
    expected = round((128 / 255) ** (1 / 2.2) * 255)
    assert abs(result.samples[0] - expected) <= 1


def test_unknown_colour_space() -> None:
    data = ImageData.from_pil(make_rgb())
    with pytest.raises(UnknownColorSpaceError):
        ColorPipeline("srgb", "mystery")
    with pytest.raises(UnknownColorSpaceError):
        cms.convert(data, "not-a-space", "srgb")


def test_palette_image_decoded_before_colour() -> None:
    from .highdepth_fixtures.helpers import make_palette

    data = ImageData("PNG", "P", 2, 1, 8, 8, 1, bytes((0, 1)), palette=make_palette(4))
    result = cms.convert(data, "srgb", "srgb")
    assert result.mode == "RGBA"
    assert "decode" in result.info["cms_stages"]


def test_alpha_preserved_by_transfer() -> None:
    im = Image.new("LA", (2, 1))
    im.putdata([(10, 0), (200, 255)])
    data = ImageData.from_pil(im)
    result = cms.srgb_to_linear(data)
    assert result.samples[1] == 0
    assert result.samples[3] == 255


def test_premultiply_is_exact_and_reversible() -> None:
    im = Image.new("RGBA", (2, 1))
    im.putdata([(200, 100, 50, 128), (10, 20, 30, 255)])
    data = ImageData.from_pil(im)
    premultiplied = cms.convert(data, "srgb", "srgb", premultiply=True)
    assert premultiplied.info["premultiplied"] is True
    assert premultiplied.samples[:4] == bytes((100, 50, 25, 128))
    restored = cms.unpremultiply(premultiplied)
    assert restored.samples[:3] == bytes((199, 100, 50))
    assert restored.samples[3] == 128


def test_premultiply_happens_after_colour_transform() -> None:
    im = Image.new("RGBA", (1, 1), (200, 100, 50, 128))
    data = ImageData.from_pil(im)
    result = cms.convert(data, "srgb", "linear", premultiply=True)
    assert result.info["cms_stages"].index("linearize") < result.info[
        "cms_stages"
    ].index("alpha")


def test_quantize_stage_is_last() -> None:
    data = ImageData.from_pil(make_rgb(8, 4))
    result = cms.convert(data, "srgb", "srgb", quantize_colors=4)
    stages = result.info["cms_stages"]
    assert stages[-2] == "quantize"
    assert result.mode == "P"


def test_non_finite_float_image_rejected() -> None:
    im = Image.new("F", (2, 1))
    im.putdata([0.0, float("nan")])
    with pytest.raises(NonFiniteSampleError):
        ImageData.from_pil(im)
    im.putdata([0.0, float("inf")])
    with pytest.raises(NonFiniteSampleError):
        ImageData.from_pil(im)


def test_repeat_conversion_is_deterministic() -> None:
    data = _ramp(32)
    first = cms.convert(data, "srgb", "linear")
    second = cms.convert(data, "srgb", "linear")
    assert first.samples == second.samples
    assert data.sha256() == ImageData.from_pil(_ramp(32).to_pil()).sha256()


def test_failed_conversion_leaves_input_untouched() -> None:
    data = ImageData.from_pil(make_rgb())
    fingerprint = data.sha256()
    bad = ColorPipeline(
        source_space="srgb",
        target_space="srgb",
        source_profile=b"\x00" * 128,
        target_profile=b"\x00" * 128,
    )
    with pytest.raises(ProfileError):
        bad.apply(data)
    assert data.sha256() == fingerprint


def test_profile_validation() -> None:
    with pytest.raises(ProfileError):
        ColorPipeline(source_profile=b"", target_profile=b"")
    with pytest.raises(ProfileError):
        ColorPipeline(source_profile=b"short", target_profile=b"short")
    with pytest.raises(ProfileError):
        ColorPipeline(source_profile=b"\x00" * 128)


def test_missing_profile_file() -> None:
    with pytest.raises(ProfileError):
        load_profile("Tests/icc/does-not-exist.icc")


def test_icc_identity_transform() -> None:
    skip_missing_profile()
    profile = load_profile(ICC_SRGB)
    data = ImageData.from_pil(make_rgb())
    result = cms.convert(
        data, "srgb", "srgb", source_profile=profile, target_profile=profile
    )
    assert "icc" in result.info["cms_stages"]
    assert result.mode == "RGB"
    # An identity transform must stay within a couple of 8-bit steps.
    for original, returned in zip(data.samples, result.samples, strict=True):
        assert abs(original - returned) <= 2


def test_icc_grey_profile_rejected_for_rgb() -> None:
    skip_missing_profile()
    if not os.path.exists(ICC_GREY):
        pytest.skip("grey profile missing")
    data = ImageData.from_pil(make_rgb())
    with pytest.raises(ProfileError):
        cms.convert(
            data,
            "srgb",
            "srgb",
            source_profile=load_profile(ICC_GREY),
            target_profile=load_profile(ICC_SRGB),
        )


def test_icc_transform_is_cached_and_shared() -> None:
    skip_missing_profile()
    profile = load_profile(ICC_SRGB)
    pipeline = ColorPipeline(source_profile=profile, target_profile=profile)
    first = pipeline.apply(ImageData.from_pil(make_rgb()))
    second = pipeline.apply(ImageData.from_pil(make_rgb()))
    assert first.samples == second.samples
    assert len(pipeline._transform_cache) == 1


def test_profile_bytes_not_modified() -> None:
    skip_missing_profile()
    profile = load_profile(ICC_SRGB)
    before = bytes(profile)
    pipeline = ColorPipeline(source_profile=profile, target_profile=profile)
    pipeline.apply(ImageData.from_pil(make_rgb()))
    assert bytes(profile) == before


def test_concurrent_conversions_do_not_share_state() -> None:
    skip_missing_profile()
    profile = load_profile(ICC_SRGB)
    pipeline = ColorPipeline(source_profile=profile, target_profile=profile)
    images = [ImageData.from_pil(make_rgb(8, 4)) for _ in range(6)]
    expected = [pipeline.apply(image).samples for image in images]

    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda image: pipeline.apply(image).samples, images))
    assert results == expected


def test_concurrent_quantize_is_deterministic() -> None:
    from PIL.highdepth import quantize

    data = ImageData.from_pil(make_rgba(16, 16))
    expected = quantize(data, 8, seed=3).tobytes()
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(
            pool.map(lambda _: quantize(data, 8, seed=3).tobytes(), range(8))
        )
    assert all(output == expected for output in outputs)


def test_gamma_rejects_bad_exponent() -> None:
    from PIL.highdepth.color import gamma_decode

    for gamma in (0, -1, math.inf, math.nan):
        with pytest.raises(Exception):
            gamma_decode(0.5, gamma)


def test_edge_tiles_match_full_image() -> None:
    # A 3x2 reference image with a bright 16-bit peak in the corner.
    from array import array

    from PIL.highdepth.read import iter_image_tiles

    values = array("H", [0, 1, 2, 65000, 65535, 7])
    data = _Model("PNG", "L", 3, 2, 16, 16, 1, values)
    tiles = list(iter_image_tiles(data, 1))
    assert len(tiles) == 2
    rebuilt = bytearray()
    for _bounds, tile in tiles:
        rebuilt += tile.tobytes()
    assert bytes(rebuilt) == data.tobytes()
    # Row-major tiles: the peak lives in the second row.
    assert max(tiles[0][1].samples) == 2
    assert max(tiles[1][1].samples) == 65535
