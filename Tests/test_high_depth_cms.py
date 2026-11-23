from __future__ import annotations

import pytest

from PIL import Image, ImageCms

from .helper import hopper

TYPE_CHECKING = False

try:
    ImageCms.core.profile_open
except ImportError:
    HAVE_CMS = False
else:
    HAVE_CMS = True

SRGB = "Tests/icc/sRGB_IEC61966-2-1_black_scaled.icc"


def setup_module() -> None:
    if not HAVE_CMS:
        pytest.skip("littlecms2 not available", allow_module_level=True)


def test_linearize_planes_black_and_white() -> None:
    # Arrange
    image = Image.new("RGB", (2, 1))
    image.putdata(((0, 0, 0), (255, 255, 255)))

    # Act
    planes = ImageCms.linearize_planes(image)

    # Assert
    assert [plane.hex() for plane in planes] == ["0000ffff", "0000ffff", "0000ffff"]


def test_linearize_mid_grey() -> None:
    # Arrange
    image = Image.new("L", (1, 1), 128)

    # Act
    (plane,) = ImageCms.linearize_planes(image)

    # Assert
    expected = int(((128 / 255 + 0.055) / 1.055) ** 2.4 * 65535 + 0.5)
    assert int.from_bytes(plane, "big") == expected


def test_apply_stages_is_deterministic() -> None:
    # Arrange
    image = hopper()
    srgb = ImageCms.createProfile("sRGB")

    # Act
    results = [
        ImageCms.applyStages(image, input_profile=srgb, output_profile=srgb).tobytes()
        for _ in range(2)
    ]

    # Assert
    assert results[0] == results[1]


def test_apply_stages_source_unchanged() -> None:
    # Arrange
    image = hopper()
    original = image.tobytes()
    srgb = ImageCms.createProfile("sRGB")

    # Act
    ImageCms.applyStages(image, input_profile=srgb, output_profile=srgb)

    # Assert
    assert image.tobytes() == original


def test_missing_profile_falls_back_to_srgb() -> None:
    # Arrange
    image = Image.new("RGB", (1, 1), (10, 20, 30))

    # Act
    result = ImageCms.applyStages(
        image,
        input_profile="/does/not/exist.icc",
        output_profile="/also/missing.icc",
    )

    # Assert
    assert result.mode == "RGB"
    assert result.size == image.size


def test_invalid_profile_detection() -> None:
    # Arrange / Act / Assert
    assert not ImageCms.validateProfile(b"definitely not a profile")
    assert not ImageCms.validateProfile("/does/not/exist.icc")
    assert ImageCms.validateProfile(ImageCms.createProfile("sRGB"))


def test_normalize_profile_fallback() -> None:
    # Arrange / Act
    profile = ImageCms.normalize_profile(None)

    # Assert
    assert profile.tobytes()


def test_repeat_transform_is_stable() -> None:
    # Arrange
    image = Image.new("RGB", (2, 1), (10, 20, 30))
    transform = ImageCms.buildTransform(
        ImageCms.createProfile("sRGB"), ImageCms.createProfile("sRGB"), "RGB", "RGB"
    )

    # Act
    once = ImageCms.repeatTransform(image, transform, 1)
    twice = ImageCms.repeatTransform(image, transform, 2)

    # Assert
    assert once.tobytes() == twice.tobytes() == image.tobytes()


def test_repeat_transform_in_place_source_unchanged() -> None:
    # Arrange
    image = Image.new("RGB", (2, 1), (10, 20, 30))
    original = image.tobytes()
    transform = ImageCms.buildTransform(
        ImageCms.createProfile("sRGB"), ImageCms.createProfile("sRGB"), "RGB", "RGB"
    )

    # Act
    ImageCms.repeatTransform(image, transform, 2, in_place=True)

    # Assert
    assert image.tobytes() == original


def test_stage_order() -> None:
    # Assert
    assert list(ImageCms.Stage) == [
        ImageCms.Stage.LINEARIZE,
        ImageCms.Stage.TRANSFORM,
        ImageCms.Stage.GAMMA,
        ImageCms.Stage.ENCODE,
    ]


def test_nan_and_infinity_are_deterministic() -> None:
    # Arrange / Act / Assert
    assert ImageCms._clamp_byte(float("nan")) == 0
    assert ImageCms._clamp_byte(float("inf")) == 255
    assert ImageCms._clamp_byte(float("-inf")) == 0


def test_gamma_stage_is_deterministic() -> None:
    # Arrange
    image = Image.new("RGB", (2, 1), (32, 64, 128))

    # Act
    first = ImageCms.applyStages(image, gamma=2.2, linear=True)
    second = ImageCms.applyStages(image, gamma=2.2, linear=True)

    # Assert
    assert first.tobytes() == second.tobytes()


def test_concurrent_transforms_do_not_share_state() -> None:
    # Arrange
    import concurrent.futures

    srgb = ImageCms.createProfile("sRGB")
    lab = ImageCms.createProfile("LAB")

    def convert(index: int) -> bytes:
        image = Image.new("RGB", (4, 1), (index, index, index))
        transform = ImageCms.buildTransform(srgb, lab, "RGB", "LAB")
        return transform.apply(image).tobytes()

    # Act
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(convert, range(8)))

    # Assert
    for index, result in enumerate(results):
        expected = convert(index)
        assert result == expected


def test_profile_roundtrip_bytes() -> None:
    # Arrange
    if not HAVE_CMS:
        pytest.skip("littlecms2 not available")
    profile = ImageCms.createProfile("sRGB")

    # Act
    data = ImageCms.ImageCmsProfile(profile).tobytes()

    # Assert
    assert data.startswith(b"\0\0")
    assert ImageCms.validateProfile(data)


def test_apply_stages_with_alpha_preserves_alpha() -> None:
    # Arrange
    image = Image.new("RGBA", (2, 1))
    image.putdata(((10, 20, 30, 0), (200, 100, 50, 255)))
    srgb = ImageCms.createProfile("sRGB")

    # Act
    result = ImageCms.applyStages(image, input_profile=srgb, output_profile=srgb)

    # Assert
    assert result.mode == "RGBA"
    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((1, 0))[3] == 255
