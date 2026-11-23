from __future__ import annotations

import io
import os

import pytest

from PIL import HighDepth, HighDepthWrite, Image

from .helper import assert_image_equal

TYPE_CHECKING = False
if TYPE_CHECKING:
    from pathlib import Path


def _i16(values: tuple[int, ...], size: tuple[int, int] | None = None) -> Image.Image:
    width = size[0] if size else len(values)
    height = len(values) // width
    image = Image.new("I;16", (width, height))
    image.putdata(values)
    return image


def test_png_roundtrip_16_bit() -> None:
    # Arrange
    image = _i16((0, 1000, 65535, 5))

    # Act
    out = io.BytesIO()
    HighDepthWrite.save(image, out, format="PNG")
    data = out.getvalue()
    with Image.open(io.BytesIO(data)) as reloaded:
        decoded = HighDepth.decode(data)

        # Assert
        assert reloaded.mode == "I;16"
        assert_image_equal(image, reloaded)
    assert decoded.planes[0] == b"\x00\x00\x03\xe8\xff\xff\x00\x05"


def test_tiff_roundtrip_16_bit() -> None:
    # Arrange
    image = _i16((0, 1000, 65535, 5))

    # Act
    out = io.BytesIO()
    HighDepthWrite.save(image, out, format="TIFF")
    out.seek(0)
    with Image.open(out) as reloaded:
        reloaded.load()
        original = image.tobytes()
        result = reloaded.tobytes()

    # Assert
    assert result == original


def test_bmp_roundtrip_rgba() -> None:
    # Arrange
    image = Image.new("RGBA", (2, 2))
    image.putdata(((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16)))

    # Act
    out = io.BytesIO()
    HighDepthWrite.save(image, out, format="BMP")
    out.seek(0)
    with Image.open(out) as reloaded:
        reloaded.load()

    # Assert
    assert reloaded.mode == "RGBA"
    assert reloaded.tobytes() == image.tobytes()


def test_bmp_roundtrip_palette() -> None:
    # Arrange
    image = Image.new("P", (2, 2))
    image.putpalette(bytes((255, 0, 0, 0, 255, 0, 0, 0, 255)))
    image.putdata((0, 1, 2, 0))

    # Act
    out = io.BytesIO()
    HighDepthWrite.save(image, out, format="BMP")
    out.seek(0)
    with Image.open(out) as reloaded:
        reloaded.load()

    # Assert
    assert reloaded.mode == "P"
    assert reloaded.tobytes() == image.tobytes()
    assert reloaded.getpalette()[:9] == [255, 0, 0, 0, 255, 0, 0, 0, 255]


def test_multipage_order(tmp_path: Path) -> None:
    # Arrange
    first = _i16((1, 2, 3, 4), (2, 2))
    second = _i16((5, 6, 7, 8), (2, 2))
    path = tmp_path / "pages.tif"

    # Act
    HighDepthWrite.save(
        first, path, format="TIFF", save_all=True, append_images=[second]
    )

    # Assert
    with Image.open(path) as reloaded:
        assert reloaded.n_frames == 2
        reloaded.seek(0)
        assert reloaded.tobytes() == first.tobytes()
        reloaded.seek(1)
        assert reloaded.tobytes() == second.tobytes()


def test_multipage_page_order(tmp_path: Path) -> None:
    # Arrange
    first = _i16((1, 2, 3, 4), (2, 2))
    second = _i16((5, 6, 7, 8), (2, 2))
    path = tmp_path / "ordered.tif"

    # Act
    HighDepthWrite.save(
        first,
        path,
        format="TIFF",
        save_all=True,
        append_images=[second],
        page_order=[1, 0],
    )

    # Assert
    with Image.open(path) as reloaded:
        reloaded.seek(0)
        assert reloaded.tobytes() == second.tobytes()
        reloaded.seek(1)
        assert reloaded.tobytes() == first.tobytes()


def test_multipage_rejects_non_tiff() -> None:
    # Arrange
    image = _i16((1, 2, 3, 4), (2, 2))

    # Act / Assert
    with pytest.raises(HighDepthWrite.SaveError, match="multiple pages"):
        HighDepthWrite.save(
            image, io.BytesIO(), format="PNG", save_all=True, append_images=[image]
        )


def test_page_order_must_be_permutation() -> None:
    # Arrange
    image = _i16((1, 2, 3, 4), (2, 2))

    # Act / Assert
    with pytest.raises(HighDepthWrite.SaveError, match="permutation"):
        HighDepthWrite.save(
            image,
            io.BytesIO(),
            format="TIFF",
            save_all=True,
            append_images=[image],
            page_order=[0, 0],
        )


def test_budget_rejects_before_write() -> None:
    # Arrange
    image = _i16(tuple(range(4096)), (64, 64))
    out = io.BytesIO()

    # Act / Assert
    with pytest.raises(MemoryError):
        HighDepthWrite.save(
            image, out, format="PNG", budget=HighDepth.MemoryBudget(1024)
        )
    assert out.getvalue() == b""


def test_cancel_removes_partial_file(tmp_path: Path) -> None:
    # Arrange
    image = Image.new("RGB", (256, 256), (1, 2, 3))
    path = tmp_path / "cancelled.png"
    cancel = HighDepthWrite.Cancel(True)

    # Act / Assert
    with pytest.raises(HighDepthWrite.SaveCancelled):
        HighDepthWrite.save(image, path, format="PNG", cancel=cancel)
    assert not path.exists()
    assert os.listdir(tmp_path) == []


def test_cancel_truncates_file_object() -> None:
    # Arrange
    image = Image.new("RGB", (256, 256), (1, 2, 3))
    out = io.BytesIO()
    out.write(b"existing")
    out.seek(8)
    cancel = HighDepthWrite.Cancel(True)

    # Act / Assert
    with pytest.raises(HighDepthWrite.SaveCancelled):
        HighDepthWrite.save(image, out, format="PNG", cancel=cancel)
    assert out.getvalue() == b"existing"


def test_cancel_between_blocks() -> None:
    # Arrange
    image = Image.new("RGB", (512, 512), (9, 9, 9))
    out = io.BytesIO()

    class CancelAfterFirstWrite(HighDepthWrite.Cancel):
        def __init__(self) -> None:
            super().__init__()
            self.writes = 0

        def check(self) -> None:
            self.writes += 1
            if self.writes > 1:
                self.cancelled = True
            super().check()

    cancel = CancelAfterFirstWrite()

    # Act / Assert
    with pytest.raises(HighDepthWrite.SaveCancelled):
        HighDepthWrite.save(image, out, format="PNG", cancel=cancel)
    assert out.getvalue() == b""


def test_path_is_written_atomically(tmp_path: Path) -> None:
    # Arrange
    image = _i16((1, 2, 3, 4), (2, 2))
    path = tmp_path / "atom.png"
    path.write_bytes(b"old")

    # Act
    HighDepthWrite.save(image, path)

    # Assert
    with Image.open(path) as reloaded:
        assert reloaded.mode == "I;16"
    assert [entry.name for entry in tmp_path.iterdir()] == ["atom.png"]


def test_format_required_for_file_object() -> None:
    # Arrange
    image = _i16((1, 2, 3, 4), (2, 2))

    # Act / Assert
    with pytest.raises(HighDepthWrite.SaveError, match="format is required"):
        HighDepthWrite.save(image, io.BytesIO())


def test_unsupported_format() -> None:
    # Arrange
    image = _i16((1, 2, 3, 4), (2, 2))

    # Act / Assert
    with pytest.raises(HighDepthWrite.SaveError, match="unsupported"):
        HighDepthWrite.save(image, io.BytesIO(), format="GIF")


def test_unsupported_mode_for_bmp() -> None:
    # Arrange
    image = _i16((1, 2, 3, 4), (2, 2))

    # Act / Assert: BMP's only 16-bit layout is lossy RGB 5-6-5, so the writer
    # refuses instead of silently dropping precision.
    with pytest.raises(HighDepthWrite.SaveError, match="losing precision"):
        HighDepthWrite.save(image, io.BytesIO(), format="BMP")


def test_png_transparency_roundtrip() -> None:
    # Arrange
    image = Image.new("P", (2, 1))
    image.putpalette(bytes((255, 0, 0, 0, 255, 0)))
    image.putdata((0, 1))
    image.info["transparency"] = bytes((128,))

    # Act
    out = io.BytesIO()
    HighDepthWrite.save(image, out, format="PNG")
    out.seek(0)
    with Image.open(out) as reloaded:
        rgba = reloaded.convert("RGBA")

    # Assert
    assert reloaded.info["transparency"] == bytes((128,))
    assert rgba.getpixel((0, 0)) == (255, 0, 0, 128)
    assert rgba.getpixel((1, 0)) == (0, 255, 0, 255)


def test_png_icc_profile_roundtrip() -> None:
    # Arrange
    profile = b"fake profile"
    image = Image.new("RGB", (1, 1), (1, 2, 3))
    image.info["icc_profile"] = profile

    # Act
    out = io.BytesIO()
    HighDepthWrite.save(image, out, format="PNG", icc_profile=profile)
    out.seek(0)
    with Image.open(out) as reloaded:
        result = reloaded.info.get("icc_profile")

    # Assert
    assert result == profile


def test_source_is_unchanged() -> None:
    # Arrange
    image = _i16((1, 2, 3, 4), (2, 2))
    original = image.tobytes()

    # Act
    HighDepthWrite.save(image, io.BytesIO(), format="TIFF")

    # Assert
    assert image.tobytes() == original


def test_save_writer_does_not_leak_temporary_files(tmp_path: Path) -> None:
    # Arrange
    image = _i16((1, 2, 3, 4), (2, 2))

    # Act
    for index in range(4):
        HighDepthWrite.save(image, tmp_path / f"image{index}.png")

    # Assert
    assert len(list(tmp_path.iterdir())) == 4
