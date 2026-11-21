#
# The Python Imaging Library.
#
# Bridge between PIL.highdepth and the classic Pillow entry points.
#
# See the README file for information on usage and redistribution.
#
"""Compatibility bridge for existing Pillow entry points.

Nothing in this module changes the behaviour of ``Image.open``,
``Image.save``, ``ImageFile.ImageFile`` or ``ImageCms``; it adds opt-in
helpers that share the high depth model:

* :func:`open_normalized` — ``Image.open`` then
  :meth:`ImageData.from_pil`, preserving depth where Pillow can;
* :func:`save_normalized` — :func:`ImageData.from_pil` then
  :func:`PIL.highdepth.encode.write_frames`, so ``I;16`` gray and palette
  alpha keep their meaning;
* :func:`apply_icc` — run an :class:`ImageCms` transform through the pipeline
  without touching either the image or the profile on failure;
* :func:`quantize_image` — route ``Image.Image.quantize``-style requests to the
  deterministic quantizer;
* :func:`read_pillow_frames` — iterate ``ImageSequence`` frames exactly as the
  classic plugins expose them, in the normalized model.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

from .errors import BitDepthError, ProfileError, UnsupportedFeatureError
from .model import ImageData
from .quantize import quantize

if TYPE_CHECKING:
    from typing import IO

    from PIL import Image, ImageCms
    from PIL.highdepth._typing import SourceLike, StrOrBytesPath

    from .budget import CancellationToken, MemoryBudget
    from .cms import ColorPipeline


def _pillow_source(fp: SourceLike) -> Any:
    """Adapt a high-depth source to something ``Image.open`` accepts."""
    if isinstance(fp, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(fp))
    return fp


def open_normalized(
    fp: SourceLike,
    *,
    formats: list[str] | None = None,
    bit_depth: int | None = None,
) -> ImageData:
    """``Image.open`` followed by :meth:`ImageData.from_pil`.

    ``bit_depth`` is only honoured for palette images, because PIL does not
    record whether a palette was stored with 4 or 8 bits per index.
    """
    from PIL import Image

    im = Image.open(_pillow_source(fp), formats=formats)
    with im:
        im.load()
        return ImageData.from_pil(im, format=im.format, bit_depth=bit_depth)


def read_pillow_frames(
    fp: SourceLike, *, bit_depth: int | None = None
) -> list[ImageData]:
    """Normalize every frame of an animated Pillow image, in order."""
    from PIL import Image, ImageSequence

    if isinstance(fp, (bytes, bytearray, memoryview)):
        # ``Image.open`` interprets raw bytes as a filename, so wrap them.
        fp = io.BytesIO(bytes(fp))
    im = Image.open(_pillow_source(fp))
    frames: list[ImageData] = []
    with im:
        for frame in ImageSequence.Iterator(im):
            # ``seek`` mutates the shared image, so snapshot the pixels now.
            snapshot = frame.copy()
            frames.append(
                ImageData.from_pil(snapshot, format=im.format, bit_depth=bit_depth)
            )
    return frames


def save_normalized(
    data: ImageData,
    fp: IO[bytes],
    format: str,
    **options: Any,
) -> None:
    """Write one normalized frame through the strict encoders."""
    from .encode import write_frames

    write_frames(None, [data], format, fp=fp, **options)


def quantize_image(
    im: Image.Image,
    colors: int = 256,
    *,
    method: str = "median",
    dither: bool = True,
    seed: int = 0,
    palette: Image.Image | None = None,
) -> Image.Image:
    """Deterministic drop-in for ``Image.Image.quantize``.

    Returns a Pillow ``P`` image with the same palette the quantizer built, so
    callers can inspect and re-use it.
    """
    data = ImageData.from_pil(im)
    if palette is not None:
        if palette.mode != "P":
            msg = "palette image must be mode P"
            raise UnsupportedFeatureError(msg)
        palette_mode = palette.palette.mode if palette.palette is not None else "RGB"
        palette_data = palette.getpalette(palette_mode) or palette.getpalette("RGB")
        if not palette_data:
            msg = "empty palette"
            raise UnsupportedFeatureError(msg)
        from .model import Palette

        if palette_mode == "RGBA" and len(palette_data) % 4 == 0:
            fixed = Palette.from_rgba(bytes(palette_data))
        else:
            fixed = Palette.from_rgb(bytes(palette_data))
        result = quantize(data, dither=dither, seed=seed, palette=fixed)
    else:
        result = quantize(data, colors, method=method, dither=dither, seed=seed)
    return result.to_pil()


def apply_icc(
    im: Image.Image,
    transform: ImageCms.ImageCmsTransform,
    *,
    in_place: bool = False,
) -> Image.Image:
    """Apply an :class:`ImageCms` transform with failure isolation.

    On error, :class:`ProfileError` is raised and the input image is left
    exactly as it was (``in_place`` semantics are simulated by only replacing
    the caller's buffer after a successful transform).
    """
    from io import BytesIO

    from PIL import Image

    if im.mode != transform.input_mode:
        msg = (
            f"image mode {im.mode!r} does not match the transform input mode "
            f"{transform.input_mode!r}"
        )
        raise ProfileError(msg)
    snapshot = im.tobytes()
    try:
        output = transform.apply(im.copy() if in_place else im)
    except Exception as exc:
        if in_place:
            # ``apply`` never mutates its argument unless it is also the target.
            if im.tobytes() != snapshot:  # pragma: no cover - defensive
                im.frombytes(snapshot, "raw", im.mode)
        msg = f"ICC transform failed: {exc}"
        raise ProfileError(msg) from exc
    if in_place:
        if output.mode != im.mode:
            msg = "in-place ICC output must keep the input mode"
            raise ProfileError(msg)
        # Replace the image core in place.  ``frombytes`` is not usable here
        # because it takes raw data, not another image.
        im.im = output.copy().im
        im.palette = output.palette
        im.info.update(output.info)
        return im
    del BytesIO
    del Image
    return output


def apply_pipeline(im: Image.Image, pipeline: ColorPipeline) -> Image.Image:
    """Run a :class:`ColorPipeline` on a Pillow image and return a Pillow image."""
    data = ImageData.from_pil(im)
    result = pipeline.apply(data)
    try:
        return result.to_pil()
    except BitDepthError:
        return result.to_pil(allow_reduce=True)


def to_high_depth(im: Image.Image, *, budget: MemoryBudget | None = None) -> ImageData:
    """Normalize a Pillow image, refusing implausibly large buffers first."""
    if budget is not None:
        budget.check(
            len(im.tobytes()),
            "Pillow image",
        )
    return ImageData.from_pil(im)


def from_high_depth(data: ImageData, *, allow_reduce: bool = False) -> Image.Image:
    return data.to_pil(allow_reduce=allow_reduce)


def stream_to_pillow(
    data: ImageData, tile_height: int = 64, token: CancellationToken | None = None
) -> list[Image.Image]:
    """Split a normalized image into row tiles, each as a Pillow image."""
    from .read import iter_image_tiles

    return [
        tile.to_pil()
        for _bounds, tile in iter_image_tiles(data, tile_height, token=token)
    ]


def save_path(
    data: ImageData,
    path: StrOrBytesPath,
    format: str,
    **options: Any,
) -> None:
    """Atomic single-frame write to a path."""
    from .encode import write_frames

    write_frames(path, [data], format, **options)


__all__ = [
    "apply_icc",
    "apply_pipeline",
    "from_high_depth",
    "open_normalized",
    "quantize_image",
    "read_pillow_frames",
    "save_normalized",
    "save_path",
    "stream_to_pillow",
    "to_high_depth",
]
