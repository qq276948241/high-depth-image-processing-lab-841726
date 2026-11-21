#
# The Python Imaging Library.
#
# Bridge helpers shared by the public high depth entry points.
#
# See the README file for information on usage and redistribution.
#
"""Detection and Pillow fallback helpers."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from .errors import FormatError, UnsupportedFeatureError
from .model import ImageData
from .source import Source, open_source

if TYPE_CHECKING:
    from .budget import CancellationToken, MemoryBudget

_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"BM", "BMP"),
    (b"II\x2a\x00", "TIFF"),
    (b"MM\x00\x2a", "TIFF"),
    (b"II\x2b\x00", "TIFF"),
    (b"MM\x00\x2b", "TIFF"),
)


def detect_format(source: Source) -> str:
    """Identify a supported container from its magic bytes."""
    header = source.read_at(0, 16)
    for signature, name in _SIGNATURES:
        if header.startswith(signature):
            return name
    msg = "unrecognized image signature"
    raise FormatError(msg)


def pillow_decode(
    source: Source,
    *,
    format: str | None = None,
    budget: MemoryBudget | None = None,
    token: CancellationToken | None = None,
) -> ImageData:
    """Decode with Pillow and normalize the result.

    Used as a fallback when the strict native decoder does not implement a
    variant (for example APNG animation frames).  The original bytes are
    handed to Pillow; the source keeps them, so no second copy is made.
    """
    from PIL import Image, ImageFile

    if token is not None:
        token.raise_if_cancelled()
    data = source.bytes()
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(io.BytesIO(data), formats=[format] if format else None) as im:
            bit_depth = im.info.get("bit_depth")
            normalized = ImageData.from_pil(
                im, format=format or im.format, bit_depth=bit_depth
            )
            if token is not None:
                token.raise_if_cancelled()
            return normalized
    except (SyntaxError, OSError, ValueError, Image.DecompressionBombError) as exc:
        msg = f"Pillow could not decode the {format or 'image'} stream: {exc}"
        raise FormatError(msg) from exc
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def frame_count(source: Source, format: str) -> int:
    """Return the number of frames in a container without decoding pixels."""
    if format == "TIFF":
        from .tiff import tiff_page_count

        return tiff_page_count(source)
    if format == "PNG":
        # The strict decoder records the declared APNG frame count in info.
        from .png import decode_png

        return int(decode_png(source).info.get("apng_frames", 1))
    return 1


def require_frame_support(format: str) -> None:
    if format not in {"PNG", "TIFF", "BMP"}:
        msg = f"multi-frame writing is not implemented for {format}"
        raise UnsupportedFeatureError(msg)


def source_format(source: object) -> tuple[Source, str, bool]:
    """Open ``source`` and detect its format; returns ownership as well."""
    src = open_source(source)
    owns = src is not source
    try:
        return src, detect_format(src), owns
    except BaseException:
        if owns:
            src.close()
        raise


__all__ = [
    "detect_format",
    "frame_count",
    "pillow_decode",
    "require_frame_support",
    "source_format",
]
