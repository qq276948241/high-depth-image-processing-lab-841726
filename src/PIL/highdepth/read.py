#
# The Python Imaging Library.
#
# Public high depth read entry points.
#
# See the README file for information on usage and redistribution.
#
"""Reading with :mod:`PIL.highdepth`.

``read`` decodes a single frame, ``read_frames`` decodes every frame.  Both
accept the same input forms — ``bytes``, a path, an open binary file object or
an iterator of chunks — and produce identical results for identical data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._bridge import detect_format, frame_count, pillow_decode
from .budget import CancellationToken, MemoryBudget, iter_tiles
from .errors import (
    UnsupportedFeatureError,
)
from .source import Source, open_source

if TYPE_CHECKING:
    from collections.abc import Iterator

    from PIL.highdepth._typing import SourceLike

    from .budget import Tile
    from .model import ImageData

PNG = "PNG"
TIFF = "TIFF"
BMP = "BMP"


def _decode(
    source: Source,
    format: str,
    page: int,
    budget: MemoryBudget | None,
    token: CancellationToken | None,
) -> ImageData:
    if format == "PNG":
        from .png import decode_png, decode_png_frame

        data = decode_png(source, budget=budget, token=token)
        if page == 0:
            return data
        if not data.info.get("apng_fdats"):
            msg = f"PNG has no page {page}; the stream is not animated"
            raise IndexError(msg)
        return decode_png_frame(data, page)
    if format == "BMP":
        if page != 0:
            msg = f"BMP is a single page format, got page {page}"
            raise IndexError(msg)
        from .bmp import decode_bmp

        return decode_bmp(source, budget=budget, token=token)
    if format == "TIFF":
        from .tiff import decode_tiff_page

        return decode_tiff_page(source, page, budget=budget, token=token)
    msg = f"unsupported format {format}"
    raise UnsupportedFeatureError(msg)


def read(
    source: SourceLike,
    *,
    format: str | None = None,
    page: int = 0,
    budget: MemoryBudget | None = None,
    token: CancellationToken | None = None,
    temp_dir: str | None = None,
) -> ImageData:
    """Decode one frame from any supported source.

    :param source: bytes, path, binary file object or iterable of chunks.
    :param format: force a container instead of sniffing the magic bytes.
    :param page: zero-based frame index for multi-page containers.
    :param budget: allocation budget; checked before large buffers.
    :param token: cooperative cancellation, polled per row or tile.
    :param temp_dir: directory used when spooling a non-seekable source.
    """
    if page < 0:
        msg = f"page must not be negative, got {page}"
        raise IndexError(msg)
    src = open_source(source, budget=budget, token=token, temp_dir=temp_dir)
    owns = src is not source
    try:
        detected = format.upper() if format else detect_format(src)
        return _decode(src, detected, page, budget, token)
    finally:
        if owns:
            src.close()


def read_frames(
    source: SourceLike,
    *,
    format: str | None = None,
    budget: MemoryBudget | None = None,
    token: CancellationToken | None = None,
    temp_dir: str | None = None,
) -> list[ImageData]:
    """Decode every frame of a container, preserving frame order."""
    src = open_source(source, budget=budget, token=token, temp_dir=temp_dir)
    owns = src is not source
    try:
        detected = format.upper() if format else detect_format(src)
        if detected == "PNG":
            from .png import decode_png, decode_png_frame

            first = decode_png(src, budget=budget, token=token)
            count = int(first.info.get("apng_frames", 1))
            frames = [first]
            for page in range(1, count):
                if token is not None:
                    token.raise_if_cancelled()
                frames.append(decode_png_frame(first, page))
            return frames
        count = frame_count(src, detected)
        return [_decode(src, detected, page, budget, token) for page in range(count)]
    finally:
        if owns:
            src.close()


def iter_frames(
    source: SourceLike,
    *,
    format: str | None = None,
    budget: MemoryBudget | None = None,
    token: CancellationToken | None = None,
    temp_dir: str | None = None,
) -> Iterator[ImageData]:
    """Yield frames lazily; the source stays open until the generator ends."""
    src = open_source(source, budget=budget, token=token, temp_dir=temp_dir)
    owns = src is not source
    try:
        detected = format.upper() if format else detect_format(src)
        for frame in read_frames(src, format=detected, budget=budget, token=token):
            if token is not None:
                token.raise_if_cancelled()
            yield frame
    finally:
        if owns:
            src.close()


def iter_image_tiles(
    data: ImageData,
    tile_height: int = 64,
    tile_width: int | None = None,
    *,
    token: CancellationToken | None = None,
) -> Iterator[tuple[Tile, ImageData]]:
    """Yield row/tile sized views of an already decoded image.

    Each tile is a fresh :class:`ImageData` sharing the bit depth, palette and
    transparency semantics of the source.  This is the streaming path used by
    consumers that cannot hold the whole image at once.
    """
    from array import array

    from .model import ImageData as _ImageData

    for tile in iter_tiles(
        data.width, data.height, tile_height, tile_width, token=token
    ):
        rows: list[int] = []
        for y in range(tile.y0, tile.y1):
            start = (y * data.width + tile.x0) * data.channels
            rows.extend(
                int(value)
                for value in data.samples[start : start + tile.width * data.channels]
            )
        samples: bytes | array[int]
        if data.storage_bits == 16:
            samples = array("H", rows)
        else:
            samples = bytes(rows)
        yield tile, _ImageData(
            format=data.format,
            mode=data.mode,
            width=tile.width,
            height=tile.height,
            bit_depth=data.bit_depth,
            storage_bits=data.storage_bits,
            channels=data.channels,
            samples=samples,
            palette=data.palette,
            transparency=data.transparency,
            row_stride=tile.width * data.channels * (data.storage_bits // 8),
            orientation=data.orientation,
            info=dict(data.info),
        )


def stream_decode(
    source: SourceLike,
    *,
    format: str | None = None,
    budget: MemoryBudget | None = None,
    token: CancellationToken | None = None,
    tile_height: int = 64,
    temp_dir: str | None = None,
) -> Iterator[tuple[Tile, ImageData]]:
    """Decode the first frame and yield tiles, checking the budget per tile.

    The header is fully validated before the pixel buffer is created, so a
    hostile file fails at open time rather than after allocating the image.
    """
    data = read(source, format=format, budget=budget, token=token, temp_dir=temp_dir)
    yield from iter_image_tiles(data, tile_height, token=token)


def pillow_read(source: SourceLike) -> ImageData:
    """Decode through Pillow and normalize, for comparison and fixtures."""
    src = open_source(source)
    owns = src is not source
    try:
        return pillow_decode(src, format=detect_format(src))
    finally:
        if owns:
            src.close()


def read_header(source: SourceLike, *, format: str | None = None) -> dict[str, object]:
    """Validate and describe an image without decoding its pixels."""
    src = open_source(source)
    owns = src is not source
    try:
        detected = format.upper() if format else detect_format(src)
        if detected == "PNG":
            from .png import _iter_chunks, _parse_ihdr

            chunks = _iter_chunks(src)
            width, height, depth, color, channels, interlace = _parse_ihdr(chunks[0])
            return {
                "format": "PNG",
                "size": (width, height),
                "bit_depth": depth,
                "channels": channels,
                "color_type": color,
                "interlaced": interlace == 1,
                "frames": frame_count(src, "PNG"),
            }
        if detected == "TIFF":
            from .tiff import tiff_page_count

            return {"format": "TIFF", "frames": tiff_page_count(src)}
        if detected == "BMP":
            from .bmp import decode_bmp

            data = decode_bmp(src)
            return {
                "format": "BMP",
                "size": (data.width, data.height),
                "bit_depth": data.bit_depth,
                "channels": data.channels,
                "frames": 1,
            }
        msg = f"unsupported format {detected}"
        raise UnsupportedFeatureError(msg)
    finally:
        if owns:
            src.close()


__all__ = [
    "BMP",
    "PNG",
    "TIFF",
    "iter_frames",
    "iter_image_tiles",
    "pillow_read",
    "read",
    "read_frames",
    "read_header",
    "stream_decode",
]
