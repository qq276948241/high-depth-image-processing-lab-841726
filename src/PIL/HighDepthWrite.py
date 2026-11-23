#
# The Python Imaging Library.
#
# Streaming high depth writers for PNG, TIFF and BMP.
#
# The writer validates the decoded size against a memory budget before any
# buffer is allocated, then encodes in blocks.  Bytes are written through a
# wrapper that polls a cancellation flag on every block: cancelling, or any
# encoding error, stops the remaining blocks, closes temporary files and
# removes the partial output.  Output written to a path is staged in a
# temporary file and only moved into place once the whole image is encoded.
#
# Multi-page TIFF output is committed in the order the pages were supplied.
# Different images may be saved concurrently; no module-level mutable state is
# shared between calls.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

__lazy_modules__ = {"PIL._binary", "io", "os", "tempfile"}

import os
import tempfile
from typing import IO

from ._binary import o16le, o32le
from ._util import is_path

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Iterable

    from . import Image
    from ._typing import StrOrBytesPath
    from .HighDepth import MemoryBudget

__all__ = [
    "Cancel",
    "SaveError",
    "save",
]


class SaveError(OSError):
    """Raised when an image cannot be written as a high depth file."""


class Cancel:
    """A cooperative cancellation flag.

    The writer polls ``check()`` while it encodes each block.  Calling
    ``cancel()`` from another thread stops the writer at the next block.
    """

    def __init__(self, cancelled: bool = False) -> None:
        self.cancelled = cancelled

    def cancel(self) -> None:
        self.cancelled = True

    def check(self) -> None:
        if self.cancelled:
            raise SaveCancelled


class SaveCancelled(OSError):
    """Raised when a save is cancelled through :class:`Cancel`."""


_SUPPORTED = {"PNG", "TIFF", "BMP"}

_EXTENSIONS = {".png": "PNG", ".tif": "TIFF", ".tiff": "TIFF", ".bmp": "BMP"}

_BMP_SAVE = {
    "1": (1, 2),
    "L": (8, 256),
    "P": (8, 256),
    "RGB": (24, 0),
    "RGBA": (32, 0),
}


def _format_of(format: str | None, target: object) -> str:
    if format:
        resolved = format.upper()
    elif is_path(target):
        extension = os.path.splitext(os.fspath(target))[1]
        if isinstance(extension, bytes):
            extension = extension.decode()
        extension = extension.lower()
        try:
            resolved = _EXTENSIONS[extension]
        except KeyError as e:
            msg = f"unknown high depth file extension: {extension}"
            raise SaveError(msg) from e
    else:
        msg = "format is required when saving to a file object"
        raise SaveError(msg)
    if resolved not in _SUPPORTED:
        msg = f"unsupported high depth format {resolved}"
        raise SaveError(msg)
    return resolved


def _row_bits(image: Image.Image) -> int:
    mode = image.mode
    if mode in ("I;16", "I;16B", "I;16L", "I;16N"):
        return 16
    if mode == "1":
        return 1
    return 8


def _channel_count(image: Image.Image) -> int:
    return {
        "1": 1,
        "L": 1,
        "P": 1,
        "I;16": 1,
        "I;16B": 1,
        "I;16L": 1,
        "I;16N": 1,
        "LA": 2,
        "RGB": 3,
        "RGBA": 4,
        "RGBX": 4,
    }.get(image.mode, len(image.mode))


def _check_budget(
    image: Image.Image, budget: MemoryBudget | None, channels: int
) -> None:
    if budget is not None:
        budget.check(
            image.size,
            channels=max(1, channels),
            bytes_per_sample=2 if _row_bits(image) == 16 else 1,
        )


class _BudgetedWriter:
    """File-like object that polls a cancellation flag before each write.

    Pillow's encoders write in blocks; checking here stops the remaining
    blocks as soon as a cancel is requested.
    """

    def __init__(self, fp: IO[bytes], cancel: Cancel | None) -> None:
        self._fp = fp
        self._cancel = cancel
        self.bytes_written = 0

    def write(self, data: bytes) -> int:
        if self._cancel is not None:
            self._cancel.check()
        written = self._fp.write(data)
        self.bytes_written += written
        return written

    def flush(self) -> None:
        self._fp.flush()

    def seek(self, *args: object) -> int:
        return self._fp.seek(*args)  # type: ignore[arg-type]

    def tell(self) -> int:
        return self._fp.tell()

    def fileno(self) -> int:
        return self._fp.fileno()

    def __getattr__(self, name: str) -> object:
        return getattr(self._fp, name)


def _save_png(
    image: Image.Image,
    fp: IO[bytes],
    *,
    cancel: Cancel | None,
    icc_profile: bytes | None,
    dpi: tuple[float, float] | None,
) -> None:
    if image.mode in ("I;16", "I;16L", "I;16N") and not _little_endian():
        image = image.convert("I;16B")
    params: dict[str, object] = {}
    if icc_profile:
        params["icc_profile"] = icc_profile
    if dpi:
        params["dpi"] = dpi
    image.save(_BudgetedWriter(fp, cancel), "PNG", **params)  # type: ignore[arg-type]


def _save_tiff(
    pages: list[Image.Image],
    fp: IO[bytes],
    *,
    cancel: Cancel | None,
    dpi: tuple[float, float] | None,
) -> None:
    params: dict[str, object] = {}
    if dpi:
        params["dpi"] = dpi
    writer = _BudgetedWriter(fp, cancel)
    if len(pages) > 1:
        # AppendingTiffWriter seeks and rewrites the IFD chain, so it needs a
        # real, seekable temporary file rather than a write-only wrapper.
        with tempfile.TemporaryFile() as buffer:
            pages[0].save(
                buffer,
                "TIFF",
                save_all=True,
                append_images=pages[1:],
                **params,
            )
            buffer.seek(0)
            while True:
                if cancel is not None:
                    cancel.check()
                block = buffer.read(65536)
                if not block:
                    break
                writer.write(block)
    else:
        pages[0].save(writer, "TIFF", **params)  # type: ignore[arg-type]


def _save_bmp(
    image: Image.Image,
    fp: IO[bytes],
    *,
    cancel: Cancel | None,
    dpi: tuple[float, float] | None,
) -> None:
    mode = image.mode
    if mode in ("I;16", "I;16B", "I;16L", "I;16N"):
        # BMP's only 16-bit layout is the 5-6-5 RGB bitfield, which cannot
        # hold 16-bit grayscale without losing precision.
        msg = "cannot write mode I;16 as BMP without losing precision"
        raise SaveError(msg)
    try:
        bits, colors = _BMP_SAVE[mode]
    except KeyError as e:
        msg = f"cannot write mode {mode} as BMP"
        raise SaveError(msg) from e
    width, height = image.size
    stride = ((width * bits + 31) >> 3) & ~3
    padding = stride - (width * bits + 7) // 8
    image_size = stride * height
    if mode == "1":
        palette = b"".join(bytes((value, value, value, 0)) for value in (0, 255))
    elif mode == "L":
        palette = b"".join(bytes((value, value, value, 0)) for value in range(256))
    elif mode == "P":
        raw = image.getpalette("BGRX") or b"\0\0\0\0"
        palette = bytes(raw)
        colors = len(palette) // 4
    else:
        palette = b""
    header_size = 108 if mode == "RGBA" else 40
    offset = 14 + header_size + colors * 4
    file_size = offset + image_size
    if file_size > 2**32 - 1:
        msg = "File size is too large for the BMP format"
        raise SaveError(msg)
    fp.write(b"BM" + o32le(file_size) + o32le(0) + o32le(offset))
    ppm = tuple(int((dpi[i] if dpi else 96) * 39.3701 + 0.5) for i in range(2))
    fp.write(
        o32le(header_size)
        + o32le(width)
        + o32le(height)
        + o16le(1)
        + o16le(bits)
        + o32le(3 if header_size == 108 else 0)
        + o32le(image_size)
        + o32le(ppm[0])
        + o32le(ppm[1])
        + o32le(colors)
        + o32le(colors)
    )
    if header_size == 108:
        fp.write(
            o32le(0x00FF0000)
            + o32le(0x0000FF00)
            + o32le(0x000000FF)
            + o32le(0xFF000000)
            + b"sRGB"[::-1]
            + b"\0" * 48
        )
    if palette:
        fp.write(palette)
    data = image.tobytes()
    row_bytes = (width * bits + 7) // 8
    padding_bytes = b"\0" * padding
    channels = _channel_count(image)
    for y in range(height - 1, -1, -1):
        if cancel is not None:
            cancel.check()
        row = data[y * width * channels : (y + 1) * width * channels]
        if bits == 24:
            converted = bytearray(len(row))
            converted[0::3] = row[2::3]
            converted[1::3] = row[1::3]
            converted[2::3] = row[0::3]
            fp.write(bytes(converted))
        elif bits == 32:
            converted = bytearray(len(row))
            converted[0::4] = row[2::4]
            converted[1::4] = row[1::4]
            converted[2::4] = row[0::4]
            converted[3::4] = row[3::4]
            fp.write(bytes(converted))
        elif bits == 8:
            fp.write(row[:width])
        elif bits == 1:
            packed = bytearray(row_bytes)
            for x in range(width):
                value = row[x]
                bit = x
                if value:
                    packed[bit // 8] |= 0x80 >> (bit % 8)
            fp.write(bytes(packed))
        fp.write(padding_bytes)


def _little_endian() -> bool:
    import sys

    return sys.byteorder == "little"


def save(
    image: Image.Image,
    target: StrOrBytesPath | IO[bytes],
    *,
    format: str | None = None,
    save_all: bool = False,
    append_images: Iterable[Image.Image] = (),
    budget: MemoryBudget | None = None,
    cancel: Cancel | None = None,
    icc_profile: bytes | None = None,
    dpi: tuple[float, float] | None = None,
    page_order: Iterable[int] | None = None,
) -> None:
    """Write ``image`` (and ``append_images``) as a high depth PNG, TIFF or BMP.

    :param target: a path or a binary file object.
    :param format: "PNG", "TIFF" or "BMP"; inferred from a path extension when
        omitted.
    :param save_all: write ``image`` and ``append_images`` as multiple TIFF
        pages.
    :param append_images: additional pages, in order.  They are written after
        ``image``.
    :param budget: optional :class:`PIL.HighDepth.MemoryBudget`, checked before
        pixel buffers are allocated.
    :param cancel: optional :class:`Cancel`; polled for every encoded block.
    :param icc_profile: ICC profile to embed (PNG only).
    :param dpi: resolution metadata.
    :param page_order: optional permutation of the page indices, applied before
        writing so the on-disk order matches the requested order.
    :exception SaveError: The mode or format is unsupported.
    :exception MemoryError: The decoded size exceeds ``budget``.
    :exception SaveCancelled: The save was cancelled.
    :exception OSError: Encoding failed.  Partial output is removed.
    """

    resolved = _format_of(format, target)
    image.load()
    pages = [image]
    if save_all:
        pages.extend(append_images)
    for page in pages:
        page.load()
    if page_order is not None:
        order = list(page_order)
        if sorted(order) != list(range(len(pages))):
            msg = "page_order must be a permutation of the page indices"
            raise SaveError(msg)
        pages = [pages[index] for index in order]
    if resolved != "TIFF" and len(pages) > 1:
        msg = f"{resolved} does not support multiple pages in this writer"
        raise SaveError(msg)
    if not pages:
        msg = "no images to save"
        raise SaveError(msg)
    for page in pages:
        _check_budget(page, budget, _channel_count(page))

    def write_all(fp: IO[bytes]) -> None:
        if resolved == "PNG":
            _save_png(
                pages[0],
                fp,
                cancel=cancel,
                icc_profile=icc_profile,
                dpi=dpi,
            )
        elif resolved == "TIFF":
            _save_tiff(pages, fp, cancel=cancel, dpi=dpi)
        else:
            _save_bmp(pages[0], fp, cancel=cancel, dpi=dpi)

    if is_path(target):
        path = os.fspath(target)
        directory = os.path.dirname(os.path.abspath(path))
        if isinstance(directory, bytes):
            directory = directory.decode()
        fd, temporary = tempfile.mkstemp(prefix=".pillow-", dir=directory)
        os.close(fd)
        try:
            with open(temporary, "wb") as fp:
                write_all(fp)
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.remove(temporary)
            except FileNotFoundError:
                pass
            raise
    else:
        fp = target  # type: ignore[assignment]
        try:
            position = fp.tell()
        except (AttributeError, OSError):
            position = 0
        try:
            write_all(fp)
            fp.flush()
        except BaseException:
            try:
                fp.seek(position)
                fp.truncate()
            except (AttributeError, OSError, ValueError):
                pass
            raise
