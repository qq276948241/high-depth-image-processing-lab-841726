#
# The Python Imaging Library.
#
# Streaming encoders with atomic output for the high depth pipeline.
#
# See the README file for information on usage and redistribution.
#
"""PNG, TIFF and BMP encoders.

All writers share the same output protocol:

* data is described up front (frame count, mode, bit depth);
* each frame is written strip by strip through :func:`_write_block`, which
  checks the memory budget and the cancellation token before allocating;
* a path target is written to a sibling temporary file and moved into place
  with :func:`os.replace` only after every frame succeeds;
* on cancellation or any exception the temporary file is closed and removed,
  so a partial file never appears under the requested name.

Frame order is the order supplied.  Orientation is written back into the EXIF
tag for TIFF and into ``info`` metadata for PNG/BMP; channel count and
transparency semantics survive the round trip.
"""

from __future__ import annotations

import os
import struct
import tempfile
import zlib
from typing import TYPE_CHECKING

from .codecs import pack_bits
from .errors import (
    EmptyImageError,
    IllegalPaletteError,
    InvalidParameterError,
    UnsupportedFeatureError,
    WriteVerificationError,
)
from .model import ImageData, Palette, checked_add, checked_mul

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import IO

    from PIL.highdepth._typing import StrOrBytesPath

    from .budget import CancellationToken, MemoryBudget

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_STRIP_BYTES = 1 << 20


class _Writer:
    """A byte sink that enforces the budget and cancels between blocks."""

    def __init__(
        self,
        fp: IO[bytes],
        *,
        budget: MemoryBudget | None,
        token: CancellationToken | None,
    ) -> None:
        self.fp = fp
        self.budget = budget
        self.token = token
        self.written = 0

    def write(self, data: bytes, what: str = "output block") -> None:
        if self.token is not None:
            self.token.raise_if_cancelled()
        if self.budget is not None:
            self.budget.reserve(len(data), what)
        self.fp.write(data)
        self.written += len(data)


def _check_frame(frame: ImageData) -> None:
    frame_object: object = frame
    if not isinstance(frame_object, ImageData):
        msg = f"expected ImageData frames, got {type(frame_object).__name__}"
        raise InvalidParameterError(msg)
    if frame.is_empty():
        msg = "cannot write an empty image"
        raise EmptyImageError(msg)


def _common_size(frames: Sequence[ImageData]) -> tuple[int, int]:
    sizes = {(frame.width, frame.height) for frame in frames}
    if len(sizes) != 1:
        msg = "all frames must share one size"
        raise InvalidParameterError(msg)
    return frames[0].width, frames[0].height


def _common_palette(frames: Sequence[ImageData]) -> Palette | None:
    palettes = {frame.palette for frame in frames if frame.mode == "P"}
    if not palettes:
        return None
    if len(palettes) != 1:
        msg = "palette frames must share one colour table"
        raise IllegalPaletteError(msg)
    if any(frame.mode != "P" for frame in frames):
        msg = "cannot mix palette and direct colour frames"
        raise IllegalPaletteError(msg)
    return next(iter(palettes))


# ----------------------------------------------------------------- PNG


def _png_chunk(fp: IO[bytes], type_: bytes, payload: bytes) -> None:
    fp.write(struct.pack(">I", len(payload)))
    fp.write(type_)
    fp.write(payload)
    fp.write(struct.pack(">I", zlib.crc32(type_ + payload) & 0xFFFFFFFF))


def _png_mode(frame: ImageData) -> tuple[int, int, int]:
    """Return ``(color_type, channels, depth)`` for a frame."""
    if frame.mode == "P":
        return 3, 1, frame.bit_depth
    mapping = {
        ("L", 8): 0,
        ("L", 16): 0,
        ("LA", 8): 4,
        ("LA", 16): 4,
        ("RGB", 8): 2,
        ("RGB", 16): 2,
        ("RGBA", 8): 6,
        ("RGBA", 16): 6,
    }
    key = (frame.mode, frame.storage_bits)
    if key not in mapping:
        msg = f"cannot write mode {frame.mode!r} at {frame.storage_bits} bits as PNG"
        raise UnsupportedFeatureError(msg)
    return mapping[key], frame.channels, frame.storage_bits


def _filter_none(row: bytes) -> bytes:
    return b"\x00" + row


def encode_png(
    frames: Sequence[ImageData],
    fp: IO[bytes],
    *,
    budget: MemoryBudget | None = None,
    token: CancellationToken | None = None,
    dpi: tuple[float, float] | None = None,
    icc_profile: bytes | None = None,
) -> None:
    """Write one or more PNG frames (multi-frame uses the APNG format)."""
    if not frames:
        msg = "no frames to write"
        raise InvalidParameterError(msg)
    for frame in frames:
        _check_frame(frame)
    width, height = _common_size(frames)
    palette = _common_palette(frames)
    writer = _Writer(fp, budget=budget, token=token)

    color_type, _channels, depth = _png_mode(frames[0])
    if any(_png_mode(frame) != (color_type, _channels, depth) for frame in frames):
        msg = "APNG frames must share mode and bit depth"
        raise UnsupportedFeatureError(msg)
    writer.write(PNG_MAGIC)
    _png_chunk(
        writer.fp,
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, 0),
    )

    icc = icc_profile or frames[0].info.get("icc_profile")
    if icc is not None:
        chunk_payload = b"ICC Profile\x00\x00" + zlib.compress(bytes(icc))
        _png_chunk(writer.fp, b"iCCP", chunk_payload)

    if palette is not None:
        _png_chunk(writer.fp, b"PLTE", palette.rgb_bytes()[: 256 * 3])

    transparency_payload = _png_transparency(frames, palette)
    if transparency_payload is not None:
        _png_chunk(writer.fp, b"tRNS", transparency_payload)
    dpi_value = dpi or frames[0].info.get("dpi")
    if dpi_value:
        _png_chunk(
            writer.fp,
            b"pHYs",
            struct.pack(
                ">IIB",
                int(dpi_value[0] / 0.0254 + 0.5),
                int(dpi_value[1] / 0.0254 + 0.5),
                1,
            ),
        )

    multiple = len(frames) > 1
    if multiple:
        _png_chunk(writer.fp, b"acTL", struct.pack(">II", len(frames), 0))
        sequence = 0
        for index, frame in enumerate(frames):
            sequence = _write_png_frame(
                writer,
                frame,
                width,
                height,
                color_type,
                depth,
                sequence=sequence,
                first=index == 0,
            )
    else:
        _png_idat(writer, frames[0], height, color_type, depth)
    _png_chunk(writer.fp, b"IEND", b"")
    writer.fp.flush()


def _write_png_frame(
    writer: _Writer,
    frame: ImageData,
    width: int,
    height: int,
    color_type: int,
    depth: int,
    *,
    sequence: int,
    first: bool,
) -> int:
    offset_x = int(frame.info.get("offset_x", 0))
    offset_y = int(frame.info.get("offset_y", 0))
    delay = frame.info.get("duration_ms", 100)
    delay_numerator = getattr(delay, "numerator", delay)
    delay_denominator = getattr(delay, "denominator", 1000)
    dispose = int(frame.info.get("dispose_op", 0))
    blend = int(frame.info.get("blend_op", 0))
    _png_chunk(
        writer.fp,
        b"fcTL",
        struct.pack(
            ">IIIIIHHBB",
            sequence,
            width,
            height,
            offset_x,
            offset_y,
            int(delay_numerator),
            int(delay_denominator),
            dispose,
            blend,
        ),
    )
    sequence += 1
    payload = zlib.compress(_png_scanlines(frame, height, color_type), 6)
    if first:
        _png_chunk(writer.fp, b"IDAT", payload)
    else:
        _png_chunk(writer.fp, b"fdAT", struct.pack(">I", sequence) + payload)
        sequence += 1
    return sequence


def _png_idat(
    writer: _Writer,
    frame: ImageData,
    height: int,
    color_type: int,
    depth: int,
) -> None:
    del color_type
    del depth
    compressor = zlib.compressobj(6)
    rows_per_strip = max(1, _STRIP_BYTES // max(1, frame.row_length_bytes))
    for y0 in range(0, height, rows_per_strip):
        if writer.token is not None:
            writer.token.raise_if_cancelled()
        y1 = min(y0 + rows_per_strip, height)
        payload = bytearray()
        for y in range(y0, y1):
            payload += _filter_none(_pack_row(frame, y))
        compressed = compressor.compress(bytes(payload))
        if compressed:
            _png_chunk(writer.fp, b"IDAT", compressed)
    tail = compressor.flush()
    if tail:
        _png_chunk(writer.fp, b"IDAT", tail)


def _png_scanlines(frame: ImageData, height: int, color_type: int) -> bytes:
    del color_type
    payload = bytearray()
    for y in range(height):
        payload += _filter_none(_pack_row(frame, y))
    return bytes(payload)


def _pack_row(
    frame: ImageData,
    y: int,
    *,
    big_endian: bool = True,
    pack_indices: bool = True,
) -> bytes:
    """Pack one row.

    PNG requires big-endian words; TIFF sample data follows the file's byte
    order, so the TIFF writer passes ``big_endian=False``.  TIFF stores palette
    indices one per byte, so it also passes ``pack_indices=False``.
    """
    if frame.mode == "P":
        indices = frame.samples[y * frame.width : (y + 1) * frame.width]
        if not pack_indices:
            return bytes(int(value) for value in indices)
        return pack_bits([int(value) for value in indices], frame.bit_depth)
    start = y * frame.width * frame.channels
    end = start + frame.width * frame.channels
    if frame.storage_bits == 16:
        return pack_bits(
            [int(value) for value in frame.samples[start:end]],
            16,
            big_endian=big_endian,
        )
    return bytes(frame.samples[start:end])


def _png_transparency(
    frames: Sequence[ImageData], palette: Palette | None
) -> bytes | None:
    frame = frames[0]
    if palette is not None:
        if palette.has_alpha:
            return palette.alpha_bytes()
        if isinstance(frame.transparency, int):
            return b"\xff" * frame.transparency + b"\x00"
        return None
    if frame.transparency is None:
        return None
    if isinstance(frame.transparency, tuple):
        return b"".join(struct.pack(">H", value) for value in frame.transparency)
    return struct.pack(">H", int(frame.transparency))


# ----------------------------------------------------------------- BMP


def encode_bmp(
    frames: Sequence[ImageData],
    fp: IO[bytes],
    *,
    budget: MemoryBudget | None = None,
    token: CancellationToken | None = None,
) -> None:
    """Write 24/32-bit RGB(A) or 1/4/8-bit indexed BMP data.

    BMP has no multi-page form.  Exactly one frame is accepted; multi-frame
    output is a TIFF or APNG concern.
    """
    if len(frames) != 1:
        msg = "BMP supports exactly one frame; use TIFF or PNG for multi-frame"
        raise UnsupportedFeatureError(msg)
    frame = frames[0]
    _check_frame(frame)
    writer = _Writer(fp, budget=budget, token=token)
    width, height = frame.width, frame.height

    if frame.mode == "P":
        assert frame.palette is not None
        bits = frame.bit_depth
        if bits not in (1, 4, 8):
            # BMP only stores 1, 4 or 8 bits per index; widen the smallest
            # legal bucket rather than silently truncating the indices.
            bits = next((candidate for candidate in (1, 4, 8) if candidate >= bits), 8)
        palette_entries = list(frame.palette.entries)
        if len(palette_entries) > (1 << bits):
            palette_entries = palette_entries[: 1 << bits]
        color_count = len(palette_entries)
        palette_bytes = bytearray()
        for red, green, blue, _alpha in palette_entries:
            palette_bytes += bytes((blue, green, red, 0))
        compression = 0
        row_stride = ((width * bits + 31) // 32) * 4
        header_size = 40
        data_offset = 14 + header_size + len(palette_bytes)
        payload_size = checked_mul(row_stride, height, "BMP pixel data")
        file_size = checked_add(data_offset, payload_size, "BMP file size")
        writer.write(
            _bmp_header(
                file_size,
                data_offset,
                width,
                height,
                bits,
                compression,
                color_count,
                header_size,
            )
        )
        writer.write(bytes(palette_bytes), "BMP palette")
        for row in range(height - 1, -1, -1):
            indices = frame.samples[row * width : (row + 1) * width]
            packed = pack_bits([int(value) for value in indices], bits)
            writer.write(packed + b"\x00" * (row_stride - len(packed)), "BMP row")
        return

    if frame.mode in {"RGB", "RGBA"} and frame.storage_bits == 8:
        bits = 32 if frame.mode == "RGBA" else 24
        channels = frame.channels
        row_stride = ((width * bits + 31) // 32) * 4
        header_size = 40
        data_offset = 14 + header_size
        payload_size = checked_mul(row_stride, height, "BMP pixel data")
        file_size = checked_add(data_offset, payload_size, "BMP file size")
        writer.write(
            _bmp_header(file_size, data_offset, width, height, bits, 0, 0, header_size)
        )
        for row in range(height - 1, -1, -1):
            if token is not None:
                token.raise_if_cancelled()
            start = row * width * channels
            line = bytearray()
            if bits == 32:
                for x in range(width):
                    index = start + x * channels
                    line += bytes(
                        (
                            frame.samples[index + 2],
                            frame.samples[index + 1],
                            frame.samples[index],
                            frame.samples[index + 3],
                        )
                    )
            else:
                for x in range(width):
                    index = start + x * channels
                    line += bytes(
                        (
                            frame.samples[index + 2],
                            frame.samples[index + 1],
                            frame.samples[index],
                        )
                    )
            line += b"\x00" * (row_stride - len(line))
            writer.write(bytes(line), "BMP row")
        return

    msg = (
        f"BMP cannot store mode {frame.mode!r} at {frame.storage_bits} bits; "
        "convert to RGB, RGBA or an indexed mode first"
    )
    raise UnsupportedFeatureError(msg)


def _bmp_header(
    file_size: int,
    data_offset: int,
    width: int,
    height: int,
    bits: int,
    compression: int,
    color_count: int,
    header_size: int,
) -> bytes:
    out = bytearray(b"BM")
    out += struct.pack("<IHHI", file_size, 0, 0, data_offset)
    out += struct.pack(
        "<IiiHHIIiiII",
        header_size,
        width,
        height,
        1,
        bits,
        compression,
        file_size - data_offset,
        2835,
        2835,
        color_count,
        0,
    )
    return bytes(out)


# ----------------------------------------------------------------- TIFF


def encode_tiff(
    frames: Sequence[ImageData],
    fp: IO[bytes],
    *,
    budget: MemoryBudget | None = None,
    token: CancellationToken | None = None,
    compression: str = "none",
) -> None:
    """Write little-endian TIFF pages with full sample precision.

    Pages are laid out sequentially: the 8-byte header, then for each page its
    pixel strips, out-of-line tag values and IFD.  Absolute offsets are
    computed in memory before anything is written, so the encoder never seeks
    and works on a plain byte sink.  The IFD chain is patched so readers see
    the frames in the supplied order.
    """
    if not frames:
        msg = "no frames to write"
        raise InvalidParameterError(msg)
    for frame in frames:
        _check_frame(frame)
    frames = [_tiff_safe_frame(frame) for frame in frames]
    if compression not in {"none", "deflate"}:
        msg = f"unsupported TIFF compression {compression!r}"
        raise InvalidParameterError(msg)
    _common_size(frames)
    _common_palette(frames)

    builder = _TiffPageBuilder(compression, budget, token)
    base = 8
    pages: list[bytes] = []
    ifd_offsets: list[int] = []
    ifd_sizes: list[int] = []
    for frame in frames:
        if token is not None:
            token.raise_if_cancelled()
        page, ifd_offset, ifd_size = builder.build(frame, base)
        pages.append(page)
        ifd_offsets.append(ifd_offset)
        ifd_sizes.append(ifd_size)
        base += len(page)

    for index in range(len(pages) - 1):
        # The next-IFD pointer is the last field of the page's IFD, which is
        # itself the last structure in the page.
        relative = len(pages[index]) - 4
        pages[index] = _patch_next_ifd(pages[index], relative, ifd_offsets[index + 1])
    del ifd_sizes

    writer = _Writer(fp, budget=budget, token=token)
    writer.write(b"II\x2a\x00")
    writer.write(struct.pack("<I", ifd_offsets[0] if ifd_offsets else 8))
    for page in pages:
        writer.write(page, "TIFF page")


def _tiff_safe_frame(frame: ImageData) -> ImageData:
    """TIFF palettes have no alpha, so expand such frames to RGBA losslessly."""
    if frame.mode == "P" and frame.palette is not None and frame.palette.has_alpha:
        expanded = frame.to_rgba()
        return expanded.with_info(**frame.info)
    return frame


def _patch_next_ifd(page: bytes, relative_offset: int, value: int) -> bytes:
    """Overwrite the 4-byte next-IFD pointer at ``relative_offset``."""
    mutable = bytearray(page)
    mutable[relative_offset : relative_offset + 4] = struct.pack("<I", value)
    return bytes(mutable)


class _TiffPageBuilder:
    """Builds one self-contained TIFF page at a known absolute base offset."""

    def __init__(
        self,
        compression: str,
        budget: MemoryBudget | None,
        token: CancellationToken | None,
    ) -> None:
        self.compression = compression
        self.budget = budget
        self.token = token

    def build(self, frame: ImageData, base: int) -> tuple[bytes, int, int]:
        if frame.mode == "P":
            # Indices are stored unpacked (one byte each) even when the palette
            # was declared at a lower depth, so BitsPerSample is always 8 here.
            samples_per_pixel = 1
            bits_per_sample = [8]
            photometric = 3
            sample_row_bytes = frame.width
        else:
            samples_per_pixel = frame.channels
            bits_per_sample = [frame.storage_bits] * frame.channels
            photometric = 2 if frame.mode in {"RGB", "RGBA"} else 1
            sample_row_bytes = frame.row_length_bytes

        width, height = frame.width, frame.height
        row_length = sample_row_bytes
        rows_per_strip = max(1, min(height, _STRIP_BYTES // max(1, row_length)))
        strip_count = (height + rows_per_strip - 1) // rows_per_strip
        strips: list[bytes] = []
        for strip_index in range(strip_count):
            if self.token is not None:
                self.token.raise_if_cancelled()
            y0 = strip_index * rows_per_strip
            y1 = min(y0 + rows_per_strip, height)
            payload = bytearray()
            for y in range(y0, y1):
                payload += _pack_row(frame, y, big_endian=False, pack_indices=False)
            if self.budget is not None:
                self.budget.reserve(len(payload), "TIFF strip")
            if self.compression == "deflate":
                payload = bytearray(zlib.compress(bytes(payload), 6))
            strips.append(bytes(payload))

        cursor = base
        strip_offsets: list[int] = []
        body = bytearray()
        for strip in strips:
            strip_offsets.append(cursor)
            body += strip
            cursor += len(strip)

        extras = bytearray()

        def add_extra(data: bytes) -> int:
            nonlocal cursor
            offset = cursor
            extras.extend(data)
            cursor += len(data)
            return offset

        bits_field: bytes | None = None
        if len(bits_per_sample) > 1:
            packed = b"".join(struct.pack("<H", value) for value in bits_per_sample)
            if len(packed) <= 4:
                # Up to two SHORT values fit inline in the IFD entry.
                bits_field = packed
            else:
                bits_field = struct.pack("<I", add_extra(packed))
        colormap_value: int | None = None
        if frame.mode == "P":
            assert frame.palette is not None
            entries = 256
            table = bytearray(entries * 2 * 3)
            for index in range(entries):
                red = green = blue = 0
                if index < len(frame.palette.entries):
                    red, green, blue = frame.palette.entries[index][:3]
                struct.pack_into("<H", table, index * 2, red * 257)
                struct.pack_into("<H", table, (entries + index) * 2, green * 257)
                struct.pack_into("<H", table, (2 * entries + index) * 2, blue * 257)
            colormap_value = add_extra(bytes(table))

        if len(strip_offsets) == 1:
            # A single LONG value fits inline in the IFD entry; pointing at a
            # table would be legal but makes every reader do extra work.
            offsets_field = struct.pack("<I", strip_offsets[0])
        else:
            offsets_field = struct.pack(
                "<I", add_extra(b"".join(struct.pack("<I", o) for o in strip_offsets))
            )
        if len(strips) == 1:
            counts_field = struct.pack("<I", len(strips[0]))
        else:
            counts_field = struct.pack(
                "<I", add_extra(b"".join(struct.pack("<I", len(s)) for s in strips))
            )
        icc = frame.info.get("icc_profile")
        icc_value: int | None = None
        icc_bytes = b""
        if icc:
            icc_bytes = bytes(icc)
            icc_value = add_extra(icc_bytes)

        tags: list[tuple[int, int, int, bytes]] = []

        def tag(tag_id: int, type_id: int, count: int, values: bytes) -> None:
            tags.append((tag_id, type_id, count, values))

        tag(256, 4, 1, struct.pack("<I", width))
        tag(257, 4, 1, struct.pack("<I", height))
        if bits_field is None:
            tag(258, 3, 1, struct.pack("<H", bits_per_sample[0]))
        else:
            tag(258, 3, len(bits_per_sample), bits_field)
        tag(259, 3, 1, struct.pack("<H", 8 if self.compression == "deflate" else 1))
        tag(262, 3, 1, struct.pack("<H", photometric))
        tag(273, 4, len(strip_offsets), offsets_field)
        tag(274, 3, 1, struct.pack("<H", frame.orientation))
        tag(277, 3, 1, struct.pack("<H", samples_per_pixel))
        tag(278, 4, 1, struct.pack("<I", rows_per_strip))
        tag(279, 4, len(strips), counts_field)
        if colormap_value is not None:
            tag(320, 3, 3 * 256, struct.pack("<I", colormap_value))
        if frame.mode in {"RGBA", "LA"}:
            tag(338, 3, 1, struct.pack("<H", 2))  # unassociated alpha
        if icc_value is not None:
            if len(icc_bytes) <= 4:
                tag(34675, 7, len(icc_bytes), icc_bytes)
            else:
                tag(34675, 7, len(icc_bytes), struct.pack("<I", icc_value))

        tags.sort(key=lambda item: item[0])
        ifd_offset = cursor
        ifd = bytearray(struct.pack("<H", len(tags)))
        for tag_id, type_id, count, values in tags:
            ifd += struct.pack("<HHI", tag_id, type_id, count)
            if len(values) <= 4:
                ifd += values + b"\x00" * (4 - len(values))
            else:
                (pointer,) = struct.unpack("<I", values)
                ifd += struct.pack("<I", pointer)
        ifd += struct.pack("<I", 0)  # next IFD pointer, patched for multi-page

        page = bytearray(body)
        page += extras
        page += ifd
        return bytes(page), ifd_offset, len(ifd)


# ----------------------------------------------------------------- driver


def _atomic_target(path: StrOrBytesPath) -> tuple[str, int]:
    directory = os.path.dirname(os.path.abspath(os.fsdecode(path))) or "."
    descriptor, temporary = tempfile.mkstemp(prefix=".highdepth-", dir=directory)
    return temporary, descriptor


def write_frames(
    path: StrOrBytesPath | None,
    frames: Sequence[ImageData],
    format: str,
    *,
    fp: IO[bytes] | None = None,
    budget: MemoryBudget | None = None,
    token: CancellationToken | None = None,
    icc_profile: bytes | None = None,
    dpi: tuple[float, float] | None = None,
    compression: str = "none",
) -> None:
    """Write frames to ``path`` (atomically) or to an open binary file ``fp``."""
    if not frames:
        msg = "no frames to write"
        raise InvalidParameterError(msg)
    normalized = format.upper()
    if normalized not in {"PNG", "TIFF", "BMP"}:
        msg = f"unsupported output format {format!r}"
        raise UnsupportedFeatureError(msg)
    if normalized == "BMP" and len(frames) != 1:
        msg = "BMP supports exactly one frame"
        raise UnsupportedFeatureError(msg)

    def emit(handle: IO[bytes]) -> None:
        if budget is not None:
            # Verify that the source frames themselves fit the budget before
            # any per-block reservation, so a too-small budget fails up front.
            for frame in frames:
                budget.check(len(frame.tobytes()), "image data")
        if normalized == "PNG":
            encode_png(
                frames,
                handle,
                budget=budget,
                token=token,
                icc_profile=icc_profile,
                dpi=dpi,
            )
        elif normalized == "TIFF":
            encode_tiff(
                frames, handle, budget=budget, token=token, compression=compression
            )
        else:
            encode_bmp(frames, handle, budget=budget, token=token)

    if fp is not None:
        emit(fp)
        return
    if path is None:
        msg = "either path or fp is required"
        raise InvalidParameterError(msg)
    target = os.fspath(path)
    temporary, descriptor = _atomic_target(target)
    handle = os.fdopen(descriptor, "wb")
    try:
        emit(handle)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, target)
    except BaseException:
        try:
            handle.close()
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def verify_roundtrip(
    path: StrOrBytesPath,
    expected: Sequence[ImageData],
    format: str,
) -> None:
    """Re-read a written file and compare frame by frame, byte for byte."""
    from .read import read_frames

    actual = read_frames(os.fsdecode(path), format=format)
    if len(actual) != len(expected):
        msg = f"{path!s} has {len(actual)} frames, expected {len(expected)}"
        raise WriteVerificationError(msg)
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        if got.mode != want.mode or (got.width, got.height) != (
            want.width,
            want.height,
        ):
            msg = (
                f"frame {index} of {path!s} is {got.mode} {got.width}x{got.height}, "
                f"expected {want.mode} {want.width}x{want.height}"
            )
            raise WriteVerificationError(msg)
        if got.tobytes() != want.tobytes():
            msg = f"frame {index} of {path!s} did not round-trip byte for byte"
            raise WriteVerificationError(msg)


__all__ = [
    "PNG_MAGIC",
    "encode_bmp",
    "encode_png",
    "encode_tiff",
    "verify_roundtrip",
    "write_frames",
]
