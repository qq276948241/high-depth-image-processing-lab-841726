#
# The Python Imaging Library.
#
# Normalized, lossless image container used by the high depth pipeline.
#
# See the README file for information on usage and redistribution.
#
"""Normalized image representation.

The model keeps samples at their stored precision.  Nothing is rescaled as a
side effect of reading: an 8-bit PNG stays 8-bit, a 16-bit TIFF stays 16-bit
and a palette image stays indexed.  Conversions that do change precision are
explicit methods with documented rounding and are allowed to fail.

``mode`` is derived from the channel count (``L``, ``LA``, ``RGB``, ``RGBA``
or ``P``) and ``storage_bits`` records the precision (8 or 16).  Unlike a
:class:`PIL.Image.Image`, a 16-bit RGB image is therefore representable.
"""

from __future__ import annotations

import sys
from array import array
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from .errors import (
    BitDepthError,
    DimensionOverflowError,
    EmptyImageError,
    IllegalPaletteError,
    InvalidParameterError,
    NonFiniteSampleError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

MAX_DIMENSION = 2**31 - 1
"""Largest accepted width or height, matching the PNG specification."""

MAX_PIXELS = 2**31 - 1
"""Largest accepted pixel count before an explicit budget is consulted."""

_MODE_FOR_CHANNELS = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}
_CHANNELS_FOR_MODE = {value: key for key, value in _MODE_FOR_CHANNELS.items()}


def checked_mul(a: int, b: int, what: str = "size") -> int:
    """Return ``a * b`` or raise :class:`DimensionOverflowError`."""
    if a < 0 or b < 0:
        msg = f"{what} is negative ({a} * {b})"
        raise DimensionOverflowError(msg)
    if a and b > sys.maxsize // a:
        msg = f"{what} overflows the addressable range ({a} * {b})"
        raise DimensionOverflowError(msg)
    return a * b


def checked_add(a: int, b: int, what: str = "offset") -> int:
    """Return ``a + b`` or raise :class:`DimensionOverflowError`."""
    if a < 0 or b < 0:
        msg = f"{what} is negative ({a} + {b})"
        raise DimensionOverflowError(msg)
    if b > sys.maxsize - a:
        msg = f"{what} overflows the addressable range ({a} + {b})"
        raise DimensionOverflowError(msg)
    return a + b


def row_bytes(width: int, channels: int, bit_depth: int) -> int:
    """Number of bytes in one packed row, rounded up to a byte boundary."""
    if width <= 0:
        msg = f"width must be positive, got {width}"
        raise InvalidParameterError(msg)
    if channels not in (1, 2, 3, 4):
        msg = f"unsupported channel count {channels}"
        raise InvalidParameterError(msg)
    if not 1 <= bit_depth <= 16:
        msg = f"unsupported bit depth {bit_depth}"
        raise InvalidParameterError(msg)
    bits = checked_mul(checked_mul(width, channels, "row bits"), bit_depth, "row bits")
    return (bits + 7) // 8


def storage_bytes(width: int, height: int, channels: int, storage_bits: int) -> int:
    """Bytes needed for an unpacked, byte aligned buffer."""
    if storage_bits not in (8, 16):
        msg = f"unsupported storage width {storage_bits}"
        raise InvalidParameterError(msg)
    return checked_mul(
        checked_mul(height, width, "pixel count"),
        channels * (storage_bits // 8),
        "buffer size",
    )


def _validate_size(width: int, height: int) -> None:
    if width < 0 or height < 0:
        msg = f"negative image size {(width, height)}"
        raise InvalidParameterError(msg)
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        msg = f"image size {(width, height)} exceeds the supported maximum"
        raise DimensionOverflowError(msg)
    pixels = checked_mul(width, height, "pixel count")
    if pixels > MAX_PIXELS:
        msg = f"image size {(width, height)} exceeds the pixel limit"
        raise DimensionOverflowError(msg)
    # Honour Pillow's decompression-bomb limits so a hostile header cannot
    # make this path allocate far more than the rest of the library allows.
    from PIL import Image

    limit = getattr(Image, "MAX_IMAGE_PIXELS", None)
    if limit is not None and pixels > 2 * limit:
        msg = (
            f"Image size ({pixels} pixels) exceeds limit of {2 * limit} pixels, "
            "could be decompression bomb DOS attack."
        )
        raise DimensionOverflowError(msg)


def check_image_size(width: int, height: int) -> None:
    """Validate declared dimensions before any pixel buffer is allocated.

    Enforces the format limit, the addressable-range limit and Pillow's
    decompression-bomb limit, so a hostile header fails immediately.
    """
    _validate_size(width, height)


def scale_sample(value: int, from_bits: int, to_bits: int) -> int:
    """Scale an integer sample with round-half-up, without floating point."""
    if from_bits == to_bits:
        return value
    source_max = (1 << from_bits) - 1
    target_max = (1 << to_bits) - 1
    if to_bits > from_bits:
        return (value * target_max + source_max // 2) // source_max
    return (value * target_max + source_max // 2) // source_max


@dataclass(frozen=True)
class Palette:
    """An immutable RGBA colour table.

    ``entries`` always has 1..256 items.  Alpha is stored per entry so that a
    PNG ``tRNS`` table and a quantizer that needs a transparent entry share the
    same representation.
    """

    entries: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.entries) <= 256:
            msg = f"palette must have 1..256 entries, got {len(self.entries)}"
            raise IllegalPaletteError(msg)
        for entry in self.entries:
            if len(entry) != 4 or any(not 0 <= v <= 255 for v in entry):
                msg = f"invalid palette entry {entry!r}"
                raise IllegalPaletteError(msg)

    def __len__(self) -> int:
        return len(self.entries)

    def rgb_bytes(self) -> bytes:
        out = bytearray()
        for red, green, blue, _alpha in self.entries:
            out.extend((red, green, blue))
        return bytes(out)

    def rgba_bytes(self) -> bytes:
        out = bytearray()
        for entry in self.entries:
            out.extend(entry)
        return bytes(out)

    def alpha_bytes(self) -> bytes:
        return bytes(entry[3] for entry in self.entries)

    @property
    def has_alpha(self) -> bool:
        return any(entry[3] != 255 for entry in self.entries)

    @property
    def opaque(self) -> bool:
        return all(entry[3] == 255 for entry in self.entries)

    def transparent_indices(self) -> tuple[int, ...]:
        return tuple(i for i, entry in enumerate(self.entries) if entry[3] == 0)

    def with_entry(self, index: int, entry: tuple[int, int, int, int]) -> Palette:
        if not 0 <= index < len(self.entries):
            msg = f"palette index {index} outside table"
            raise IllegalPaletteError(msg)
        entries = list(self.entries)
        entries[index] = entry
        return Palette(tuple(entries))

    @classmethod
    def from_rgb(cls, data: bytes | bytearray | memoryview) -> Palette:
        data = bytes(data)
        if not data or len(data) % 3:
            msg = f"RGB palette data has invalid length {len(data)}"
            raise IllegalPaletteError(msg)
        count = len(data) // 3
        if count > 256:
            msg = f"RGB palette has {count} entries, maximum is 256"
            raise IllegalPaletteError(msg)
        entries = tuple(
            (data[i], data[i + 1], data[i + 2], 255) for i in range(0, len(data), 3)
        )
        return cls(entries)

    @classmethod
    def from_rgba(cls, data: bytes | bytearray | memoryview) -> Palette:
        data = bytes(data)
        if not data or len(data) % 4:
            msg = f"RGBA palette data has invalid length {len(data)}"
            raise IllegalPaletteError(msg)
        count = len(data) // 4
        if count > 256:
            msg = f"RGBA palette has {count} entries, maximum is 256"
            raise IllegalPaletteError(msg)
        entries = tuple(
            (data[i], data[i + 1], data[i + 2], data[i + 3])
            for i in range(0, len(data), 4)
        )
        return cls(entries)


@dataclass
class ImageData:
    """A decoded image with its samples at their stored precision.

    ``samples`` is either ``bytes`` (8-bit storage) or an
    :class:`array.array` of ``"H"`` (16-bit storage).  Values are interleaved
    in canonical band order (R, G, B, A for direct colour, a single index band
    for palette images), row major, top-left origin regardless of the
    container's native orientation.
    """

    format: str
    mode: str
    width: int
    height: int
    bit_depth: int
    storage_bits: int
    channels: int
    samples: bytes | array[int]
    palette: Palette | None = None
    transparency: int | tuple[int, ...] | None = None
    row_stride: int = 0
    orientation: int = 1
    interlaced: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_size(self.width, self.height)
        if self.row_stride == 0:
            # Default to the canonical unpadded row length so hand-built
            # images do not need to pass it.  A zero-sized image has no rows.
            if self.width > 0:
                object.__setattr__(
                    self,
                    "row_stride",
                    row_bytes(self.width, self.channels, self.storage_bits),
                )
        if self.mode == "P":
            if self.palette is None:
                msg = "palette image requires a palette"
                raise IllegalPaletteError(msg)
            if self.channels != 1:
                msg = "palette image must have one index band"
                raise IllegalPaletteError(msg)
            if not 1 <= self.bit_depth <= 16:
                msg = f"invalid palette index depth {self.bit_depth}"
                raise IllegalPaletteError(msg)
            if self.storage_bits not in (8, 16):
                msg = f"invalid index storage width {self.storage_bits}"
                raise IllegalPaletteError(msg)
            max_index = (1 << self.bit_depth) - 1
            if any(int(index) > max_index for index in self.samples):
                msg = "palette index outside the declared bit depth"
                raise IllegalPaletteError(msg)
            if isinstance(self.transparency, int) and not (
                0 <= self.transparency < len(self.palette)
            ):
                msg = "transparent palette index is outside the palette"
                raise IllegalPaletteError(msg)
        else:
            if self.channels not in _MODE_FOR_CHANNELS:
                msg = f"unsupported channel count {self.channels}"
                raise InvalidParameterError(msg)
            expected_mode = _MODE_FOR_CHANNELS[self.channels]
            if self.mode != expected_mode:
                msg = (
                    f"mode {self.mode!r} does not match {self.channels} channels "
                    f"(expected {expected_mode!r})"
                )
                raise InvalidParameterError(msg)
            if self.palette is not None:
                msg = "direct colour image must not carry a palette"
                raise IllegalPaletteError(msg)
            if not 1 <= self.bit_depth <= self.storage_bits:
                msg = f"invalid bit depth {self.bit_depth}"
                raise InvalidParameterError(msg)
            transparency: object = self.transparency
            if transparency is None:
                pass
            elif isinstance(transparency, int):
                if not 0 <= transparency <= (1 << self.bit_depth) - 1:
                    msg = "transparency value outside the sample range"
                    raise InvalidParameterError(msg)
            elif isinstance(transparency, tuple):
                if len(transparency) != self.channels:
                    msg = "transparency tuple length must match the channels"
                    raise InvalidParameterError(msg)
                limit = (1 << self.bit_depth) - 1
                if any(not 0 <= value <= limit for value in transparency):
                    msg = "transparency value outside the sample range"
                    raise InvalidParameterError(msg)
            else:
                # Defensive: untyped callers may pass anything else.
                msg = "transparency must be an integer or a tuple"
                raise InvalidParameterError(msg)
        expected = storage_bytes(
            self.width, self.height, self.channels, self.storage_bits
        )
        # ``bytes`` holds one byte per entry while ``array('H')`` holds one
        # 16-bit sample per entry, so compare serialized byte lengths.
        if isinstance(self.samples, bytes):
            actual_bytes = len(self.samples)
        else:
            actual_bytes = len(self.samples) * (self.storage_bits // 8)
        if actual_bytes != expected:
            msg = f"sample buffer has {actual_bytes} bytes, expected {expected}"
            raise InvalidParameterError(msg)
        if self.row_stride < 0:
            msg = "row stride must not be negative"
            raise InvalidParameterError(msg)
        if self.orientation not in range(1, 9):
            msg = f"invalid orientation {self.orientation}"
            raise InvalidParameterError(msg)

    # -- basic properties ------------------------------------------------

    @property
    def pixel_count(self) -> int:
        return checked_mul(self.width, self.height, "pixel count")

    @property
    def row_length_bytes(self) -> int:
        """Canonical (unpadded) length of one packed row in bytes."""
        return row_bytes(self.width, self.channels, self.storage_bits)

    @property
    def sample_max(self) -> int:
        return (1 << self.bit_depth) - 1

    @property
    def bands(self) -> tuple[str, ...]:
        if self.mode == "P":
            return ("P",)
        if self.channels == 1:
            return ("L",)
        return tuple(self.mode)

    @property
    def pil_mode(self) -> str:
        """The Pillow mode that can carry these samples, if any."""
        if self.mode == "P":
            return "P" if self.bit_depth <= 8 else "P;16"
        if self.storage_bits == 16:
            if self.channels == 1:
                return "I;16"
            return f"{self.mode};16"
        return self.mode

    def validate(self) -> None:
        """Run the structural checks again; useful after hand construction."""
        self.__post_init__()

    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0

    # -- sample access ---------------------------------------------------

    def _row_slice(self, y: int) -> slice:
        if not 0 <= y < self.height:
            msg = f"row {y} outside image"
            raise InvalidParameterError(msg)
        step = self.width * self.channels
        start = checked_mul(y, step, "row offset")
        return slice(start, start + step)

    def row_samples(self, y: int) -> tuple[int, ...]:
        """Return one row of interleaved samples."""
        return tuple(int(v) for v in self.samples[self._row_slice(y)])

    def sample_at(self, x: int, y: int) -> tuple[int, ...]:
        if not (0 <= x < self.width and 0 <= y < self.height):
            msg = f"pixel {(x, y)} outside image"
            raise InvalidParameterError(msg)
        start = (y * self.width + x) * self.channels
        return tuple(int(v) for v in self.samples[start : start + self.channels])

    def iter_rows(self) -> Iterable[tuple[int, ...]]:
        """Yield one row of samples at a time without copying the image."""
        step = self.width * self.channels
        for y in range(self.height):
            start = checked_mul(y, step, "row offset")
            yield tuple(int(v) for v in self.samples[start : start + step])

    def with_samples(self, samples: bytes | array[int]) -> ImageData:
        return replace(self, samples=samples)

    def with_info(self, **info: Any) -> ImageData:
        merged = dict(self.info)
        merged.update(info)
        return replace(self, info=merged)

    # -- serialization ---------------------------------------------------

    def tobytes(self) -> bytes:
        """Canonical byte serialization of the sample buffer.

        8-bit storage is emitted as-is.  16-bit storage is emitted big-endian,
        two bytes per sample, matching PNG.  ``storage_bits == 8`` samples are
        assumed to fit in a byte (the model enforces that at construction).
        """
        if isinstance(self.samples, bytes):
            return self.samples
        if self.storage_bits == 8:
            return bytes(bytearray(self.samples))
        out = bytearray()
        for value in self.samples:
            out.append((value >> 8) & 0xFF)
            out.append(value & 0xFF)
        return bytes(out)

    def sha256(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        digest.update(self.tobytes())
        digest.update(self.width.to_bytes(8, "big"))
        digest.update(self.height.to_bytes(8, "big"))
        digest.update(self.mode.encode())
        digest.update(bytes((self.bit_depth, self.storage_bits, self.channels)))
        if self.palette is not None:
            digest.update(self.palette.rgba_bytes())
        if self.transparency is not None:
            digest.update(repr(self.transparency).encode())
        return digest.hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "mode": self.mode,
            "size": (self.width, self.height),
            "bit_depth": self.bit_depth,
            "storage_bits": self.storage_bits,
            "channels": self.channels,
            "palette_entries": len(self.palette) if self.palette else 0,
            "palette_alpha": bool(self.palette and self.palette.has_alpha),
            "transparency": self.transparency,
            "row_stride": self.row_stride,
            "orientation": self.orientation,
            "interlaced": self.interlaced,
        }

    # -- Pillow bridge ---------------------------------------------------

    def to_pil(self, *, allow_reduce: bool = False) -> Any:
        """Return an :class:`PIL.Image.Image` view of this data.

        Pillow has no 16-bit multi-band mode.  Converting such an image
        therefore raises :class:`BitDepthError` unless ``allow_reduce`` is set,
        in which case samples are rounded to 8 bits with documented half-up
        rounding.
        """
        from PIL import Image

        if self.is_empty():
            msg = "cannot export an empty image"
            raise EmptyImageError(msg)

        if self.mode == "P":
            if self.bit_depth > 8:
                msg = "Pillow cannot represent palette indices deeper than 8 bits"
                if not allow_reduce:
                    raise BitDepthError(msg)
                indices = bytes(
                    scale_sample(int(v), self.bit_depth, 8) for v in self.samples
                )
            else:
                indices = bytes(int(v) for v in self.samples)
            im = Image.frombytes("P", (self.width, self.height), indices)
            assert self.palette is not None
            if self.palette.has_alpha:
                im.putpalette(self.palette.rgba_bytes(), "RGBA")
            else:
                im.putpalette(self.palette.rgb_bytes())
            if isinstance(self.transparency, int):
                im.info["transparency"] = self.transparency
            elif self.palette.has_alpha:
                im.info["transparency"] = self.palette.alpha_bytes()
            return im

        if self.storage_bits == 16 and self.channels > 1 and not allow_reduce:
            msg = (
                "Pillow cannot represent 16-bit multi-band images; "
                "pass allow_reduce=True for explicit 8-bit rounding"
            )
            raise BitDepthError(msg)

        if self.storage_bits == 16:
            if self.channels == 1:
                # tobytes() is big-endian; decode through the explicit rawmode.
                im = Image.frombytes(
                    "I;16", (self.width, self.height), self.tobytes(), "raw", "I;16B"
                )
            else:
                out = bytearray()
                for value in self.samples:
                    out.append(scale_sample(int(value), self.bit_depth, 8))
                im = Image.frombytes(
                    _MODE_FOR_CHANNELS[self.channels],
                    (self.width, self.height),
                    bytes(out),
                )
        else:
            im = Image.frombytes(self.mode, (self.width, self.height), self.tobytes())
        if isinstance(self.transparency, int):
            im.info["transparency"] = self.transparency
        elif self.transparency is not None:
            im.info["transparency"] = self.transparency
        if self.info.get("icc_profile"):
            im.info["icc_profile"] = self.info["icc_profile"]
        return im

    @classmethod
    def from_pil(
        cls, im: Any, *, format: str | None = None, bit_depth: int | None = None
    ) -> ImageData:
        """Normalize a loaded Pillow image, preserving precision where it exists.

        ``bit_depth`` overrides the depth inferred for palette images; Pillow
        itself does not remember whether a palette was stored with 4 or 8 bits
        per index.
        """
        im.load()
        width, height = im.size
        mode = im.mode
        info = dict(im.info)
        stride = row_bytes(width, im.im.bands, 8)
        source_format = format or im.format or ""

        if mode == "P":
            return cls._palette_from_pil(
                im, source_format, info, stride, bit_depth=bit_depth
            )

        if mode in {"I;16", "I;16L", "I;16B", "I;16N", "I;16S", "I;16LS", "I;16BS"}:
            values = _load_16bit(im, mode)
            return cls(
                format=source_format,
                mode="L",
                width=width,
                height=height,
                bit_depth=16,
                storage_bits=16,
                channels=1,
                samples=values,
                row_stride=stride,
                info=info,
            )

        if mode == "F":
            values = array("f", im.tobytes())
            if any(
                value != value or value in (float("inf"), float("-inf"))
                for value in values
            ):
                msg = "floating point image contains NaN or Infinity"
                raise NonFiniteSampleError(msg)
            return cls(
                format=source_format,
                mode="L",
                width=width,
                height=height,
                bit_depth=8,
                storage_bits=8,
                channels=1,
                samples=bytes(max(0, min(255, int(value + 0.5))) for value in values),
                row_stride=stride,
                info={**info, "float_samples": list(values)},
            )

        if mode == "I":
            values = array("i", im.tobytes())
            if any(value < 0 or value > 65535 for value in values):
                msg = "32-bit integer image cannot be narrowed to 16 bits"
                raise BitDepthError(msg)
            return cls(
                format=source_format,
                mode="L",
                width=width,
                height=height,
                bit_depth=16,
                storage_bits=16,
                channels=1,
                samples=array("H", values),
                row_stride=stride,
                info=info,
            )

        if mode == "1":
            data = im.convert("L").tobytes()
            return cls._from_direct(
                source_format, width, height, 1, data, info, stride, bit_depth=1
            )

        if mode == "RGBX":
            packed = im.tobytes()
            data = bytes(
                component for index, component in enumerate(packed) if index % 4 != 3
            )
            return cls._from_direct(source_format, width, height, 3, data, info, stride)

        if mode in {"RGBa", "La"}:
            base = "RGBA" if mode == "RGBa" else "LA"
            data = im.convert(base).tobytes()
            return cls._from_direct(
                source_format,
                width,
                height,
                4 if base == "RGBA" else 2,
                data,
                {**info, "premultiplied": True},
                stride,
            )

        if mode in {"L", "LA", "RGB", "RGBA"}:
            channels = _CHANNELS_FOR_MODE[mode]
            return cls._from_direct(
                source_format, width, height, channels, im.tobytes(), info, stride
            )

        msg = f"cannot normalize mode {mode!r}"
        raise BitDepthError(msg)

    @classmethod
    def _palette_from_pil(
        cls,
        im: Any,
        source_format: str,
        info: dict[str, Any],
        stride: int,
        *,
        bit_depth: int | None = None,
    ) -> ImageData:
        palette_mode = im.palette.mode if im.palette is not None else "RGB"
        palette_data = im.getpalette(palette_mode) or im.getpalette("RGB")
        if not palette_data:
            msg = "palette image without palette data"
            raise IllegalPaletteError(msg)
        if palette_mode == "RGBA" and len(palette_data) % 4 == 0:
            palette = Palette.from_rgba(palette_data)
        elif len(palette_data) % 3 == 0:
            palette = Palette.from_rgb(palette_data)
        else:
            msg = (
                f"palette data length {len(palette_data)} is not a whole number "
                "of RGB or RGBA entries"
            )
            raise IllegalPaletteError(msg)
        indices = im.tobytes()
        if bit_depth is None:
            max_index = max(indices) if indices else 0
            bit_depth = max(1, max_index.bit_length())
        transparency = info.get("transparency")
        transparent_index: int | None = None
        if isinstance(transparency, bytes):
            alpha = bytearray(palette.alpha_bytes())
            for index, value in enumerate(transparency[: len(palette)]):
                alpha[index] = value
            palette = Palette(
                tuple((*entry[:3], alpha[i]) for i, entry in enumerate(palette.entries))
            )
        elif isinstance(transparency, int):
            if not 0 <= transparency < len(palette):
                msg = "transparent palette index is outside the palette"
                raise IllegalPaletteError(msg)
            palette = palette.with_entry(
                transparency, (*palette.entries[transparency][:3], 0)
            )
            transparent_index = transparency
        return cls(
            format=source_format,
            mode="P",
            width=im.width,
            height=im.height,
            bit_depth=bit_depth,
            storage_bits=8,
            channels=1,
            samples=indices,
            palette=palette,
            transparency=transparent_index,
            row_stride=stride,
            info=info,
        )

    @classmethod
    def _from_direct(
        cls,
        format: str,
        width: int,
        height: int,
        channels: int,
        data: bytes,
        info: dict[str, Any],
        stride: int,
        *,
        bit_depth: int = 8,
    ) -> ImageData:
        transparency = info.get("transparency")
        if not isinstance(transparency, (int, tuple)):
            transparency = None
        return cls(
            format=format,
            mode=_MODE_FOR_CHANNELS[channels],
            width=width,
            height=height,
            bit_depth=bit_depth,
            storage_bits=8,
            channels=channels,
            samples=data,
            transparency=transparency,
            row_stride=stride,
            info=info,
        )

    # -- palette helpers -------------------------------------------------

    def to_rgba(self) -> ImageData:
        """Expand a palette image to RGBA using its colour table.

        Entries whose alpha is zero keep their RGB values; no pixel is turned
        black.
        """
        if self.mode != "P":
            return self
        assert self.palette is not None
        out = bytearray()
        for index in self.samples:
            out.extend(self.palette.entries[int(index)])
        return ImageData(
            format=self.format,
            mode="RGBA",
            width=self.width,
            height=self.height,
            bit_depth=8,
            storage_bits=8,
            channels=4,
            samples=bytes(out),
            row_stride=self.width * 4,
            info=dict(self.info),
        )

    def flatten(self, background: tuple[int, int, int] = (255, 255, 255)) -> ImageData:
        """Alpha-composite onto an opaque background with exact integer rounding.

        The result has no alpha band: ``LA`` becomes ``L`` and ``RGBA`` becomes
        ``RGB``.  ``background`` is indexed per colour band.
        """
        if self.mode not in {"LA", "RGBA"}:
            return self
        channels = self.channels
        colour = channels - 1
        out = bytearray()
        for index in range(0, len(self.samples), channels):
            alpha = self.samples[index + colour]
            for band in range(colour):
                value = self.samples[index + band]
                colour_value = background[0] if colour == 1 else background[band]
                out.append((value * alpha + colour_value * (255 - alpha) + 127) // 255)
        mode = "L" if colour == 1 else "RGB"
        return ImageData(
            format=self.format,
            mode=mode,
            width=self.width,
            height=self.height,
            bit_depth=8,
            storage_bits=8,
            channels=colour,
            samples=bytes(out),
            row_stride=self.width * colour,
            info=dict(self.info),
        )

    def normalize_orientation(self) -> ImageData:
        """Return a top-left-origin copy with the EXIF orientation applied.

        The decoder keeps samples in file order and records the orientation
        tag; this method applies it once.  Applying it again is a no-op
        because the returned image has ``orientation == 1``.
        """
        from array import array as _array

        from .codecs import rotate_samples

        if self.orientation == 1:
            return self
        values = [int(value) for value in self.samples]
        rotated, width, height = rotate_samples(
            values, self.width, self.height, self.channels, self.orientation
        )
        samples = _array("H", rotated) if self.storage_bits == 16 else bytes(rotated)
        return ImageData(
            format=self.format,
            mode=self.mode,
            width=width,
            height=height,
            bit_depth=self.bit_depth,
            storage_bits=self.storage_bits,
            channels=self.channels,
            samples=samples,
            palette=self.palette,
            transparency=self.transparency,
            row_stride=width * self.channels * (self.storage_bits // 8),
            orientation=1,
            info=dict(self.info),
        )


def _load_16bit(im: Any, mode: str) -> array[int]:
    """Return 16-bit samples in canonical unsigned form.

    ``array('H')`` interprets raw bytes in native order, while Pillow's
    ``tobytes`` for ``I;16B``/``I;16BS`` is big-endian regardless of host.
    Bytes are swapped exactly when the stored order differs from the host
    order.  Signed modes are reinterpreted as unsigned without changing bits.
    """
    raw = im.tobytes()
    stored_big = mode in {"I;16B", "I;16BS"}
    host_big = sys.byteorder == "big"
    if mode.endswith("S"):
        signed = array("h")
        signed.frombytes(raw)
        if stored_big != host_big:
            signed.byteswap()
        return array("H", (value & 0xFFFF for value in signed))
    values = array("H")
    values.frombytes(raw)
    if stored_big != host_big:
        values.byteswap()
    return values


__all__ = [
    "ImageData",
    "Palette",
    "checked_add",
    "checked_mul",
    "row_bytes",
    "scale_sample",
    "storage_bytes",
]
