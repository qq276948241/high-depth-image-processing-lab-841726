#
# The Python Imaging Library.
#
# Deterministic high depth quantization.
#
# Pillow's core quantizer works on 8-bit "L" and "RGB" data, and on "RGBA" with
# the fast octree method.  It cannot quantize "I;16", "I", "LA" or palette with
# alpha ("PA") images.  This module adds a pure Python path for those modes and
# for 16-bit luminance, keeping the source sample depth.
#
# All decisions are fixed so the same input, palette size, parameters and seed
# produce identical bytes:
#
# * colours are compared with a squared distance in source sample units;
# * palette entries are ordered by their channel tuple, ascending;
# * nearest-colour ties go to the lowest palette index;
# * median cut splits the widest channel (ties to the lowest channel), then
#   the box with the largest population, then the lowest box index;
# * error diffusion uses Floyd-Steinberg weights with integer arithmetic,
#   left to right and top to bottom, and rounding half away from zero.
#
# Alpha uses premultiplied colour, so transparent edges do not darken.  Fully
# transparent pixels do not diffuse colour into their neighbours.  Samples are
# never shifted down before the colour decision.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

__lazy_modules__ = {"sys"}

import sys
from typing import Literal, NamedTuple

from . import Image

TYPE_CHECKING = False
if TYPE_CHECKING:
    from .HighDepth import MemoryBudget

__all__ = [
    "HIGH_DEPTH_MODES",
    "MAX_QUANTIZE_COLORS",
    "QuantizeError",
    "quantize",
    "quantize_to_palette",
    "quantize_to_web_palette",
]

HIGH_DEPTH_MODES = frozenset({"I;16", "I;16L", "I;16B", "I;16N", "I", "LA", "PA"})
"""Modes that use this module instead of the core 8-bit quantizer."""

MAX_QUANTIZE_COLORS = 256

# Pillow's browser palette: ten leading black entries and a 6x6x6 colour cube.
_WEB_PALETTE = bytes(10 * 3) + bytes(
    value
    for blue in range(0, 256, 51)
    for green in range(0, 256, 51)
    for red in range(0, 256, 51)
    for value in (red, green, blue)
)


class QuantizeError(ValueError):
    """Raised for quantization requests that cannot be satisfied."""


class _Samples(NamedTuple):
    """Source pixels, normalised to a colour + alpha representation.

    ``colors`` holds one tuple per pixel.  Grayscale modes use one channel,
    colour modes three, and alpha modes a trailing alpha channel.  Values are
    in the source sample units, so ``max_value`` is 65535 for 16-bit data.
    """

    width: int
    height: int
    colors: tuple[tuple[int, ...], ...]
    alpha: bool
    max_value: int

    @property
    def channels(self) -> int:
        return len(self.colors[0]) if self.colors else (2 if self.alpha else 1)


def _to_int(value: int, max_value: int) -> int:
    if value < 0:
        return 0
    if value > max_value:
        return max_value
    return value


def _samples(image: Image.Image) -> _Samples:
    mode = image.mode
    width, height = image.size
    data = image.tobytes()
    count = width * height
    if mode in ("L", "P"):
        return _Samples(width, height, tuple((value,) for value in data), False, 255)
    if mode == "LA":
        return _Samples(
            width,
            height,
            tuple((data[i], data[i + 1]) for i in range(0, count * 2, 2)),
            True,
            255,
        )
    if mode == "PA":
        palette = image.getpalette("RGBA") or []
        entries = [tuple(palette[i : i + 4]) for i in range(0, len(palette) - 3, 4)]
        colors = []
        for index, alpha in zip(data[0::2], data[1::2]):
            red, green, blue, palette_alpha = (
                entries[index] if index < len(entries) else (0, 0, 0, 255)
            )
            colors.append((red, green, blue, (alpha * palette_alpha + 127) // 255))
        return _Samples(width, height, tuple(colors), True, 255)
    if mode in ("RGB", "RGBX", "RGBA", "RGBa"):
        channels = len(mode)
        pixels = tuple(
            tuple(data[i : i + channels]) for i in range(0, count * channels, channels)
        )
        return _Samples(width, height, pixels, mode in ("RGBA", "RGBa"), 255)
    if mode == "I":
        values = tuple(
            _to_int(int.from_bytes(data[i : i + 4], sys.byteorder, signed=True), 65535)
            for i in range(0, count * 4, 4)
        )
        return _Samples(width, height, tuple((v,) for v in values), False, 65535)
    if mode.startswith("I;16"):
        if mode in ("I;16", "I;16L"):
            order: Literal["little", "big"] = (
                "little" if mode == "I;16" and sys.byteorder == "little" else "big"
            )
        elif mode == "I;16B":
            order = "big"
        else:
            order = sys.byteorder
        values = tuple(
            int.from_bytes(data[i : i + 2], order) for i in range(0, count * 2, 2)
        )
        return _Samples(width, height, tuple((v,) for v in values), False, 65535)
    msg = f"cannot quantize images in mode {mode}"
    raise QuantizeError(msg)


def _premultiply(
    color: tuple[int, ...], alpha: bool, max_value: int
) -> tuple[int, ...]:
    if not alpha:
        return color
    a = color[-1]
    return tuple(value * a for value in color[:-1]) + (a * a,)


def _distance(
    a: tuple[int, ...], b: tuple[int, ...], alpha: bool, max_value: int
) -> int:
    if alpha:
        premultiplied_a = _premultiply(a, alpha, max_value)
        premultiplied_b = _premultiply(b, alpha, max_value)
        total = 0
        for index in range(len(premultiplied_a) - 1):
            diff = premultiplied_a[index] - premultiplied_b[index]
            total += diff * diff
        # Colour terms are scaled by alpha; normalise them to the source scale
        # before adding the alpha term so both contribute comparably.
        total //= max_value * max_value
        alpha_diff = a[-1] - b[-1]
        return total + alpha_diff * alpha_diff
    total = 0
    for left, right in zip(a, b):
        diff = left - right
        total += diff * diff
    return total


def _median_cut(
    histogram: dict[tuple[int, ...], int],
    colors: int,
    alpha: bool,
    max_value: int,
) -> tuple[tuple[int, ...], ...]:
    entries = sorted(histogram.items())
    if not entries:
        return ()
    channel_count = len(entries[0][0])
    boxes: list[list[tuple[tuple[int, ...], int]]] = [entries]
    while len(boxes) < colors:
        best_index = -1
        best_key: tuple[int, int, int] | None = None
        for index, box in enumerate(boxes):
            if len(box) < 2:
                continue
            ranges = [
                max(color[channel] for color, _ in box)
                - min(color[channel] for color, _ in box)
                for channel in range(channel_count)
            ]
            key = (max(ranges), sum(count for _, count in box), -index)
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index < 0:
            break
        box = boxes.pop(best_index)
        ranges = [
            max(color[channel] for color, _ in box)
            - min(color[channel] for color, _ in box)
            for channel in range(channel_count)
        ]
        split_channel = ranges.index(max(ranges))
        box.sort(key=lambda item: (item[0][split_channel], item[0]))
        total = sum(count for _, count in box)
        running = 0
        split_at = 1
        for offset, (_, count) in enumerate(box):
            running += count
            if running * 2 >= total:
                split_at = offset + 1
                break
        split_at = max(1, min(len(box) - 1, split_at))
        boxes.append(box[:split_at])
        boxes.append(box[split_at:])
    representatives = sorted({_representative(box, alpha) for box in boxes})
    return tuple(representatives[:colors])


def _representative(
    box: list[tuple[tuple[int, ...], int]], alpha: bool
) -> tuple[int, ...]:
    channel_count = len(box[0][0])
    total = sum(count for _, count in box)
    if not alpha:
        return tuple(
            (sum(color[channel] * count for color, count in box) + total // 2) // total
            for channel in range(channel_count)
        )
    alpha_total = sum(color[-1] * count for color, count in box)
    if alpha_total:
        values = [
            (
                sum(color[channel] * color[-1] * count for color, count in box)
                + alpha_total // 2
            )
            // alpha_total
            for channel in range(channel_count - 1)
        ]
    else:
        values = [0] * (channel_count - 1)
    values.append((alpha_total + total // 2) // total)
    return tuple(values)


def _nearest(
    color: tuple[int, ...],
    palette: tuple[tuple[int, ...], ...],
    alpha: bool,
    max_value: int,
) -> int:
    best_index = 0
    best_distance: int | None = None
    for index, entry in enumerate(palette):
        distance = _distance(color, entry, alpha, max_value)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _clamp(value: int, max_value: int) -> int:
    if value < 0:
        return 0
    if value > max_value:
        return max_value
    return value


def _round_divide(value: int, divisor: int) -> int:
    if value >= 0:
        return (value + divisor // 2) // divisor
    return -((-value + divisor // 2) // divisor)


def _diffuse(samples: _Samples, palette: tuple[tuple[int, ...], ...]) -> bytes:
    width, height = samples.width, samples.height
    max_value = samples.max_value
    alpha = samples.alpha
    pixels = samples.colors
    channels = samples.channels
    output = bytearray(width * height)
    if width == 0 or height == 0 or not palette:
        return bytes(output)
    current = [[0] * (width * channels) for _ in range(2)]
    for y in range(height):
        row, next_row = current[0], current[1]
        for x in range(width):
            base = x * channels
            source = pixels[y * width + x]
            adjusted = tuple(
                _clamp(source[c] + row[base + c], max_value) for c in range(channels)
            )
            index = _nearest(adjusted, palette, alpha, max_value)
            output[y * width + x] = index
            entry = palette[index]
            error = [adjusted[c] - entry[c] for c in range(channels)]
            if alpha and source[-1] == 0:
                # Fully transparent pixels do not carry colour error.
                error = [0] * (channels - 1) + [error[-1]]
            if x + 1 < width:
                target = base + channels
                for c in range(channels):
                    row[target + c] += _round_divide(error[c] * 7, 16)
            if y + 1 < height:
                if x > 0:
                    target = base - channels
                    for c in range(channels):
                        next_row[target + c] += _round_divide(error[c] * 3, 16)
                for c in range(channels):
                    next_row[base + c] += _round_divide(error[c] * 5, 16)
                if x + 1 < width:
                    target = base + channels
                    for c in range(channels):
                        next_row[target + c] += _round_divide(error[c], 16)
        current[0], current[1] = current[1], [0] * (width * channels)
    return bytes(output)


def _histogram(samples: _Samples) -> dict[tuple[int, ...], int]:
    histogram: dict[tuple[int, ...], int] = {}
    for color in samples.colors:
        histogram[color] = histogram.get(color, 0) + 1
    return histogram


def _scale_to_byte(value: int, max_value: int) -> int:
    if max_value == 255:
        return value
    return min(255, (value * 255 + max_value // 2) // max_value)


def palette_bytes(
    colors: tuple[tuple[int, ...], ...], alpha: bool, max_value: int
) -> tuple[bytes, str]:
    """Return an 8-bit palette and its raw mode ("RGB" or "RGBA").

    Grayscale palettes are written as grey RGB triples so they can be attached
    to a "P" image.  Alpha palettes are written as RGBA quads; a two channel
    (luminance, alpha) palette becomes (grey, grey, grey, alpha).
    """

    out = bytearray()
    for color in colors:
        if alpha:
            if len(color) == 2:
                grey = _scale_to_byte(color[0], max_value)
                out += bytes((grey, grey, grey, _scale_to_byte(color[1], max_value)))
            else:
                out += bytes(_scale_to_byte(value, max_value) for value in color[:4])
        else:
            if len(color) == 1:
                grey = _scale_to_byte(color[0], max_value)
                out += bytes((grey, grey, grey))
            else:
                out += bytes(_scale_to_byte(value, max_value) for value in color[:3])
    return bytes(out), ("RGBA" if alpha else "RGB")


def _check_colors(colors: int) -> None:
    if not 1 <= colors <= MAX_QUANTIZE_COLORS:
        msg = f"colors must be between 1 and {MAX_QUANTIZE_COLORS}"
        raise QuantizeError(msg)


def _result(
    image: Image.Image,
    indices: bytes,
    colors: tuple[tuple[int, ...], ...],
    alpha: bool,
    max_value: int,
) -> Image.Image:
    result = Image.frombytes("P", image.size, indices)
    data, mode = palette_bytes(colors, alpha, max_value)
    result.putpalette(data, mode)
    result.load()
    return result


def quantize(
    image: Image.Image,
    colors: int = 256,
    *,
    dither: bool = True,
    budget: MemoryBudget | None = None,
) -> Image.Image:
    """Quantize a high depth image to at most ``colors`` palette entries.

    :param image: an image in a mode listed in :data:`HIGH_DEPTH_MODES`.
    :param colors: target palette size, 1 to 256.
    :param dither: apply Floyd-Steinberg error diffusion.
    :param budget: optional :class:`PIL.HighDepth.MemoryBudget`, checked before
        pixel samples are materialised.
    :returns: a "P" image; its palette is RGBA when the input has alpha.
    """

    _check_colors(colors)
    image.load()
    samples = _samples(image)
    if budget is not None:
        budget.check(
            image.size,
            channels=max(1, samples.channels),
            bytes_per_sample=2 if samples.max_value > 255 else 1,
        )
    if not samples.colors:
        return _result(image, b"", (), samples.alpha, samples.max_value)
    palette = _median_cut(_histogram(samples), colors, samples.alpha, samples.max_value)
    if dither:
        indices = _diffuse(samples, palette)
    else:
        indices = bytes(
            _nearest(color, palette, samples.alpha, samples.max_value)
            for color in samples.colors
        )
    return _result(image, indices, palette, samples.alpha, samples.max_value)


def _reference_palette(
    reference: Image.Image, samples: _Samples
) -> tuple[tuple[tuple[int, ...], ...], bytes, str]:
    palette_data = reference.getpalette("RGBA")
    if not palette_data:
        msg = "empty palette"
        raise QuantizeError(msg)
    entries = [
        tuple(palette_data[i : i + 4]) for i in range(0, len(palette_data) - 3, 4)
    ]
    max_value = samples.max_value
    adapted: list[tuple[int, ...]] = []
    for entry in entries:
        red, green, blue, alpha = entry
        if samples.alpha:
            if samples.channels == 2:
                grey = (red * 299 + green * 587 + blue * 114) // 1000
                adapted.append(
                    (
                        grey * max_value // 255,
                        alpha * max_value // 255,
                    )
                )
            else:
                adapted.append(tuple(value * max_value // 255 for value in entry))
        elif samples.channels == 1:
            grey = (red * 299 + green * 587 + blue * 114) // 1000
            adapted.append((grey * max_value // 255,))
        else:
            adapted.append(
                (
                    red * max_value // 255,
                    green * max_value // 255,
                    blue * max_value // 255,
                )
            )
    return tuple(adapted), bytes(palette_data), reference.palette.mode  # type: ignore[union-attr]


def quantize_to_palette(
    image: Image.Image,
    reference: Image.Image,
    *,
    dither: bool = True,
    budget: MemoryBudget | None = None,
) -> Image.Image:
    """Map a high depth image to an existing palette image.

    ``reference`` must be a "P" image.  Errors are diffused in the input's
    sample units, so 16-bit input is never shifted down before the colour
    decision.
    """

    image.load()
    reference.load()
    if reference.mode != "P":
        msg = "bad mode for palette image"
        raise QuantizeError(msg)
    if reference.palette is None or not reference.palette.palette:
        msg = "empty palette"
        raise QuantizeError(msg)
    samples = _samples(image)
    if budget is not None:
        budget.check(
            image.size,
            channels=max(1, samples.channels),
            bytes_per_sample=2 if samples.max_value > 255 else 1,
        )
    palette, palette_data, palette_mode = _reference_palette(reference, samples)
    if not samples.colors:
        result = Image.new("P", image.size)
        result.palette = reference.palette.copy()
        result.palette.dirty = 1
        result.load()
        return result
    if dither:
        indices = _diffuse(samples, palette)
    else:
        indices = bytes(
            _nearest(color, palette, samples.alpha, samples.max_value)
            for color in samples.colors
        )
    result = Image.frombytes("P", image.size, indices)
    result.palette = reference.palette.copy()
    result.palette.dirty = 1
    result.load()
    return result


def quantize_to_web_palette(
    image: Image.Image,
    *,
    dither: bool = True,
    budget: MemoryBudget | None = None,
) -> Image.Image:
    """Map a high depth image to Pillow's fixed browser palette."""

    image.load()
    reference = Image.new("P", (1, 1))
    reference.putpalette(_WEB_PALETTE)
    return quantize_to_palette(image, reference, dither=dither, budget=budget)
