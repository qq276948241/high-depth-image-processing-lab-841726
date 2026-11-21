#
# The Python Imaging Library.
#
# Deterministic palette quantization for the high depth pipeline.
#
# See the README file for information on usage and redistribution.
#
"""Palette quantization.

The implementation is intentionally pure Python and single threaded so that
the same inputs, palette size, dither setting and seed always reproduce the
same bytes.  Two algorithms are provided:

``median_cut``
    The classic variance-style median cut, refined with optional k-means
    iterations.  Splits and bucket ordering are fully specified by the
    documented tie-break rules below.

``octree``
    A lossless index builder over an exact colour histogram, used when every
    colour must be retained.

Colour distance
    Squared Euclidean distance in RGBA space.  Alpha is weighted by
    :data:`ALPHA_WEIGHT` so that a transparent pixel is not dragged towards an
    opaque neighbour of similar RGB.

Transparent pixels
    A pixel with alpha ``0`` is "fully transparent".  Its RGB is irrelevant to
    the colour distance: all fully transparent pixels are mapped to a single
    dedicated palette entry whose RGB is the mean of the transparent pixels
    (so compositing never invents a new hue) and whose alpha is zero.  The
    dedicated entry is always index 0 when any fully transparent pixel exists.

Deterministic ordering and tie-breaks
    * Palette entries are sorted by ``(alpha, red, green, blue)`` ascending.
    * When two candidate colours have exactly equal quantized distance to two
      palette entries, the entry with the lower index wins.
    * Median cut splits the box on the channel with the largest range; ties go
      to R, then G, then B, then A.  The split position is the first entry
      whose cumulative count reaches half the box population, computed on the
      histogram sorted by ``(channel value, red, green, blue, alpha)``.
    * Empty buckets are never emitted, so a request for more colours than
      exist yields the exact distinct colours and no padding entry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import (
    BitDepthError,
    EmptyImageError,
    IllegalPaletteError,
    InvalidParameterError,
)
from .model import ImageData, Palette

RGBA = tuple[int, int, int, int]
RGBAPixel = RGBA

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .budget import CancellationToken

ALPHA_WEIGHT = 2.0
"""Weight applied to the squared alpha difference in the colour distance."""

MAX_COLORS = 256


@dataclass(frozen=True)
class QuantizeOptions:
    """Fully resolved quantization parameters."""

    colors: int = 256
    method: str = "median"
    dither: bool = True
    seed: int = 0
    max_dither_error: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.colors, int) or not 1 <= self.colors <= MAX_COLORS:
            msg = f"colors must be an integer in 1..{MAX_COLORS}, got {self.colors!r}"
            raise InvalidParameterError(msg)
        if self.method not in {"median", "octree", "palette"}:
            msg = f"unknown quantization method {self.method!r}"
            raise InvalidParameterError(msg)
        seed_object: object = self.seed
        if not isinstance(seed_object, int) or isinstance(seed_object, bool):
            msg = "seed must be an integer"
            raise InvalidParameterError(msg)


def _rgba_samples(data: ImageData) -> tuple[list[tuple[int, int, int, int]], int]:
    """Return ``(pixels, bit_depth_written)`` as 8-bit RGBA tuples.

    A 16-bit source is rounded to 8 bits *explicitly* here; callers that must
    not reduce depth use :func:`quantize` directly and receive
    :class:`BitDepthError` when the palette cannot represent the samples.
    """
    if data.mode == "P":
        assert data.palette is not None
        pixels = [data.palette.entries[int(i)] for i in data.samples]
        return pixels, data.bit_depth
    if data.mode == "RGBA":
        src = data.samples
        return [
            (src[i], src[i + 1], src[i + 2], src[i + 3]) for i in range(0, len(src), 4)
        ], data.bit_depth
    if data.mode == "RGB":
        src = data.samples
        return [
            (src[i], src[i + 1], src[i + 2], 255) for i in range(0, len(src), 3)
        ], data.bit_depth
    if data.mode == "LA":
        src = data.samples
        return [
            (src[i], src[i], src[i], src[i + 1]) for i in range(0, len(src), 2)
        ], data.bit_depth
    if data.mode == "L":
        src = data.samples
        return [(value, value, value, 255) for value in src], data.bit_depth
    msg = f"cannot quantize mode {data.mode!r}"
    raise BitDepthError(msg)


def _require_image_data(data: object) -> ImageData:
    if not isinstance(data, ImageData):
        msg = f"expected ImageData, got {type(data).__name__}"
        raise InvalidParameterError(msg)
    return data


def _require_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{name} must be an integer"
        raise InvalidParameterError(msg)
    return value


def _require_quantizable(data: ImageData) -> None:
    if data.storage_bits != 8:
        msg = (
            f"quantizing {data.storage_bits}-bit samples would reduce the bit "
            "depth; export or resample explicitly first"
        )
        raise BitDepthError(msg)
    if data.mode == "P" and data.bit_depth > 8:
        msg = "cannot quantize palette indices deeper than 8 bits"
        raise BitDepthError(msg)


def _distance(
    a: RGBA,
    b: RGBA,
    alpha_weight: float = ALPHA_WEIGHT,
) -> float:
    if a[3] == 0 and b[3] == 0:
        return 0.0
    dr = a[0] - b[0]
    dg = a[1] - b[1]
    db = a[2] - b[2]
    da = a[3] - b[3]
    return dr * dr + dg * dg + db * db + alpha_weight * da * da


def _sort_key(entry: RGBA) -> RGBA:
    return (entry[3], entry[0], entry[1], entry[2])


def _histogram(
    pixels: Iterable[tuple[int, int, int, int]],
) -> dict[tuple[int, int, int, int], int]:
    counts: dict[tuple[int, int, int, int], int] = {}
    for pixel in pixels:
        counts[pixel] = counts.get(pixel, 0) + 1
    return counts


def _merge_transparent(
    counts: dict[tuple[int, int, int, int], int],
) -> tuple[dict[tuple[int, int, int, int], int], tuple[int, int, int, int] | None]:
    """Collapse fully transparent colours into one representative entry."""
    transparent_total = 0
    red_total = green_total = blue_total = 0
    remaining: dict[tuple[int, int, int, int], int] = {}
    for entry, count in counts.items():
        if entry[3] == 0:
            transparent_total += count
            red_total += entry[0] * count
            green_total += entry[1] * count
            blue_total += entry[2] * count
        else:
            remaining[entry] = count
    if not transparent_total:
        return remaining, None
    representative = (
        (red_total + transparent_total // 2) // transparent_total,
        (green_total + transparent_total // 2) // transparent_total,
        (blue_total + transparent_total // 2) // transparent_total,
        0,
    )
    remaining[representative] = remaining.get(representative, 0) + transparent_total
    return remaining, representative


class _Box:
    __slots__ = ("entries", "count", "representative")

    def __init__(self, entries: list[tuple[RGBA, int]]) -> None:
        self.entries = entries
        self.count = sum(count for _entry, count in entries)
        total = [0, 0, 0, 0]
        for entry, count in entries:
            for band in range(4):
                total[band] += entry[band] * count
        self.representative: RGBA = (
            (total[0] + self.count // 2) // self.count,
            (total[1] + self.count // 2) // self.count,
            (total[2] + self.count // 2) // self.count,
            (total[3] + self.count // 2) // self.count,
        )

    def ranges(self) -> tuple[int, int, int, int]:
        minima = [255, 255, 255, 255]
        maxima = [0, 0, 0, 0]
        for entry, _count in self.entries:
            for band in range(4):
                minima[band] = min(minima[band], entry[band])
                maxima[band] = max(maxima[band], entry[band])
        return tuple(maxima[band] - minima[band] for band in range(4))  # type: ignore[return-value]


def _split_box(box: _Box) -> tuple[_Box, _Box] | None:
    if len(box.entries) < 2:
        return None
    ranges = box.ranges()
    # Largest range wins; ties resolve R, G, B, A in that order.
    channel = max(range(4), key=lambda band: (ranges[band], -band))
    ordered = sorted(
        box.entries, key=lambda item: (item[0][channel], *_sort_key(item[0]))
    )
    half = box.count / 2
    cumulative = 0
    split_at = 1
    for index, (_entry, count) in enumerate(ordered):
        cumulative += count
        if cumulative >= half and index + 1 < len(ordered):
            split_at = index + 1
            break
    else:
        split_at = max(1, len(ordered) // 2)
    if split_at <= 0 or split_at >= len(ordered):
        return None
    return _Box(ordered[:split_at]), _Box(ordered[split_at:])


def _median_cut(
    counts: dict[tuple[int, int, int, int], int], colors: int
) -> list[tuple[int, int, int, int]]:
    entries = sorted(counts.items(), key=lambda item: _sort_key(item[0]))
    boxes = [_Box(entries)]
    while len(boxes) < colors:
        splittable = [box for box in boxes if len(box.entries) > 1]
        if not splittable:
            break
        # Split the box with the largest pixel population; ties prefer the
        # lower creation order (list order), keeping the result deterministic.
        target = max(
            range(len(boxes)),
            key=lambda index: (boxes[index].count, -index),
        )
        if len(boxes[target].entries) < 2:
            break
        halves = _split_box(boxes[target])
        if halves is None:
            break
        boxes[target] = halves[0]
        boxes.insert(target + 1, halves[1])
    return [box.representative for box in boxes]


def _kmeans(
    counts: dict[tuple[int, int, int, int], int],
    centers: list[tuple[int, int, int, int]],
    iterations: int,
) -> list[tuple[int, int, int, int]]:
    points = list(counts.items())
    for _iteration in range(iterations):
        buckets: list[list[tuple[RGBA, int]]] = [[] for _ in centers]
        for entry, count in points:
            best_index = 0
            best_distance = _distance(entry, centers[0])
            for index in range(1, len(centers)):
                distance = _distance(entry, centers[index])
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            buckets[best_index].append((entry, count))
        updated: list[RGBA] = []
        for index, bucket in enumerate(buckets):
            if not bucket:
                updated.append(centers[index])
                continue
            totals = [0, 0, 0, 0]
            total_count = 0
            for entry, count in bucket:
                for band in range(4):
                    totals[band] += entry[band] * count
                total_count += count
            updated.append(
                (
                    (totals[0] + total_count // 2) // total_count,
                    (totals[1] + total_count // 2) // total_count,
                    (totals[2] + total_count // 2) // total_count,
                    (totals[3] + total_count // 2) // total_count,
                )
            )
        if updated == centers:
            break
        centers = updated
    return centers


def build_palette(
    data: ImageData,
    colors: int = 256,
    *,
    method: str = "median",
    kmeans: int = 0,
    alpha_weight: float = ALPHA_WEIGHT,
) -> Palette:
    """Build a deterministic palette for ``data``.

    ``kmeans`` is the number of refinement passes applied to a median cut
    palette; it is ignored for the ``octree`` and ``palette`` methods.
    """
    if not 1 <= colors <= MAX_COLORS:
        msg = f"colors must be in 1..{MAX_COLORS}, got {colors}"
        raise InvalidParameterError(msg)
    if not data.pixel_count:
        msg = "cannot quantize an empty image"
        raise EmptyImageError(msg)
    if kmeans < 0:
        msg = "kmeans must not be negative"
        raise InvalidParameterError(msg)
    pixels, _depth = _rgba_samples(data)
    counts = _histogram(pixels)
    counts, transparent = _merge_transparent(counts)

    if not counts:
        # The image is entirely transparent.
        assert transparent is not None
        return Palette((transparent,))

    if method == "octree" or (method == "median" and len(counts) <= colors):
        entries = sorted(counts, key=_sort_key)
    elif method == "median":
        entries = _median_cut(counts, colors)
        if kmeans:
            entries = _kmeans(counts, entries, kmeans)
    else:  # pragma: no cover - validated by QuantizeOptions
        msg = f"unknown quantization method {method!r}"
        raise InvalidParameterError(msg)

    palette_entries = sorted(
        {*entries, *([transparent] if transparent else [])}, key=_sort_key
    )
    if transparent is not None:
        # Fully transparent pixels always map to index 0 when present.
        palette_entries = [transparent] + [
            entry for entry in palette_entries if entry != transparent
        ]
    if len(palette_entries) > MAX_COLORS:
        msg = f"palette has {len(palette_entries)} entries, maximum is {MAX_COLORS}"
        raise IllegalPaletteError(msg)
    return Palette(tuple(palette_entries))


def _nearest(
    pixel: RGBA,
    entries: list[RGBA],
    alpha_weight: float,
) -> int:
    """Return the nearest palette index, preferring the lower index on ties."""
    best_index = 0
    best_distance = _distance(pixel, entries[0], alpha_weight)
    for index in range(1, len(entries)):
        distance = _distance(pixel, entries[index], alpha_weight)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def map_pixels(
    data: ImageData,
    palette: Palette,
    *,
    dither: bool = True,
    seed: int = 0,
    alpha_weight: float = ALPHA_WEIGHT,
    token: CancellationToken | None = None,
) -> ImageData:
    """Map ``data`` to palette indices, optionally with Floyd-Steinberg dithering.

    Dithering is **alpha aware**: error is only diffused onto neighbours that
    are fully opaque and whose source alpha matches the current pixel's alpha
    exactly.  A transparent or partially transparent pixel therefore never
    receives or emits error, which is what stops transparent edges from being
    darkened.  ``seed`` deterministically perturbs the initial error value of
    achromatic runs; for a given seed the output is byte-for-byte stable.
    """
    pixels, depth = _rgba_samples(data)
    entries = list(palette.entries)
    width, height = data.width, data.height
    if len(pixels) != width * height:
        msg = "pixel count does not match the image size"
        raise InvalidParameterError(msg)
    indices = [0] * len(pixels)
    if not dither:
        for index, pixel in enumerate(pixels):
            if token is not None and index % 4096 == 0:
                token.raise_if_cancelled()
            indices[index] = _nearest(pixel, entries, alpha_weight)
    else:
        seed_offset = int(hashlib.sha256(str(seed).encode()).hexdigest()[:4], 16) % 16
        # Two rows of per-channel error, one entry per column plus one guard
        # slot on the right so weights never need a bounds check.
        current = [[0.0, 0.0, 0.0, 0.0] for _ in range(width + 1)]
        following = [[0.0, 0.0, 0.0, 0.0] for _ in range(width + 1)]
        for y in range(height):
            if token is not None:
                token.raise_if_cancelled()
            for x in range(width):
                index = y * width + x
                source = pixels[index]
                adjusted = [
                    max(0.0, min(255.0, source[band] + current[x][band]))
                    for band in range(4)
                ]
                quantized: RGBA = (
                    int(adjusted[0] + 0.5),
                    int(adjusted[1] + 0.5),
                    int(adjusted[2] + 0.5),
                    int(adjusted[3] + 0.5),
                )
                best = _nearest(quantized, entries, alpha_weight)
                indices[index] = best
                if source[3] != 255:
                    # Fully or partially transparent pixels neither emit nor
                    # receive error, which keeps transparent edges untainted.
                    continue
                target = entries[best]
                residual = [adjusted[band] - target[band] for band in range(4)]
                if (
                    y == 0
                    and x == 0
                    and seed_offset
                    and residual[0] == residual[1] == residual[2]
                ):
                    # Seeded perturbation of an achromatic starting pixel; the
                    # value is derived only from ``seed`` and its position.
                    residual[0] += float((seed_offset % 3) - 1)
                if x + 1 < width:
                    current[x + 1] = [
                        current[x + 1][band] + residual[band] * (7 / 16)
                        for band in range(4)
                    ]
                if x > 0:
                    following[x - 1] = [
                        following[x - 1][band] + residual[band] * (3 / 16)
                        for band in range(4)
                    ]
                following[x] = [
                    following[x][band] + residual[band] * (5 / 16) for band in range(4)
                ]
                if x + 1 < width:
                    following[x + 1] = [
                        following[x + 1][band] + residual[band] * (1 / 16)
                        for band in range(4)
                    ]
            current = following
            following = [[0.0, 0.0, 0.0, 0.0] for _ in range(width + 1)]

    _require_quantizable(data)
    bit_depth = max(1, (len(entries) - 1).bit_length())
    if depth > 8:
        msg = "palette output cannot carry samples deeper than 8 bits"
        raise BitDepthError(msg)
    return ImageData(
        format=data.format,
        mode="P",
        width=width,
        height=height,
        bit_depth=bit_depth,
        storage_bits=8,
        channels=1,
        samples=bytes(indices),
        palette=palette,
        transparency=0 if palette.entries[0][3] == 0 else None,
        row_stride=width,
        orientation=data.orientation,
        info=dict(data.info),
    )


def quantize(
    data: ImageData,
    colors: int = 256,
    method: str = "median",
    *,
    dither: bool = True,
    seed: int = 0,
    kmeans: int = 0,
    palette: Palette | None = None,
    alpha_weight: float = ALPHA_WEIGHT,
    token: CancellationToken | None = None,
) -> ImageData:
    """Quantize ``data`` to a palette image.

    When ``palette`` is given the image is mapped to that fixed palette and
    ``colors``/``method``/``kmeans`` are ignored.  A supplied palette is never
    reordered, so reproduction is exact for the caller's table.
    """
    data = _require_image_data(data)
    if data.is_empty():
        msg = "cannot quantize an empty image"
        raise EmptyImageError(msg)
    _require_quantizable(data)
    if palette is None:
        palette = build_palette(
            data, colors, method=method, kmeans=kmeans, alpha_weight=alpha_weight
        )
    else:
        if len(palette) == 0:  # pragma: no cover - Palette forbids this
            msg = "empty palette"
            raise IllegalPaletteError(msg)
    return map_pixels(
        data,
        palette,
        dither=dither,
        seed=seed,
        alpha_weight=alpha_weight,
        token=token,
    )


def validate_reproducible(data: ImageData, **options: object) -> str:
    """Return a hash that must be identical across runs for identical input."""
    first = quantize(data, **options)  # type: ignore[arg-type]
    second = quantize(data, **options)  # type: ignore[arg-type]
    if first.tobytes() != second.tobytes() or first.palette != second.palette:
        msg = "quantization is not reproducible for identical inputs"
        raise IllegalPaletteError(msg)
    return first.sha256()


__all__ = [
    "ALPHA_WEIGHT",
    "MAX_COLORS",
    "QuantizeOptions",
    "build_palette",
    "map_pixels",
    "quantize",
    "validate_reproducible",
]
