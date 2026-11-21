#
# The Python Imaging Library.
#
# High depth and palette processing pipeline.
#
# See the README file for information on usage and redistribution.
#
"""High depth and palette processing for Pillow.

:mod:`PIL.highdepth` adds a strict, lossless path for images that Pillow's
classic ``Image`` API cannot fully represent:

* reading PNG, TIFF and BMP into a normalized
  :class:`~PIL.highdepth.model.ImageData` that keeps 1/2/4/8/16 bit samples,
  16-bit multi-band pixels, palette indices and transparency indexes;
* deterministic palette quantization (median cut, exact octree, fixed
  palettes) with alpha-aware Floyd-Steinberg dithering and a documented
  tie-break rule;
* a fixed-stage colour management pipeline (ICC, sRGB/linear/gamma, palette,
  alpha premultiplication) that never mutates its input;
* streaming, budget-checked encoding with atomic output and multi-page TIFF
  and APNG writing;
* a concurrent batch executor that guarantees per-image ordering.

The existing ``PIL.Image``, ``PIL.ImageFile``, ``PIL.ImageCms``,
``PIL.PngImagePlugin``, ``PIL.TiffImagePlugin`` and ``PIL.BmpImagePlugin``
entry points keep working unchanged; :mod:`PIL.highdepth` is additive.
"""

from __future__ import annotations

from .budget import (
    DEFAULT_BUDGET,
    DEFAULT_TILE_HEIGHT,
    CancellationToken,
    MemoryBudget,
    Tile,
    iter_rows,
    iter_tiles,
)
from .cms import (
    ColorPipeline,
    apply_gamma,
    linear_to_srgb,
    load_profile,
    srgb_to_linear,
    unpremultiply,
)
from .cms import (
    convert as convert_colors,
)
from .errors import (
    BitDepthError,
    BudgetExceededError,
    CancelledError,
    ChecksumError,
    ChunkLengthError,
    DimensionOverflowError,
    EmptyImageError,
    FormatError,
    HighDepthError,
    IllegalPaletteError,
    InvalidParameterError,
    NonFiniteSampleError,
    ProfileError,
    SourceError,
    TrailingDataError,
    TruncatedDataError,
    UnknownColorSpaceError,
    UnsupportedFeatureError,
    WriteVerificationError,
)
from .model import ImageData, Palette, row_bytes, storage_bytes
from .quantize import QuantizeOptions, build_palette, map_pixels, quantize
from .read import (
    iter_frames,
    iter_image_tiles,
    read,
    read_frames,
    read_header,
    stream_decode,
)
from .serialize import dump, load

__all__ = [
    "DEFAULT_BUDGET",
    "DEFAULT_TILE_HEIGHT",
    "BitDepthError",
    "BudgetExceededError",
    "CancellationToken",
    "CancelledError",
    "ChecksumError",
    "ChunkLengthError",
    "ColorPipeline",
    "DimensionOverflowError",
    "EmptyImageError",
    "FormatError",
    "HighDepthError",
    "IllegalPaletteError",
    "ImageData",
    "InvalidParameterError",
    "MemoryBudget",
    "NonFiniteSampleError",
    "Palette",
    "ProfileError",
    "QuantizeOptions",
    "SourceError",
    "Tile",
    "TrailingDataError",
    "TruncatedDataError",
    "UnknownColorSpaceError",
    "UnsupportedFeatureError",
    "WriteVerificationError",
    "apply_gamma",
    "build_palette",
    "convert_colors",
    "dump",
    "iter_frames",
    "iter_image_tiles",
    "iter_rows",
    "iter_tiles",
    "linear_to_srgb",
    "load",
    "load_profile",
    "map_pixels",
    "quantize",
    "read",
    "read_frames",
    "read_header",
    "row_bytes",
    "srgb_to_linear",
    "storage_bytes",
    "stream_decode",
    "unpremultiply",
]
