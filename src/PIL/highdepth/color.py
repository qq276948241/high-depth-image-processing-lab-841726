#
# The Python Imaging Library.
#
# Known colour spaces, transfer functions and gamma handling.
#
# See the README file for information on usage and redistribution.
#
"""Colour space definitions shared by the pipeline and ICC stages.

Values are normalized to ``[0.0, 1.0]`` as doubles.  The transfer functions
below are the exact ones named by the sRGB and linear-light spaces; they are
kept independent of littleCMS so that a transform can be validated against
them without an ICC backend.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .errors import (
    InvalidParameterError,
    NonFiniteSampleError,
    UnknownColorSpaceError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

LINEAR_ALPHA = "linear"
"""Marker used in ``info`` when a buffer is premultiplied."""

KNOWN_SPACES = ("linear", "srgb", "gamma", "palette", "gray", "cmyk", "lab", "xyz")


def check_space(space: str) -> str:
    """Validate a colour space name, returning it."""
    space_object: object = space
    if not isinstance(space_object, str):
        msg = f"colour space must be a string, got {type(space_object).__name__}"
        raise UnknownColorSpaceError(msg)
    normalized = space_object.strip().lower()
    if normalized not in KNOWN_SPACES:
        msg = (
            f"unknown colour space {space!r}; expected one of "
            f"{', '.join(KNOWN_SPACES)}"
        )
        raise UnknownColorSpaceError(msg)
    return normalized


def srgb_to_linear(sample: float) -> float:
    """Decode one sRGB-encoded sample in ``[0, 1]`` to linear light."""
    if sample <= 0.04045:
        return sample / 12.92
    return ((sample + 0.055) / 1.055) ** 2.4


def linear_to_srgb(sample: float) -> float:
    """Encode one linear-light sample in ``[0, 1]`` with the sRGB curve."""
    if sample <= 0.0031308:
        return sample * 12.92
    return 1.055 * (sample ** (1 / 2.4)) - 0.055


def gamma_decode(sample: float, gamma: float) -> float:
    if not math.isfinite(gamma) or gamma <= 0:
        msg = f"gamma must be a positive finite number, got {gamma!r}"
        raise InvalidParameterError(msg)
    if not math.isfinite(sample):
        msg = f"non-finite sample {sample!r}"
        raise NonFiniteSampleError(msg)
    return max(0.0, min(1.0, sample)) ** gamma


def gamma_encode(sample: float, gamma: float) -> float:
    if not math.isfinite(gamma) or gamma <= 0:
        msg = f"gamma must be a positive finite number, got {gamma!r}"
        raise InvalidParameterError(msg)
    if not math.isfinite(sample):
        msg = f"non-finite sample {sample!r}"
        raise NonFiniteSampleError(msg)
    return max(0.0, min(1.0, sample)) ** (1.0 / gamma)


def to_linear_function(
    space: str, gamma: float | None = None
) -> Callable[[float], float]:
    """Return the decode function for ``space``."""
    normalized = check_space(space)
    if normalized == "srgb":
        return srgb_to_linear
    if normalized == "linear":
        return lambda value: value
    if normalized == "gamma":
        if gamma is None:
            msg = "the gamma space requires a gamma value"
            raise InvalidParameterError(msg)
        gamma_decode(0.0, gamma)  # validate the exponent eagerly
        return lambda value: gamma_decode(value, gamma)
    msg = f"colour space {space!r} has no scalar transfer function"
    raise UnknownColorSpaceError(msg)


def from_linear_function(
    space: str, gamma: float | None = None
) -> Callable[[float], float]:
    normalized = check_space(space)
    if normalized == "srgb":
        return linear_to_srgb
    if normalized == "linear":
        return lambda value: value
    if normalized == "gamma":
        if gamma is None:
            msg = "the gamma space requires a gamma value"
            raise InvalidParameterError(msg)
        return lambda value: gamma_encode(value, gamma)
    msg = f"colour space {space!r} has no scalar transfer function"
    raise UnknownColorSpaceError(msg)


def quantize_unit(value: float, bits: int, *, alpha: bool = False) -> int:
    """Round a normalized sample to an integer of ``bits`` precision.

    Round-half-up is applied without float truncation bias: the scaled value
    is rounded with :func:`round` after adding one half ULP in integer space.
    Values outside ``[0, 1]`` are clamped unless ``alpha`` is set, in which
    case negative alpha is clamped to zero.
    """
    if not 0 < bits <= 16:
        msg = f"unsupported sample bits {bits}"
        raise InvalidParameterError(msg)
    if not math.isfinite(value):
        msg = f"non-finite sample {value!r}"
        raise NonFiniteSampleError(msg)
    if value < 0.0:
        if not alpha:
            msg_0 = f"negative sample {value!r}"
            raise InvalidParameterError(msg_0)
        value = 0.0
    elif value > 1.0:
        value = 1.0
    maximum = (1 << bits) - 1
    return min(maximum, int(value * maximum + 0.5))


__all__ = [
    "KNOWN_SPACES",
    "check_space",
    "from_linear_function",
    "gamma_decode",
    "gamma_encode",
    "linear_to_srgb",
    "quantize_unit",
    "srgb_to_linear",
    "to_linear_function",
]
