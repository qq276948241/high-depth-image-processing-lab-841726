#
# The Python Imaging Library.
#
# Fixed-stage ICC colour management for the high depth pipeline.
#
# See the README file for information on usage and redistribution.
#
"""Colour management pipeline.

A conversion is a fixed sequence of stages.  Each stage is explicit, and a
failure in any stage leaves the input image and the input profile untouched:

1. **decode**: palette indices expand to RGBA through the image's own table
   (no pixel is turned black for a transparent entry);
2. **linearize**: a named transfer function (``srgb``, ``linear`` or
   ``gamma``) maps samples into linear light and then into the target
   encoding;
3. **icc**: when both profiles are supplied, littleCMS maps between them;
4. **alpha**: alpha is either left untouched or premultiplied *after* the
   colour transform, never before it, so rounding cannot bleed into edges;
5. **quantize**: an optional final quantize stage, always last.

Samples are carried as IEEE-754 doubles between stages and rounded to integers
**once**, at the end, with half-up rounding.  That is what makes
``srgb -> linear -> srgb`` an exact identity for integer inputs instead of
losing low-light precision at every hop.  The output sample width always
equals the input sample width; no stage silently reduces it.  The one
exception is the ICC stage, whose engine is 8-bit: asking it to transform a
16-bit image raises :class:`BitDepthError` rather than truncating.

The stage list is recorded in ``ImageData.info["cms_stages"]`` so callers can
assert which path ran.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .color import (
    check_space,
    from_linear_function,
    quantize_unit,
    to_linear_function,
)
from .errors import (
    BitDepthError,
    InvalidParameterError,
    ProfileError,
)
from .model import ImageData

if TYPE_CHECKING:
    from typing import Any


@dataclass
class _Working:
    """Float sample buffer in normalized units, between stages."""

    mode: str
    width: int
    height: int
    channels: int
    values: list[float]
    bits: int
    info: dict[str, Any]

    @property
    def alpha_band(self) -> int | None:
        return self.channels - 1 if self.mode in {"LA", "RGBA"} else None


def _to_working(data: ImageData) -> _Working:
    if data.mode == "P":
        data = data.to_rgba()
    channels = data.channels
    bits = data.storage_bits
    maximum = (1 << bits) - 1 if bits == 16 else 255
    values = [value / maximum for value in data.samples]
    return _Working(
        mode=data.mode,
        width=data.width,
        height=data.height,
        channels=channels,
        values=values,
        bits=bits,
        info=dict(data.info),
    )


def _to_image(working: _Working) -> ImageData:
    from array import array

    alpha_band = working.alpha_band
    quantized = [
        quantize_unit(
            value,
            working.bits,
            alpha=alpha_band is not None and index % working.channels == alpha_band,
        )
        for index, value in enumerate(working.values)
    ]
    samples: bytes | array[int] = (
        array("H", quantized) if working.bits == 16 else bytes(quantized)
    )
    return ImageData(
        format="",
        mode=working.mode,
        width=working.width,
        height=working.height,
        bit_depth=working.bits,
        storage_bits=working.bits,
        channels=working.channels,
        samples=samples,
        row_stride=working.width * working.channels * (working.bits // 8),
        info=dict(working.info),
    )


@dataclass(frozen=True)
class ColorPipeline:
    """A resolved, immutable colour conversion plan.

    Instances are cheap to share between threads: :meth:`apply` keeps no
    mutable state on the instance.  A lazily built littleCMS transform is
    cached under a lock and used read-only afterwards.
    """

    source_space: str = "srgb"
    target_space: str = "srgb"
    gamma: float | None = None
    target_gamma: float | None = None
    source_profile: bytes | None = None
    target_profile: bytes | None = None
    intent: int = 0
    premultiply: bool = False
    output_bits: int | None = None
    quantize_colors: int | None = None
    quantize_dither: bool = True
    quantize_seed: int = 0
    _transform_cache: dict[str, Any] = field(
        default_factory=dict, compare=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        check_space(self.source_space)
        check_space(self.target_space)
        if not isinstance(self.intent, int) or not 0 <= self.intent <= 3:
            msg = f"rendering intent must be in 0..3, got {self.intent!r}"
            raise ProfileError(msg)
        if self.source_profile is not None:
            _check_profile_bytes(self.source_profile, "source")
        if self.target_profile is not None:
            _check_profile_bytes(self.target_profile, "target")
        if (self.source_profile is None) != (self.target_profile is None):
            msg = "both an input and an output profile are required for ICC conversion"
            raise ProfileError(msg)
        if self.output_bits is not None and self.output_bits not in (8, 16):
            msg = f"output_bits must be 8 or 16, got {self.output_bits!r}"
            raise InvalidParameterError(msg)
        if self.quantize_colors is not None and not (
            isinstance(self.quantize_colors, int) and 1 <= self.quantize_colors <= 256
        ):
            msg = f"quantize_colors must be in 1..256, got {self.quantize_colors!r}"
            raise InvalidParameterError(msg)
        # Validate the transfer functions eagerly so a bad gamma fails at
        # construction, before any image is touched.
        to_linear_function(self.source_space, self.gamma)
        from_linear_function(self.target_space, self.target_gamma)

    @property
    def has_icc(self) -> bool:
        return self.source_profile is not None

    def describe(self) -> dict[str, Any]:
        return {
            "source_space": self.source_space,
            "target_space": self.target_space,
            "gamma": self.gamma,
            "target_gamma": self.target_gamma,
            "icc": self.has_icc,
            "premultiply": self.premultiply,
            "output_bits": self.output_bits,
            "quantize_colors": self.quantize_colors,
            "quantize_dither": self.quantize_dither,
            "quantize_seed": self.quantize_seed,
            "intent": self.intent,
        }

    def digest(self) -> str:
        """Stable hash of the plan, excluding cached transforms."""
        digest = hashlib.sha256()
        for key, value in sorted(self.describe().items()):
            digest.update(f"{key}={value!r};".encode())
        for name in ("source_profile", "target_profile"):
            profile = getattr(self, name)
            if profile is not None:
                digest.update(hashlib.sha256(profile).digest())
        return digest.hexdigest()

    def apply(self, data: ImageData) -> ImageData:
        """Run the stage sequence, returning a new :class:`ImageData`."""
        data_object: object = data
        if not isinstance(data_object, ImageData):
            msg = f"expected ImageData, got {type(data_object).__name__}"
            raise InvalidParameterError(msg)
        data = data_object
        stages: list[str] = []
        working = _to_working(data)
        if data.mode == "P":
            stages.append("decode")

        # Linearize / re-encode in doubles; a single rounding happens at the end.
        if self.source_space != self.target_space or self.source_space == "gamma":
            _transfer(working, self)
            stages.append("linearize")

        # ICC operates on 8-bit encoded samples.
        if self.has_icc:
            working = self._apply_icc(working)
            stages.append("icc")

        if self.output_bits is not None:
            working.bits = self.output_bits

        # Alpha boundary.
        if self.premultiply and working.alpha_band is not None:
            _premultiply_working(working)
            stages.append("alpha")
            working.info["premultiplied"] = True
        else:
            working.info["premultiplied"] = False

        result = _to_image(working)

        if self.quantize_colors is not None:
            from .quantize import quantize

            if result.storage_bits != 8:
                # Quantization is an 8-bit output stage; reducing 16-bit to a
                # palette is an explicit request in this case.
                result = result.to_pil(allow_reduce=True)
                result = ImageData.from_pil(result)
            result = quantize(
                result,
                self.quantize_colors,
                dither=self.quantize_dither,
                seed=self.quantize_seed,
            )
            stages.append("quantize")

        stages.append("done")
        return result.with_info(
            cms_stages=tuple(stages),
            cms_source=self.source_space,
            cms_target=self.target_space,
            cms_plan=self.digest(),
        )

    def _apply_icc(self, working: _Working) -> _Working:
        if working.bits != 8:
            msg = (
                "the ICC stage operates on 8-bit samples; reduce to 8 bits "
                "explicitly before applying an ICC transform to 16-bit data"
            )
            raise BitDepthError(msg)
        if working.mode not in {"RGB", "RGBA"}:
            # Gray and gray+alpha expand to RGB for the transform; alpha is
            # carried separately and re-attached without transformation.
            working = _expand_gray(working)
        transform = self._cached_transform()
        image = _to_image(working)
        colour = image.to_pil()
        try:
            output = transform.apply(colour)
        except Exception as exc:
            msg = f"ICC transform failed: {exc}"
            raise ProfileError(msg) from exc
        transformed = ImageData.from_pil(output, format=image.format)
        result = _to_working(transformed)
        result.info = {**working.info, **result.info}
        if self.target_profile is not None:
            result.info["icc_profile"] = bytes(self.target_profile)
        return result

    def _cached_transform(self) -> Any:
        key = self.digest()
        with self._lock:
            cached = self._transform_cache.get(key)
        if cached is not None:
            return cached
        assert self.source_profile is not None
        assert self.target_profile is not None
        try:
            from io import BytesIO

            from PIL import ImageCms

            source = ImageCms.ImageCmsProfile(BytesIO(bytes(self.source_profile)))
            target = ImageCms.ImageCmsProfile(BytesIO(bytes(self.target_profile)))
            transform = ImageCms.buildTransform(
                source,
                target,
                "RGB",
                "RGB",
                renderingIntent=ImageCms.Intent(self.intent),
            )
        except Exception as exc:
            msg = f"cannot build ICC transform: {exc}"
            raise ProfileError(msg) from exc
        with self._lock:
            self._transform_cache[key] = transform
        return transform


def _check_profile_bytes(profile: bytes, name: str) -> None:
    profile_object: object = profile
    if not isinstance(profile_object, (bytes, bytearray, memoryview)):
        msg = f"{name} profile must be bytes-like"
        raise ProfileError(msg)
    data = bytes(profile_object)
    if not data:
        msg = f"{name} ICC profile is empty"
        raise ProfileError(msg)
    if len(data) < 128:
        msg = f"{name} ICC profile is truncated ({len(data)} bytes)"
        raise ProfileError(msg)


def load_profile(path: str) -> bytes:
    """Read an ICC profile, raising :class:`ProfileError` on failure."""
    try:
        with open(path, "rb") as fp:
            data = fp.read()
    except OSError as exc:
        msg = f"cannot read ICC profile {path!r}: {exc}"
        raise ProfileError(msg) from exc
    _check_profile_bytes(data, path)
    return data


def _transfer(working: _Working, plan: ColorPipeline) -> None:
    decode = to_linear_function(plan.source_space, plan.gamma)
    encode = from_linear_function(plan.target_space, plan.target_gamma)
    alpha_band = working.alpha_band
    for index, value in enumerate(working.values):
        if alpha_band is not None and index % working.channels == alpha_band:
            continue
        working.values[index] = encode(decode(value))


def _premultiply_working(working: _Working) -> None:
    channels = working.channels
    alpha_band = working.alpha_band
    assert alpha_band is not None
    for index in range(0, len(working.values), channels):
        alpha = working.values[index + alpha_band]
        for band in range(alpha_band):
            working.values[index + band] *= alpha


def _expand_gray(working: _Working) -> _Working:
    """Expand L/LA to RGB/RGBA, replicating the grey sample three times."""
    out: list[float] = []
    if working.mode == "L":
        for value in working.values:
            out.extend((value, value, value))
        return _Working(
            "RGB", working.width, working.height, 3, out, working.bits, working.info
        )
    for index in range(0, len(working.values), 2):
        grey, alpha = working.values[index], working.values[index + 1]
        out.extend((grey, grey, grey, alpha))
    return _Working(
        "RGBA", working.width, working.height, 4, out, working.bits, working.info
    )


def _premultiply(data: ImageData) -> ImageData:
    """Multiply colour bands by alpha with exact half-up integer rounding."""
    channels = data.channels
    colour = channels - 1
    out = bytearray()
    for index in range(0, len(data.samples), channels):
        alpha = data.samples[index + colour]
        for band in range(colour):
            value = data.samples[index + band]
            out.append((value * alpha + 127) // 255)
        out.append(alpha)
    return data.with_samples(bytes(out)).with_info(premultiplied=True)


def _unpremultiply(data: ImageData) -> ImageData:
    channels = data.channels
    colour = channels - 1
    out = bytearray()
    for index in range(0, len(data.samples), channels):
        alpha = data.samples[index + colour]
        for band in range(colour):
            value = data.samples[index + band]
            out.append(min(255, (value * 255 + alpha // 2) // alpha) if alpha else 0)
        out.append(alpha)
    return data.with_samples(bytes(out)).with_info(premultiplied=False)


def unpremultiply(data: ImageData) -> ImageData:
    """Inverse of the premultiply stage; exposed for round-trip tests."""
    return _unpremultiply(data)


def srgb_to_linear(data: ImageData, *, output_bits: int | None = None) -> ImageData:
    return ColorPipeline("srgb", "linear", output_bits=output_bits).apply(data)


def linear_to_srgb(data: ImageData, *, output_bits: int | None = None) -> ImageData:
    return ColorPipeline("linear", "srgb", output_bits=output_bits).apply(data)


def apply_gamma(
    data: ImageData, gamma: float, *, output_bits: int | None = None
) -> ImageData:
    """Encode a linear-light image with a pure gamma curve.

    The input is assumed linear; see :func:`srgb_to_linear` to decode sRGB
    first.  ``output_bits`` selects an 8- or 16-bit intermediate.
    """
    return ColorPipeline(
        "linear", "gamma", target_gamma=gamma, output_bits=output_bits
    ).apply(data)


def convert(
    data: ImageData,
    source_space: str = "srgb",
    target_space: str = "srgb",
    *,
    gamma: float | None = None,
    target_gamma: float | None = None,
    source_profile: bytes | str | None = None,
    target_profile: bytes | str | None = None,
    intent: int = 0,
    premultiply: bool = False,
    output_bits: int | None = None,
    quantize_colors: int | None = None,
    quantize_dither: bool = True,
    quantize_seed: int = 0,
) -> ImageData:
    """Convenience wrapper around :class:`ColorPipeline`."""
    source_bytes = (
        load_profile(source_profile)
        if isinstance(source_profile, str)
        else source_profile
    )
    target_bytes = (
        load_profile(target_profile)
        if isinstance(target_profile, str)
        else target_profile
    )
    plan = ColorPipeline(
        source_space=source_space,
        target_space=target_space,
        gamma=gamma,
        target_gamma=target_gamma,
        source_profile=source_bytes,
        target_profile=target_bytes,
        intent=intent,
        premultiply=premultiply,
        output_bits=output_bits,
        quantize_colors=quantize_colors,
        quantize_dither=quantize_dither,
        quantize_seed=quantize_seed,
    )
    return plan.apply(data)


def check_repeatable(data: ImageData, **options: Any) -> ImageData:
    """Apply a conversion twice and assert bit-identical, non-mutating output."""
    original_hash = data.sha256()
    first = convert(data, **options)
    second = convert(data, **options)
    if first.tobytes() != second.tobytes():
        msg = "colour conversion is not repeatable for identical input"
        raise ProfileError(msg)
    if data.sha256() != original_hash:
        msg = "colour conversion modified its input image"
        raise ProfileError(msg)
    return first


__all__ = [
    "ColorPipeline",
    "apply_gamma",
    "check_repeatable",
    "convert",
    "linear_to_srgb",
    "load_profile",
    "srgb_to_linear",
    "unpremultiply",
]
