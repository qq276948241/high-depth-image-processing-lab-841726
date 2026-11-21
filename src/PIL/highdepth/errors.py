#
# The Python Imaging Library.
#
# Stable error hierarchy for the high depth / palette pipeline.
#
# See the README file for information on usage and redistribution.
#
"""Stable, programmatically identifiable errors.

Every error carries a ``code`` attribute.  The codes, types and message
prefixes are part of the public contract of :mod:`PIL.highdepth`: callers may
match on them instead of parsing human readable text.

All errors derive from :class:`HighDepthError`; format errors additionally
derive from :class:`ValueError` and resource errors from :class:`MemoryError`
so that existing ``except ValueError`` / ``except MemoryError`` handlers keep
working.
"""

from __future__ import annotations


class HighDepthError(Exception):
    """Base class for all errors raised by :mod:`PIL.highdepth`."""

    code = "highdepth-error"


class SourceError(HighDepthError, OSError):
    """The input source could not be read as a byte stream."""

    code = "source-error"


class FormatError(HighDepthError, ValueError):
    """The input is not a well formed file of the detected format."""

    code = "format-error"


class UnsupportedFeatureError(FormatError):
    """The file is valid but uses a feature this implementation rejects."""

    code = "unsupported-feature"


class TruncatedDataError(FormatError):
    """The file ends before all declared data has been read."""

    code = "truncated-data"


class ChunkLengthError(FormatError):
    """A container chunk, strip or palette declares an invalid length."""

    code = "chunk-length"


class ChecksumError(FormatError):
    """A container checksum does not match the payload."""

    code = "checksum"


class TrailingDataError(FormatError):
    """Bytes follow the end of the logical image stream."""

    code = "trailing-data"


class IllegalPaletteError(FormatError):
    """A colour table is missing, malformed or out of range."""

    code = "illegal-palette"


class DimensionOverflowError(FormatError, OverflowError):
    """A size, stride or offset computation overflowed the addressable range."""

    code = "dimension-overflow"


class BudgetExceededError(HighDepthError, MemoryError):
    """An allocation was refused because it exceeds the configured budget."""

    code = "budget-exceeded"


class BitDepthError(HighDepthError, ValueError):
    """An operation would silently reduce the sample bit depth."""

    code = "bit-depth"


class EmptyImageError(HighDepthError, ValueError):
    """An operation was requested on an image with no pixels."""

    code = "empty-image"


class InvalidParameterError(HighDepthError, ValueError):
    """A parameter is outside its documented range."""

    code = "invalid-parameter"


class ProfileError(HighDepthError, ValueError):
    """An ICC profile is missing, corrupt or incompatible with the image."""

    code = "profile-error"


class UnknownColorSpaceError(ProfileError):
    """A colour space name is not known to the pipeline."""

    code = "unknown-colorspace"


class NonFiniteSampleError(HighDepthError, ValueError):
    """A floating point image contains NaN or Infinity."""

    code = "non-finite-sample"


class CancelledError(HighDepthError):
    """Processing was cancelled by the caller."""

    code = "cancelled"


class WriteVerificationError(HighDepthError, OSError):
    """A written file did not round-trip to the requested frames."""

    code = "write-verification"
