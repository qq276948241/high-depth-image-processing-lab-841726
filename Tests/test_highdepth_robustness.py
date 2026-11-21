"""Robustness: mutated inputs must produce stable, typed errors only."""

from __future__ import annotations

import io
import random

import pytest

from PIL import Image
from PIL.highdepth import errors, read

IMAGE_SOURCES = ("PNG", "BMP", "TIFF")


def _base_payloads() -> list[tuple[str, bytes]]:
    payloads = []
    im = Image.new("RGB", (6, 4))
    im.putdata([(i * 3 % 256, i * 7 % 256, i * 11 % 256) for i in range(24)])
    for format in IMAGE_SOURCES:
        out = io.BytesIO()
        im.save(out, format)
        payloads.append((format, out.getvalue()))
    return payloads


def _mutate(payload: bytes, rng: random.Random) -> bytes:
    data = bytearray(payload)
    kind = rng.choice(["flip", "truncate", "insert", "zero"])
    if kind == "flip" and data:
        for _ in range(rng.randint(1, 4)):
            index = rng.randrange(len(data))
            data[index] ^= 1 << rng.randrange(8)
    elif kind == "truncate":
        data = data[: rng.randrange(len(data) + 1)]
    elif kind == "insert":
        index = rng.randrange(len(data) + 1)
        data[index:index] = bytes(rng.randrange(256) for _ in range(rng.randint(1, 6)))
    elif data:
        index = rng.randrange(len(data))
        data[index : index + 4] = b"\x00\x00\x00\x00"
    return bytes(data)


@pytest.mark.parametrize("seed", (0, 1, 2, 3))
def test_mutated_inputs_raise_only_highdepth_errors(seed: int) -> None:
    rng = random.Random(seed)
    payloads = _base_payloads()
    for _ in range(150):
        format, payload = rng.choice(payloads)
        try:
            read(_mutate(payload, rng), format=format)
        except errors.HighDepthError:
            # Expected: every malformed input maps to a documented error.
            pass
        except MemoryError:
            # An honest resource error is acceptable; a stray IndexError or
            # KeyError from parsing is not.
            pass


def test_error_inheritance_contract() -> None:
    assert issubclass(errors.FormatError, ValueError)
    assert issubclass(errors.BudgetExceededError, MemoryError)
    assert issubclass(errors.BitDepthError, ValueError)
    assert issubclass(errors.ProfileError, ValueError)
    assert issubclass(errors.TruncatedDataError, ValueError)
    assert issubclass(errors.DimensionOverflowError, OverflowError)
    assert issubclass(errors.SourceError, OSError)


def test_error_codes_are_unique_and_stable() -> None:
    codes: dict[str, list[str]] = {}
    for name in dir(errors):
        obj = getattr(errors, name)
        if isinstance(obj, type) and issubclass(obj, errors.HighDepthError):
            codes.setdefault(obj.code, []).append(name)
    duplicates = {code: names for code, names in codes.items() if len(names) > 1}
    assert duplicates == {}
    for code in codes:
        assert code
        assert code.replace("-", "").isalnum()


def test_tiff_tag_with_zero_count_is_rejected() -> None:
    # A tag declaring zero values has no scalar; that must be a FormatError,
    # not a bare IndexError escaping from the parser.
    from PIL.highdepth.encode import encode_tiff
    from PIL.highdepth.model import ImageData

    out = io.BytesIO()
    encode_tiff([ImageData("TIFF", "L", 2, 2, 8, 8, 1, bytes((1, 2, 3, 4)))], out)
    payload = bytearray(out.getvalue())
    ifd = int.from_bytes(payload[4:8], "little")
    count = int.from_bytes(payload[ifd : ifd + 2], "little")
    for index in range(count):
        base = ifd + 2 + 12 * index
        tag = int.from_bytes(payload[base : base + 2], "little")
        if tag == 256:  # ImageWidth
            payload[base + 4 : base + 8] = b"\x00\x00\x00\x00"
            break
    with pytest.raises(errors.FormatError):
        read(bytes(payload))
