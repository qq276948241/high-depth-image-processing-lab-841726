"""Streaming, serialization and cross-process reproducibility."""

from __future__ import annotations

import io
import subprocess
import sys
from array import array

import pytest

from PIL import Image
from PIL.highdepth import (
    CancellationToken,
    ImageData,
    Palette,
    iter_image_tiles,
    iter_rows,
    quantize,
    read,
    read_frames,
    serialize,
    stream_decode,
)

from .highdepth_fixtures.helpers import make_palette, make_rgb, save


def _rgb16(width: int = 7, height: int = 5) -> ImageData:
    values = array("H", [(i * 977) & 0xFFFF for i in range(width * height * 3)])
    return ImageData("PNG", "RGB", width, height, 16, 16, 3, values)


# ------------------------------------------------------------- tiles


@pytest.mark.parametrize("height, width", ((64, None), (2, 3), (1, 1)))
def test_tiles_reassemble_exactly(height: int, width: int | None) -> None:
    data = _rgb16()
    tiles = list(iter_image_tiles(data, height, tile_width=width))
    assert sum(tile.width * tile.height for _bounds, tile in tiles) == data.pixel_count
    for bounds, tile in tiles:
        for y in range(tile.height):
            for x in range(tile.width):
                assert tile.sample_at(x, y) == data.sample_at(
                    bounds.x0 + x, bounds.y0 + y
                )


def test_tile_keeps_depth_and_palette() -> None:
    palette = make_palette(4, with_alpha=True)
    data = ImageData(
        "PNG", "P", 3, 3, 2, 8, 1, bytes((0, 1, 2, 3, 2, 1, 0, 1, 2)), palette=palette
    )
    for _bounds, tile in iter_image_tiles(data, 2):
        assert tile.storage_bits == 8
        assert tile.palette == palette
        assert tile.bit_depth == data.bit_depth


def test_stream_decode_yields_tiles() -> None:
    payload = save(make_rgb(6, 4), "PNG")
    tiles = list(stream_decode(payload, tile_height=2))
    assert len(tiles) == 2
    assert tiles[0][1].width == 6


def test_iter_rows_polls_cancellation() -> None:
    data = ImageData.from_pil(make_rgb(6, 4))
    token = CancellationToken()
    rows = iter_rows(data, token=token)
    next(rows)
    token.cancel()
    with pytest.raises(Exception):
        list(rows)


# ------------------------------------------------------------- serialization


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(_rgb16(), id="rgb16"),
        pytest.param(
            ImageData(
                "PNG",
                "P",
                3,
                1,
                2,
                8,
                1,
                bytes((0, 1, 2)),
                palette=Palette(((0, 0, 0, 0), (1, 2, 3, 255), (9, 8, 7, 128))),
            ),
            id="palette-alpha",
        ),
        pytest.param(ImageData.from_pil(make_rgb()), id="rgb8"),
    ],
)
def test_serialize_roundtrip_is_lossless(data: ImageData) -> None:
    loaded = serialize.load(serialize.dump(data))
    assert loaded.sha256() == data.sha256()
    assert loaded.tobytes() == data.tobytes()
    assert loaded.palette == data.palette
    assert loaded.describe() == data.describe()


def test_serialize_rejects_garbage() -> None:
    with pytest.raises(Exception):
        serialize.load(b"not a dump")


def test_serialize_detects_truncation() -> None:
    payload = serialize.dump(_rgb16())
    with pytest.raises(Exception):
        serialize.load(payload[:-3])


# ------------------------------------------------------------- reproducibility


def test_quantize_hash_is_stable_across_processes() -> None:
    """The same image quantized twice in separate interpreters must match."""
    data = ImageData.from_pil(make_rgb(24, 12))
    local = quantize(data, 8, dither=True, seed=5).sha256()
    script = (
        "import sys;"
        "from PIL import Image;"
        "from PIL.highdepth import ImageData, quantize;"
        "im = Image.new('RGB',(24,12));"
        "px = [(x*31%256, y*67%256, (x*13+y*29)%256)"
        " for y in range(12) for x in range(24)];"
        "im.putdata(px);"
        "d = ImageData.from_pil(im);"
        "print(quantize(d, 8, dither=True, seed=5).sha256())"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == local


def test_encode_hash_is_stable_across_processes() -> None:
    script = (
        "import hashlib, io;"
        "from array import array;"
        "from PIL.highdepth import ImageData;"
        "from PIL.highdepth.encode import encode_tiff;"
        "vals = array('H', [(i*977)&0xFFFF for i in range(7*5*3)]);"
        "d = ImageData('TIFF','RGB',7,5,16,16,3,vals);"
        "out = io.BytesIO(); encode_tiff([d], out);"
        "print(hashlib.sha256(out.getvalue()).hexdigest())"
    )
    first = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout.strip()
    second = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert first == second


# ------------------------------------------------------------- multi-frame sources


def test_tiff_frames_identical_across_sources() -> None:
    frames = [ImageData.from_pil(Image.new("L", (4, 3), value)) for value in (1, 2, 3)]
    from PIL.highdepth.encode import encode_tiff

    out = io.BytesIO()
    encode_tiff(frames, out)
    payload = out.getvalue()
    direct = read_frames(payload)
    chunked = read_frames(
        iter([payload[i : i + 11] for i in range(0, len(payload), 11)])
    )
    assert [frame.sha256() for frame in direct] == [frame.sha256() for frame in chunked]


def test_png_frames_identical_across_sources() -> None:
    frames = [ImageData.from_pil(Image.new("L", (4, 3), value)) for value in (5, 6)]
    from PIL.highdepth.encode import encode_png

    out = io.BytesIO()
    encode_png(frames, out)
    payload = out.getvalue()
    direct = read_frames(payload)
    from_file = read_frames(io.BytesIO(payload))
    assert [frame.sha256() for frame in direct] == [
        frame.sha256() for frame in from_file
    ]


def test_apng_frame_metadata() -> None:
    data = read(_apng_payload())
    assert data.info["apng_frames"] == 3
    assert len(data.info["apng_control"]) == 3


def _apng_payload() -> bytes:
    frames = [
        ImageData.from_pil(Image.new("L", (4, 3), value)) for value in (10, 20, 30)
    ]
    from PIL.highdepth.encode import encode_png

    out = io.BytesIO()
    encode_png(frames, out)
    return out.getvalue()


def test_apng_extra_frames_have_expected_pixels() -> None:
    frames = read_frames(_apng_payload())
    assert [frame.row_samples(0)[0] for frame in frames] == [10, 20, 30]


def test_png_actl_frame_count_must_match_stream() -> None:
    import struct
    import zlib

    payload = bytearray(_apng_payload())
    position = payload.find(b"acTL")
    assert position > 0
    # Rewrite acTL with a frame count that disagrees with the stream, keeping
    # the chunk CRC valid so the frame-count check is what rejects it.
    length_offset = position - 4
    body = b"acTL" + struct.pack(">II", 2, 0)
    payload[length_offset : position + 8] = (
        struct.pack(">I", 8) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )
    with pytest.raises(Exception):
        read(bytes(payload))
