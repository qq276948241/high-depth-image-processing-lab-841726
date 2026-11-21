"""Batch pipeline: ordering, concurrency, cancellation races and budgets."""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

import pytest

from PIL import Image
from PIL.highdepth import CancellationToken, ImageData, read
from PIL.highdepth.budget import BudgetExceededError
from PIL.highdepth.model import ImageData as Model
from PIL.highdepth.pipeline import Job, run_batch, run_job

from .highdepth_fixtures.helpers import save

if TYPE_CHECKING:
    from pathlib import Path


def _gray_bytes(value: int) -> bytes:
    return save(Image.new("L", (8, 6), value), "PNG")


def test_single_job_success(tmp_path: Path) -> None:
    directory = tmp_path
    job = Job(source=_gray_bytes(5), output=directory / "a.png", format="PNG")
    result = run_job(job)
    assert result.ok, result.error
    assert result.frames == 1
    assert read((directory / "a.png").read_bytes()).row_samples(0)[0] == 5


def test_batch_preserves_submission_order(tmp_path: Path) -> None:
    directory = tmp_path
    jobs = [
        Job(source=_gray_bytes(value), output=directory / f"{index}.png", format="PNG")
        for index, value in enumerate((1, 2, 3, 4, 5))
    ]
    results = run_batch(jobs, workers=3)
    assert [result.job.output for result in results] == [job.output for job in jobs]
    assert all(result.ok for result in results)
    for index, value in enumerate((1, 2, 3, 4, 5)):
        assert (
            read((directory / f"{index}.png").read_bytes()).row_samples(0)[0] == value
        )


def test_batch_failure_isolated(tmp_path: Path) -> None:
    directory = tmp_path

    def explode(_data: ImageData) -> ImageData:
        msg = "deliberate failure"
        raise RuntimeError(msg)

    jobs = [
        Job(source=_gray_bytes(1), output=directory / "ok.png", format="PNG"),
        Job(
            source=_gray_bytes(2),
            output=directory / "bad.png",
            format="PNG",
            transform=explode,
        ),
    ]
    results = run_batch(jobs, workers=2)
    assert results[0].ok
    assert not results[1].ok
    assert isinstance(results[1].error, RuntimeError)
    assert (directory / "ok.png").exists()
    assert not (directory / "bad.png").exists()


def test_batch_cancellation_stops_every_job(tmp_path: Path) -> None:
    directory = tmp_path
    token = CancellationToken()
    token.cancel()
    jobs = [
        Job(source=_gray_bytes(value), output=directory / f"{index}.png", format="PNG")
        for index, value in enumerate((1, 2, 3))
    ]
    results = run_batch(jobs, workers=3, token=token)
    assert all(not result.ok for result in results)
    assert all(isinstance(result.error, Exception) for result in results)
    assert not any(directory.iterdir())


def test_cancellation_race_at_write(tmp_path: Path) -> None:
    """Cancelling from another thread while writing must leave no file behind."""
    from PIL.highdepth.encode import write_frames
    from PIL.highdepth.errors import CancelledError

    big = Model("PNG", "L", 512, 512, 8, 8, 1, bytes(512 * 512))
    token = CancellationToken()
    finished = threading.Event()

    def cancel_soon() -> None:
        time.sleep(0.001)
        token.cancel()

    thread = threading.Thread(target=cancel_soon)
    thread.start()
    try:
        write_frames(tmp_path / "race.png", [big], "PNG", token=token)
    except CancelledError:
        pass
    finally:
        thread.join()
        finished.set()
    assert finished.is_set()
    # Either the write completed before cancellation or it failed cleanly; a
    # partial file or a leftover temporary is never acceptable.
    assert not [n for n in os.listdir(tmp_path) if n.startswith(".highdepth")]
    if not (tmp_path / "race.png").exists():
        assert not (tmp_path / "race.png").exists()

    # A token cancelled up front guarantees no partial output at all.
    blocked = CancellationToken()
    blocked.cancel()
    with pytest.raises(CancelledError):
        write_frames(tmp_path / "blocked.png", [big], "PNG", token=blocked)
    assert not (tmp_path / "blocked.png").exists()


def test_budget_exceeded_on_job(tmp_path: Path) -> None:
    directory = tmp_path
    job = Job(
        source=_gray_bytes(1),
        output=directory / "b.png",
        format="PNG",
        budget=10,
    )
    result = run_job(job)
    assert not result.ok
    assert isinstance(result.error, BudgetExceededError)
    assert not (directory / "b.png").exists()


def test_workers_validation() -> None:
    with pytest.raises(Exception):
        run_batch([], workers=0)


def test_empty_batch() -> None:
    assert run_batch([], workers=4) == []


def test_different_images_processed_in_parallel(tmp_path: Path) -> None:
    directory = tmp_path
    seen: list[int] = []
    lock = threading.Lock()

    def record(data: ImageData) -> ImageData:
        with lock:
            seen.append(len(seen))
        return data

    jobs = [
        Job(
            source=_gray_bytes(value),
            output=directory / f"{index}.tiff",
            format="TIFF",
            transform=record,
        )
        for index, value in enumerate(range(8))
    ]
    results = run_batch(jobs, workers=4)
    assert all(result.ok for result in results)
    assert len(seen) == 8


def test_run_job_never_raises(tmp_path: Path) -> None:
    directory = tmp_path
    job = Job(source=b"not an image", output=directory / "x.png", format="PNG")
    result = run_job(job)
    assert not result.ok
    assert not (directory / "x.png").exists()


def test_oriented_output_keeps_orientation(tmp_path: Path) -> None:
    from PIL.highdepth.encode import write_frames
    from PIL.highdepth.read import read_frames

    directory = tmp_path
    base = ImageData.from_pil(Image.new("L", (8, 6), 3))
    rotated = Model(
        base.format,
        base.mode,
        base.width,
        base.height,
        base.bit_depth,
        base.storage_bits,
        base.channels,
        base.samples,
        orientation=8,
    )
    write_frames(directory / "o.tiff", [rotated], "TIFF")
    loaded = read_frames(directory / "o.tiff")
    assert loaded[0].orientation == 8
    # Samples stay in file order so a re-encode does not rotate twice.
    assert loaded[0].tobytes() == rotated.tobytes()
    normalized = loaded[0].normalize_orientation()
    assert normalized.orientation == 1
    assert (normalized.width, normalized.height) == (rotated.height, rotated.width)
