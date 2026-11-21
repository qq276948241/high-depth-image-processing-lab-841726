#
# The Python Imaging Library.
#
# Batch pipeline with deterministic ordering and atomic outputs.
#
# See the README file for information on usage and redistribution.
#
"""Concurrent batch processing.

Guarantees, by construction:

* **Per-image ordering.** Each job's output is written once, after all of its
  frames succeed.  Frames within a file never interleave.
* **Cross-image parallelism.** Jobs run on a thread pool.  No mutable state is
  shared: each job gets its own :class:`MemoryBudget` and the colour plan
  (if any) is immutable and caches only a read-only transform.
* **All-or-nothing output.** A path output is written to a temporary file and
  moved into place with :func:`os.replace`.  Cancellation or an exception
  removes the temporary file, so no half-written image is ever visible.
* **Cancellation stops later blocks.** The token is polled between rows and
  tiles; once cancelled, no further chunks are read or written.
"""

from __future__ import annotations

import concurrent.futures as futures
import dataclasses
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .budget import CancellationToken, MemoryBudget
from .errors import CancelledError, InvalidParameterError
from .read import read_frames

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from PIL.highdepth._typing import SourceLike, StrOrBytesPath

    from .cms import ColorPipeline
    from .model import ImageData


@dataclass
class Job:
    """One input -> output unit of work."""

    source: SourceLike
    output: StrOrBytesPath
    format: str = "PNG"
    budget: int | None = None
    pipeline: ColorPipeline | None = None
    transform: Callable[[ImageData], ImageData] | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.format.upper() not in {"PNG", "TIFF", "BMP"}:
            msg = f"unsupported output format {self.format!r}"
            raise InvalidParameterError(msg)


@dataclass
class JobResult:
    job: Job
    ok: bool
    error: BaseException | None = None
    frames: int = 0

    @property
    def output(self) -> StrOrBytesPath:
        return self.job.output


def apply_transform(
    data: ImageData, job: Job, token: CancellationToken | None
) -> ImageData:
    """Apply the job's stages to one frame, in the documented order."""
    if job.pipeline is not None:
        data = job.pipeline.apply(data)
    if job.transform is not None:
        data = job.transform(data)
    if token is not None:
        token.raise_if_cancelled()
    return data


def run_job(job: Job, token: CancellationToken | None = None) -> JobResult:
    """Process one job.  Never raises; the failure is captured in the result."""
    budget = MemoryBudget(job.budget)
    try:
        if token is not None:
            token.raise_if_cancelled()
        frames = read_frames(job.source, budget=budget, token=token)
        outputs = [apply_transform(frame, job, token) for frame in frames]
        if token is not None:
            token.raise_if_cancelled()
        from .encode import write_frames

        write_frames(
            job.output,
            outputs,
            job.format,
            budget=budget,
            token=token,
            **job.options,
        )
        return JobResult(job=job, ok=True, frames=len(outputs))
    except CancelledError as exc:
        return JobResult(job=job, ok=False, error=exc)
    except BaseException as exc:
        return JobResult(job=job, ok=False, error=exc)


def run_batch(
    jobs: Sequence[Job],
    *,
    workers: int = 1,
    token: CancellationToken | None = None,
) -> list[JobResult]:
    """Run jobs concurrently and return results in submission order.

    With ``workers=1`` the jobs run serially on the calling thread, which keeps
    debugging deterministic.  Results always align with the input order, so a
    caller can zip them back to its own list.
    """
    if workers < 1:
        msg = f"workers must be positive, got {workers}"
        raise InvalidParameterError(msg)
    results: list[JobResult | None] = [None] * len(jobs)
    if workers == 1 or len(jobs) <= 1:
        for index, job in enumerate(jobs):
            results[index] = run_job(job, token)
        return [result for result in results if result is not None]

    with futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="highdepth"
    ) as pool:
        pending = {
            pool.submit(run_job, job, token): index for index, job in enumerate(jobs)
        }
        for future in futures.as_completed(pending):
            index = pending[future]
            try:
                results[index] = future.result()
            except BaseException as exc:  # pragma: no cover - run_job catches
                results[index] = JobResult(job=jobs[index], ok=False, error=exc)
    return [result for result in results if result is not None]


class SerialExecutor:
    """Minimal ordered executor used when a caller wants explicit control."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._closed = False

    def submit(self, job: Job, token: CancellationToken | None = None) -> JobResult:
        with self._lock:
            if self._closed:
                msg = "executor is closed"
                raise RuntimeError(msg)
        return run_job(job, token)

    def close(self) -> None:
        with self._lock:
            self._closed = True


def plan_output(job: Job) -> dict[str, object]:
    """Describe a job without running it; useful for dry runs and tests."""
    return {
        "output": job.output,
        "format": job.format,
        "budget": job.budget,
        "has_pipeline": job.pipeline is not None,
        "has_transform": job.transform is not None,
        "options": dict(job.options),
    }


def with_transform(job: Job, transform: Callable[[ImageData], ImageData]) -> Job:
    """Return a copy of ``job`` with a different transform (immutably)."""
    return dataclasses.replace(job, transform=transform)


__all__ = [
    "Job",
    "JobResult",
    "SerialExecutor",
    "apply_transform",
    "plan_output",
    "run_batch",
    "run_job",
    "with_transform",
]
