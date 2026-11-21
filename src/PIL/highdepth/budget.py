#
# The Python Imaging Library.
#
# Allocation budgeting for streaming decoders and encoders.
#
# See the README file for information on usage and redistribution.
#
"""Memory budgets, row/tile iteration and cooperative cancellation.

All decoders in :mod:`PIL.highdepth` call :meth:`MemoryBudget.reserve` *before*
allocating the buffer for a row, tile or full image, so an over-budget request
fails with :class:`BudgetExceededError` instead of an opaque ``MemoryError``
raised halfway through a decode.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from .errors import BudgetExceededError, CancelledError, InvalidParameterError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .model import ImageData

DEFAULT_BUDGET = 512 * 1024 * 1024
"""Default peak allocation budget in bytes (512 MiB)."""

DEFAULT_TILE_HEIGHT = 64
"""Default streaming tile height in rows."""


class MemoryBudget:
    """Tracks reserved bytes and refuses allocations above the budget."""

    def __init__(self, limit: int | None = None) -> None:
        if limit is None:
            limit = DEFAULT_BUDGET
        limit_object: object = limit
        if not isinstance(limit_object, int) or isinstance(limit_object, bool):
            msg = "budget limit must be an integer"
            raise InvalidParameterError(msg)
        limit = limit_object
        if limit <= 0:
            msg = f"budget limit must be positive, got {limit}"
            raise InvalidParameterError(msg)
        self.limit = limit
        self._used = 0
        self._peak = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        return self._used

    @property
    def peak(self) -> int:
        return self._peak

    @property
    def remaining(self) -> int:
        return self.limit - self._used

    def reserve(self, amount: int, what: str = "buffer") -> int:
        """Reserve ``amount`` bytes, returning the amount.

        Raises :class:`BudgetExceededError` without changing state when the
        allocation would push usage above the limit.
        """
        if amount < 0:
            msg = "cannot reserve a negative amount"
            raise InvalidParameterError(msg)
        with self._lock:
            if self._used + amount > self.limit:
                msg = (
                    f"{what} of {amount} bytes exceeds the memory budget "
                    f"({self._used} used of {self.limit})"
                )
                raise BudgetExceededError(msg)
            self._used += amount
            self._peak = max(self._peak, self._used)
        return amount

    def release(self, amount: int) -> None:
        if amount < 0:
            msg = "cannot release a negative amount"
            raise InvalidParameterError(msg)
        with self._lock:
            self._used = max(0, self._used - amount)

    def check(self, amount: int, what: str = "buffer") -> None:
        """Verify that ``amount`` could be reserved, without reserving it."""
        with self._lock:
            if self._used + amount > self.limit:
                msg = (
                    f"{what} of {amount} bytes exceeds the memory budget "
                    f"({self._used} used of {self.limit})"
                )
                raise BudgetExceededError(msg)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self._used = 0


class CancellationToken:
    """A thread-safe cooperative cancellation flag.

    Decoders poll :meth:`raise_if_cancelled` between rows or tiles.  A token
    may be shared between threads; it is never mutated by the pipeline itself.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            msg = "operation cancelled"
            raise CancelledError(msg)


@dataclass(frozen=True)
class Tile:
    """A rectangular region of an image, in top-left origin coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if not (0 <= self.x0 <= self.x1 and 0 <= self.y0 <= self.y1):
            msg = f"invalid tile {self!r}"
            raise InvalidParameterError(msg)

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def iter_tiles(
    width: int,
    height: int,
    tile_height: int = DEFAULT_TILE_HEIGHT,
    tile_width: int | None = None,
    *,
    token: CancellationToken | None = None,
) -> Iterator[Tile]:
    """Yield tiles covering an image in raster order.

    ``tile_height`` and ``tile_width`` are clamped to the image size; a
    non-positive value is rejected.  The token is polled once per tile so a
    cancellation stops iteration promptly.
    """
    if tile_height <= 0:
        msg = f"tile_height must be positive, got {tile_height}"
        raise InvalidParameterError(msg)
    if tile_width is not None and tile_width <= 0:
        msg = f"tile_width must be positive, got {tile_width}"
        raise InvalidParameterError(msg)
    step_x = tile_width if tile_width is not None else width
    for y0 in range(0, height, tile_height):
        y1 = min(y0 + tile_height, height)
        for x0 in range(0, width, step_x):
            if token is not None:
                token.raise_if_cancelled()
            yield Tile(x0, y0, min(x0 + step_x, width), y1)


def iter_rows(
    data: ImageData,
    *,
    token: CancellationToken | None = None,
) -> Iterator[tuple[int, tuple[int, ...]]]:
    """Yield ``(y, samples)`` for every row, polling cancellation first."""
    for y, samples in enumerate(data.iter_rows()):
        if token is not None:
            token.raise_if_cancelled()
        yield y, samples


def check_full_image_budget(
    width: int,
    height: int,
    channels: int,
    storage_bits: int,
    budget: MemoryBudget | None,
    what: str = "image buffer",
) -> int:
    """Return the byte size of a full image, verifying it fits the budget."""
    from .model import storage_bytes

    size = storage_bytes(width, height, channels, storage_bits)
    if budget is not None:
        budget.check(size, what)
    return size


__all__ = [
    "DEFAULT_BUDGET",
    "DEFAULT_TILE_HEIGHT",
    "CancellationToken",
    "MemoryBudget",
    "Tile",
    "check_full_image_budget",
    "iter_rows",
    "iter_tiles",
]
