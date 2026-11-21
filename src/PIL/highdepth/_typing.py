"""Type aliases used by :mod:`PIL.highdepth` without importing Pillow eagerly."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import IO, Protocol, runtime_checkable

    from PIL.highdepth.source import Source

    ReadableSource = (
        Source | bytes | bytearray | memoryview | str | IO[bytes] | Iterable[bytes]
    )

    @runtime_checkable
    class Readable(Protocol):
        def read(self, size: int = ...) -> bytes: ...

    SourceLike = ReadableSource

StrOrBytesPath = str | bytes | os.PathLike[str] | os.PathLike[bytes]

__all__ = ["Readable", "SourceLike", "StrOrBytesPath"]
