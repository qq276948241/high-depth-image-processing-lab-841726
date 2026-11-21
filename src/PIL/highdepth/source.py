#
# The Python Imaging Library.
#
# Uniform byte sources: bytes, paths, file objects and chunk iterators.
#
# See the README file for information on usage and redistribution.
#
"""Byte sources used by every decoder.

All decoders read through a :class:`Source`, which offers the same random
access API whether the caller passed ``bytes``, a path, an open file object or
an iterator of chunks.  That is what makes the pixels, orientation,
transparency and metadata identical for the three input styles: the parsing
code never sees the difference.
"""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Iterable
from typing import TYPE_CHECKING, Self, cast

from .errors import (
    SourceError,
    TruncatedDataError,
)

if TYPE_CHECKING:
    from types import TracebackType
    from typing import IO

    from .budget import CancellationToken, MemoryBudget


class Source:
    """Random access byte source with an explicit, finite size."""

    name: str = "<source>"

    def __init__(self, size: int) -> None:
        self.size = size
        self._closed = False

    # -- required overrides ---------------------------------------------

    def read_at(self, offset: int, length: int) -> bytes:
        raise NotImplementedError

    def _close(self) -> None:
        pass

    # -- shared helpers --------------------------------------------------

    def read_exact(self, offset: int, length: int, what: str = "data") -> bytes:
        """Read exactly ``length`` bytes or raise :class:`TruncatedDataError`."""
        if offset < 0:
            msg = f"negative offset {offset}"
            raise SourceError(msg)
        if length < 0:
            msg = f"negative length {length}"
            raise SourceError(msg)
        data = self.read_at(offset, length)
        if len(data) != length:
            msg = (
                f"truncated {what}: wanted {length} bytes at offset {offset}, "
                f"got {len(data)} of {self.size}"
            )
            raise TruncatedDataError(msg)
        return data

    def bytes(self) -> bytes:
        return self.read_exact(0, self.size)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._close()

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class BytesSource(Source):
    """A source backed by an in-memory bytes-like object."""

    name = "<bytes>"

    def __init__(self, data: bytes | bytearray | memoryview) -> None:
        view = memoryview(data)
        if view.ndim != 1 or view.itemsize != 1:
            view = view.cast("B")
        self._view = view
        super().__init__(len(view))

    def read_at(self, offset: int, length: int) -> bytes:
        if offset >= self.size or length == 0:
            return b""
        end = min(offset + length, self.size)
        return bytes(self._view[offset:end])


class _RandomAccessFile(Source):
    """Common implementation for file-like objects that can seek."""

    def __init__(self, fp: IO[bytes], *, owns_fp: bool, name: str) -> None:
        self._fp = fp
        self._owns_fp = owns_fp
        self.name = name
        try:
            fp.seek(0, io.SEEK_END)
            size = fp.tell()
            fp.seek(0)
        except (OSError, ValueError) as exc:
            msg = f"source {name} is not seekable: {exc}"
            raise SourceError(msg) from exc
        super().__init__(size)

    def read_at(self, offset: int, length: int) -> bytes:
        if self._closed:
            msg = "read on a closed source"
            raise SourceError(msg)
        if offset >= self.size or length <= 0:
            return b""
        try:
            self._fp.seek(offset)
            return self._fp.read(min(length, self.size - offset))
        except (OSError, ValueError) as exc:
            msg = f"read from {self.name} failed: {exc}"
            raise SourceError(msg) from exc

    def _close(self) -> None:
        if self._owns_fp:
            try:
                self._fp.close()
            except OSError:
                pass


class FileSource(_RandomAccessFile):
    """A source backed by a file object or path."""

    def __init__(
        self, fp: IO[bytes], *, owns_fp: bool = False, name: str = "<file>"
    ) -> None:
        super().__init__(fp, owns_fp=owns_fp, name=name)


class SpoolingSource(Source):
    """A source that spools an unsized or non-seekable stream to disk.

    Chunks are validated to be bytes-like and counted as they arrive.  If the
    spool exceeds the memory budget the already written prefix is kept on disk
    and :class:`BudgetExceededError` is raised before more is read, so a
    hostile chunk iterator cannot exhaust memory.  The temporary file is
    removed when the source is closed, including after a failure.
    """

    def __init__(
        self,
        chunks: Iterable[bytes | bytearray | memoryview],
        *,
        budget: MemoryBudget | None = None,
        token: CancellationToken | None = None,
        temp_dir: str | None = None,
        name: str = "<chunks>",
    ) -> None:
        self.name = name
        self._file = tempfile.TemporaryFile(prefix="highdepth-", dir=temp_dir)
        reserved = 0
        total = 0
        try:
            for chunk in chunks:
                if token is not None:
                    token.raise_if_cancelled()
                chunk_object: object = chunk
                if not isinstance(chunk_object, (bytes, bytearray, memoryview)):
                    msg = (
                        "chunk source yielded "
                        f"{type(chunk_object).__name__}, expected a bytes-like object"
                    )
                    raise SourceError(msg)
                data = bytes(chunk_object)
                total += len(data)
                if budget is not None:
                    extra = total - reserved
                    if extra > 0:
                        budget.reserve(extra, "spooled source")
                        reserved = total
                self._file.write(data)
        except BaseException:
            self._file.close()
            if budget is not None:
                budget.release(reserved)
            raise
        self._reserved = reserved
        self._budget = budget
        self._file.flush()
        super().__init__(total)

    def read_at(self, offset: int, length: int) -> bytes:
        if self._closed:
            msg = "read on a closed source"
            raise SourceError(msg)
        if offset >= self.size or length <= 0:
            return b""
        self._file.seek(offset)
        return self._file.read(min(length, self.size - offset))

    def _close(self) -> None:
        try:
            self._file.close()
        finally:
            if self._budget is not None:
                self._budget.release(self._reserved)
                self._reserved = 0


def _as_path(obj: object) -> str | None:
    """Return a filesystem path for ``obj``, or ``None`` if it is not one.

    ``bytes`` is deliberately excluded: at this layer a bytes object is image
    data, never a filename.
    """
    if isinstance(obj, str):
        return obj
    if isinstance(obj, os.PathLike):
        value = os.fspath(obj)
        return value if isinstance(value, str) else None
    return None


def open_source(
    source: object,
    *,
    budget: MemoryBudget | None = None,
    token: CancellationToken | None = None,
    temp_dir: str | None = None,
) -> Source:
    """Normalize any supported input into a :class:`Source`.

    Accepted inputs:

    * ``bytes``/``bytearray``/``memoryview`` — used in place, never copied;
    * ``str``/``os.PathLike`` — opened read-only, owned by the returned source;
    * an object with ``read`` and ``seek`` — used in place, not owned;
    * an object with ``read`` but no ``seek`` — spooled through
      :class:`SpoolingSource` under the budget;
    * any other iterable of bytes-like chunks — spooled the same way.
    """
    if isinstance(source, Source):
        return source
    if isinstance(source, (bytes, bytearray, memoryview)):
        return BytesSource(source)
    path = _as_path(source)
    if path is not None:
        try:
            fp = open(path, "rb")
        except OSError as exc:
            msg = f"cannot open source {source!r}: {exc}"
            raise SourceError(msg) from exc
        return FileSource(fp, owns_fp=True, name=path)

    read = getattr(source, "read", None)
    seek = getattr(source, "seek", None)
    if callable(read) and callable(seek):
        return FileSource(
            cast("IO[bytes]", source),
            owns_fp=False,
            name=getattr(source, "name", "<file>"),
        )
    if callable(read):

        def _chunks() -> Iterable[bytes]:
            while True:
                if token is not None:
                    token.raise_if_cancelled()
                chunk = read(65536)
                if not chunk:
                    return
                yield chunk

        return SpoolingSource(
            _chunks(), budget=budget, token=token, temp_dir=temp_dir, name="<stream>"
        )

    if isinstance(source, Iterable):
        return SpoolingSource(
            cast("Iterable[bytes]", source),
            budget=budget,
            token=token,
            temp_dir=temp_dir,
            name="<chunks>",
        )

    msg = f"unsupported source type {type(source).__name__}"
    raise SourceError(msg)


__all__ = [
    "BytesSource",
    "FileSource",
    "Source",
    "SpoolingSource",
    "open_source",
]
