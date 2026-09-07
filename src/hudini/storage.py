"""Reads and writes the log file and its exports.

:func:`load` reads a finished ``.hudini.jsonl.gz`` log or a plain
``.partial`` one. :class:`LogWriter` appends to the ``.partial`` file
while a parse is live, flushed per line, so ``tail -f`` works. On a
clean close it writes the footer, packs the file, and renames it to the
final name. A file with the final name is always a finished parse.
:func:`write_frames` writes the per-frame export.

The finished file is one gzip stream of four members: the header line,
the body, the appendix line, and the footer line. The first member
states the compressed lengths of the last two, so a reader reaches the
tail with two seeks (:func:`tail_lengths`). ``zcat`` and every plain
gzip reader see the same JSONL document. This module does not import
torch or paddleocr.
"""

import gzip
import hashlib
import os
import time
import zlib
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Self

import msgspec

from hudini.schema import (
    APPENDIX_MARK,
    FOOTER_MARK,
    Appendix,
    CorrectionSetting,
    Footer,
    Header,
    Log,
    Observation,
    Signal,
    VideoInfo,
    decode_appendix,
    decode_footer,
    decode_header,
    decode_observation,
    encode,
)

FINAL_SUFFIX = ".gz"
PARTIAL_SUFFIX = ".partial"
HASH_CHUNK_BYTES = 2**20
GZIP_MAGIC = b"\x1f\x8b"
FEXTRA_FLAG = 4
TAIL_SUBFIELD = b"HU"
TAIL_SUBFIELD_BYTES = 8


def _member(payload: bytes, extra: bytes = b"") -> bytes:
    """One deterministic gzip member holding ``payload``.

    ``extra`` becomes the member's FEXTRA field. The timestamp is zero,
    so the same payload always gives the same bytes.
    """
    if not extra:
        return gzip.compress(payload, mtime=0)
    header = (
        GZIP_MAGIC
        + bytes([8, FEXTRA_FLAG])
        + (0).to_bytes(4, "little")
        + bytes([0, 255])
        + len(extra).to_bytes(2, "little")
        + extra
    )
    deflater = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    compressed = deflater.compress(payload) + deflater.flush()
    trailer = zlib.crc32(payload).to_bytes(4, "little") + (len(payload) % 2**32).to_bytes(
        4, "little"
    )
    return header + compressed + trailer


def pack_log(
    header_line: bytes, body: bytes, appendix_line: bytes | None, footer_line: bytes
) -> bytes:
    """The finished log file's bytes.

    Gzip members in order: the header line, the body, the appendix
    line when given, the footer line. The header member's extra field
    carries the compressed lengths of the appendix and footer members
    for :func:`tail_lengths`. Deterministic: the same input gives the
    same bytes.

    Args:
        body: the body lines as one newline-terminated block, empty for
            a log without observations.
    """
    appendix_member = b"" if appendix_line is None else _member(appendix_line + b"\n")
    footer_member = _member(footer_line + b"\n")
    extra = (
        TAIL_SUBFIELD
        + TAIL_SUBFIELD_BYTES.to_bytes(2, "little")
        + len(appendix_member).to_bytes(4, "little")
        + len(footer_member).to_bytes(4, "little")
    )
    return (
        _member(header_line + b"\n", extra=extra) + _member(body) + appendix_member + footer_member
    )


def tail_lengths(head: bytes) -> tuple[int, int] | None:
    """The compressed lengths of the appendix and footer members.

    Reads the pointer :func:`pack_log` stores in the first member's
    extra field. ``head`` must hold at least the start of the file.
    Returns (appendix_length, footer_length) in bytes. The appendix
    length is 0 for a log without one. Returns None when the file
    carries no pointer, so the caller must stream the whole file.
    """
    if len(head) < 12 or head[:2] != GZIP_MAGIC or not head[3] & FEXTRA_FLAG:
        return None
    extra_length = int.from_bytes(head[10:12], "little")
    extra = head[12 : 12 + extra_length]
    while len(extra) >= 4:
        subfield_length = int.from_bytes(extra[2:4], "little")
        if extra[:2] == TAIL_SUBFIELD and subfield_length == TAIL_SUBFIELD_BYTES:
            payload = extra[4 : 4 + TAIL_SUBFIELD_BYTES]
            return (
                int.from_bytes(payload[:4], "little"),
                int.from_bytes(payload[4:8], "little"),
            )
        extra = extra[4 + subfield_length :]
    return None


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file's content, as the header's provenance
    fields record it. Reads in chunks, so any file size is fine."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frames(
    path: Path,
    records: Iterable[dict],
    video: VideoInfo,
    corrections: Iterable[dict[str, CorrectionSetting]],
) -> None:
    """Write the frames export: gzip JSONL, one frame record per line.

    Line 1 is ``{"meta": ...}``, so the file describes itself. The meta
    block holds the video identity, the version of the package that
    shaped the records, and the correction settings that were applied.
    Every other line is one ``frame_view`` record. The export is derived
    data. A rebuild overwrites it.
    """
    meta = {
        "video": {"filename": video.filename, "sha256": video.sha256},
        "version": metadata.version("hudini"),
        "corrections": list(corrections),
    }
    encoder = msgspec.json.Encoder()
    with gzip.open(path, "wb") as file:
        file.write(encoder.encode({"meta": meta}) + b"\n")
        for record in records:
            file.write(encoder.encode(record) + b"\n")


def load(path: Path) -> Log:
    """Read one log file, finished or partial.

    A ``.gz`` path is read as gzip. Any other path, the ``.partial``
    form, is read as plain text. ``footer`` is None when the file ends
    without one: a partial parse.

    Raises:
        ValueError: the file is empty, the first line is not a header,
            or a body line is not an observation.
    """
    if path.suffix == FINAL_SUFFIX:
        with gzip.open(path, "rb") as file:
            lines = file.read().splitlines()
    else:
        lines = path.read_bytes().splitlines()
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    header = decode_header(lines[0])
    body = lines[1:]
    footer = None
    appendix = None
    if body and _line_mark(body[-1]) == FOOTER_MARK:
        footer = decode_footer(body[-1])
        body = body[:-1]
    if body and _line_mark(body[-1]) == APPENDIX_MARK:
        appendix = decode_appendix(body[-1])
        body = body[:-1]
    observations = tuple(decode_observation(line) for line in body)
    return Log(header=header, observations=observations, appendix=appendix, footer=footer)


def _line_mark(line: bytes) -> str | None:
    entry = msgspec.json.decode(line)
    if isinstance(entry, dict) and isinstance(entry.get("hudini"), str):
        return entry["hudini"]
    return None


class LogWriter:
    """One run's appender: header, then observations, finalized on close.

    Use as a context manager. Entering opens ``<final>.partial`` next to
    the final path; a clean exit writes the footer computed from what was
    written, compresses to the final ``.gz`` path, and removes the
    partial file. An exceptional exit leaves the partial file in place
    and writes nothing else.

    Attributes:
        appendix: the attached appendix, set by ``set_intervals``.
        footer: the computed footer, set at finalize; None while writing.

    Args:
        path: the final ``....hudini.jsonl.gz`` path.

    Raises:
        ValueError: ``path`` does not end in ``.gz``.
    """

    def __init__(self, path: Path) -> None:
        if path.suffix != FINAL_SUFFIX:
            raise ValueError(f"a log path must end in {FINAL_SUFFIX}: {path}")
        self.path = path
        self.partial_path = path.with_name(path.name.removesuffix(FINAL_SUFFIX) + PARTIAL_SUFFIX)
        self.appendix: Appendix | None = None
        self.footer: Footer | None = None
        self._file: BinaryIO | None = None
        self._started_at = 0.0
        self._observations = 0
        self._layout_times: list[float] = []
        self._wrote_header = False
        self._summary: tuple[int, dict] | None = None

    def __enter__(self) -> Self:
        self._file = self.partial_path.open("wb")
        self._started_at = time.monotonic()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._file is None:
            return
        if exc_type is not None:
            self._file.close()
            return
        self._finalize()

    def write_header(self, header: Header) -> None:
        """Write line 1. Must come before any observation."""
        self._append(encode(header))
        self._wrote_header = True

    def write(self, obs: Observation) -> None:
        """Append one observation line.

        Raises:
            ValueError: no header has been written yet.
        """
        if not self._wrote_header:
            raise ValueError("write the header before any observation")
        self._append(encode(obs))
        self._observations += 1
        if obs.signal is Signal.LAYOUT:
            self._layout_times.append(obs.time_s)

    def _append(self, line: bytes) -> None:
        if self._file is None:
            raise ValueError("the writer is not open; use it as a context manager")
        self._file.write(line + b"\n")
        self._file.flush()

    def set_summary(self, version: int, summary: dict) -> None:
        """Attach the summary the footer embeds.

        Call before the writer closes. The writer stores the plain JSON
        object as given and never interprets it.
        """
        self._summary = (version, summary)

    def set_intervals(self, version: int, intervals: list[dict]) -> None:
        """Attach the intervals the appendix line carries.

        Call before the writer closes. The writer stores the plain JSON
        objects as given and never interprets them.
        """
        self.appendix = Appendix(intervals_version=version, intervals=intervals)

    def _footer(self) -> Footer:
        frames = len(self._layout_times)
        span = self._layout_times[-1] - self._layout_times[0] if frames >= 2 else 0.0
        achieved_fps = (frames - 1) / span if span > 0 else 0.0
        footer = Footer(
            observations=self._observations,
            frames_sampled=frames,
            achieved_fps=round(achieved_fps, 3),
            runtime_s=round(time.monotonic() - self._started_at, 3),
        )
        if self._summary is None:
            return footer
        version, summary = self._summary
        return msgspec.structs.replace(footer, summary_version=version, summary=summary)

    def _finalize(self) -> None:
        """Footer, close, pack the members to a sibling, rename over the
        final path.

        Raises:
            ValueError: nothing was written; an empty log cannot be finalized.
        """
        if not self._wrote_header:
            raise ValueError("cannot finalize a log without a header")
        self.footer = self._footer()
        if self._file is not None:
            self._file.close()
        content = self.partial_path.read_bytes()
        newline = content.index(b"\n")
        appendix_line = None if self.appendix is None else encode(self.appendix)
        staging = self.path.with_name(self.path.name + ".tmp")
        staging.write_bytes(
            pack_log(content[:newline], content[newline + 1 :], appendix_line, encode(self.footer))
        )
        os.replace(staging, self.path)
        self.partial_path.unlink()
