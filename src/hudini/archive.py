"""Reads one directory tree of logs for the web viewer.

:class:`Archive` keeps an inventory of the ``.hudini.jsonl.gz`` logs under
a directory. It answers what the index page asks: which videos exist,
their summaries, and which entries match a query. A scan reads only the
first and last lines of each file. This module does not import torch or
paddleocr.
"""

import gzip
import os
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import msgspec
from rapidfuzz import fuzz

from hudini.catalog import Catalog
from hudini.corrections import correct, rules_from_settings
from hudini.schema import (
    APPENDIX_MARK,
    FOOTER_MARK,
    Engine,
    Footer,
    Header,
    decode_appendix,
    decode_footer,
    decode_header,
)
from hudini.storage import load, tail_lengths
from hudini.views import Interval, Summary, interval_view, intervals_from_appendix

LOG_SUFFIX = ".hudini.jsonl.gz"
PARTIAL_LOG_SUFFIX = ".hudini.jsonl.partial"
VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov", ".avi")
FUZZY_MATCH_THRESHOLD = 80.0
TAIL_WINDOW_BYTES = 2**20
HEAD_PROBE_BYTES = 2**16
SCAN_WORKERS = 8


class EntryState(StrEnum):
    """What the archive knows about one log."""

    COMPLETE = "complete"
    RUNNING = "running"
    OUTDATED = "outdated"


class SearchField(StrEnum):
    """The corpus fields a search token can be narrowed to.

    ``TEXT`` is the union of the popup and banner fields, for queries
    that do not care which kind of message carried a string.
    """

    NAME = "name"
    INSTRUMENT = "instrument"
    ACTION = "action"
    POPUP = "popup"
    BANNER = "banner"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    """One log in the archive, as the index page shows it.

    Attributes:
        name: the log's path inside the archive without the log suffix,
            in POSIX form. This is the archive-wide identity.
        state: COMPLETE for a sealed log with a current summary,
            RUNNING for a partial one, OUTDATED for a sealed log whose
            summary version is not current.
        summary: the typed summary for a COMPLETE entry, else None.
        progress: parsed fraction 0-1 for a RUNNING entry with a known
            video duration, else None.
    """

    name: str
    path: Path
    state: EntryState
    header: Header
    footer: Footer | None = None
    summary: Summary | None = None
    progress: float | None = None


def log_locale(header: Header) -> str:
    """The catalog locale a log was parsed with. ``en`` when none is
    recorded."""
    catalog_engine = header.engines.get(Engine.CATALOG)
    if catalog_engine is None or catalog_engine.locale is None:
        return "en"
    return catalog_engine.locale


def _head_line(head: bytes) -> bytes | None:
    """The first decompressed line inside a gzip head probe, or None
    when the probe did not reach a newline."""
    decompressor = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
    text = decompressor.decompress(head)
    newline = text.find(b"\n")
    if newline < 0:
        return None
    return text[:newline]


def _edge_lines(path: Path) -> tuple[bytes, bytes, bytes]:
    """Line 1 and the last two non-empty lines, no body parsing.

    A ``.gz`` file with a tail pointer costs three small reads: the
    head probe, then one seek each for the footer and appendix
    members. A ``.gz`` file without one is one streaming decompression
    pass. A plain path, the ``.partial`` form, reads only the head and
    a tail window. A missing line is empty bytes."""
    if path.suffix == ".gz":
        with path.open("rb") as file:
            head = file.read(HEAD_PROBE_BYTES)
            lengths = tail_lengths(head)
            first = _head_line(head) if lengths is not None else None
            if lengths is not None and first is not None:
                appendix_length, footer_length = lengths
                size = file.seek(0, os.SEEK_END)
                file.seek(size - footer_length)
                last = gzip.decompress(file.read(footer_length)).strip()
                second_last = b""
                if appendix_length:
                    file.seek(size - footer_length - appendix_length)
                    second_last = gzip.decompress(file.read(appendix_length)).strip()
                return first, second_last, last
        first = b""
        second_last = b""
        last = b""
        with gzip.open(path, "rb") as file:
            for line in file:
                if not line.strip():
                    continue
                if not first:
                    first = line
                second_last = last
                last = line
        return first, second_last, last
    with path.open("rb") as file:
        first = file.readline()
        file.seek(0, os.SEEK_END)
        size = file.tell()
        window = min(size, TAIL_WINDOW_BYTES)
        file.seek(size - window)
        chunk = file.read(window)
    lines = [line for line in chunk.splitlines() if line.strip()]
    if window < size and lines:
        lines = lines[1:]
    last = lines[-1] if lines else b""
    second_last = lines[-2] if len(lines) >= 2 else b""
    if not first.strip():
        first = b""
    return first, second_last, last


def _maybe_footer(line: bytes) -> Footer | None:
    entry = msgspec.json.decode(line)
    if isinstance(entry, dict) and entry.get("hudini") == FOOTER_MARK:
        return decode_footer(line)
    return None


def _cached_intervals(second_last: bytes) -> list[Interval] | None:
    """The appendix intervals from the second-to-last line, or None
    when the line is not an appendix or its version is stale."""
    if not second_last:
        return None
    entry = msgspec.json.decode(second_last)
    if not (isinstance(entry, dict) and entry.get("hudini") == APPENDIX_MARK):
        return None
    return intervals_from_appendix(decode_appendix(second_last))


def _maybe_time_s(line: bytes) -> float | None:
    """The ``t`` of an observation line, or None when the line is not
    one. The live tail of a partial file can end mid-write."""
    try:
        entry = msgspec.json.decode(line)
    except msgspec.DecodeError:
        return None
    if isinstance(entry, dict) and isinstance(entry.get("t"), (int, float)):
        return float(entry["t"])
    return None


def tokenize(query: str) -> list[tuple[SearchField | None, str]]:
    """Query tokens as (field, text). A quoted phrase is one token, and
    a ``field:`` prefix narrows a token to one corpus field."""
    tokens: list[tuple[SearchField | None, str]] = []
    fields = {member.value: member for member in SearchField}
    rest = query.strip()
    while rest:
        field = None
        prefix, colon, after = rest.partition(":")
        if colon and prefix.casefold() in fields and " " not in prefix:
            field = fields[prefix.casefold()]
            rest = after.lstrip()
        if rest.startswith('"'):
            text, _quote, rest = rest[1:].partition('"')
        else:
            text, _space, rest = rest.partition(" ")
        rest = rest.lstrip()
        if text:
            tokens.append((field, text))
    return tokens


class Archive:
    """A reader over one directory tree of logs, with an inventory.

    Discovery is explicit: :meth:`scan` walks the tree and rebuilds
    the inventory, and :meth:`entries` and :meth:`search` read the
    inventory without touching the filesystem. Per-file reads are
    memoized across scans, keyed by name, mtime, and size. A changed
    file is read again, an unchanged one is not.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self._memo: dict[str, tuple[tuple[float, int], ArchiveEntry]] = {}
        self._inventory: list[ArchiveEntry] = []
        self._scan_lock = threading.Lock()
        self._intervals: dict[str, tuple[tuple[float, int], list[Interval]]] = {}
        self._catalogs: dict[str, Catalog] = {}

    def scan(self) -> list[ArchiveEntry]:
        """Walk the tree and rebuild the inventory.

        Finds every log under the directory, nested included, and reads
        the edges of new and changed files on a thread pool. One scan
        runs at a time: a call that arrives while another scan runs
        waits for that scan and returns its inventory.
        """
        if not self._scan_lock.acquire(blocking=False):
            with self._scan_lock:
                return self._inventory
        try:
            self._inventory = self._walk()
        finally:
            self._scan_lock.release()
        return self._inventory

    def _walk(self) -> list[ArchiveEntry]:
        found: list[ArchiveEntry] = []
        unread: list[tuple[Path, str, tuple[float, int]]] = []
        for parent, _subdirectories, files in os.walk(self.directory):
            for file_name in files:
                if not file_name.endswith((LOG_SUFFIX, PARTIAL_LOG_SUFFIX)):
                    continue
                path = Path(parent) / file_name
                relative = path.relative_to(self.directory).as_posix()
                stat = path.stat()
                key = (stat.st_mtime, stat.st_size)
                memoized = self._memo.get(relative)
                if memoized is not None and memoized[0] == key:
                    found.append(memoized[1])
                    continue
                unread.append((path, relative, key))
        if unread:
            with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as pool:
                read = pool.map(lambda task: self._read_entry(task[0], task[1]), unread)
                for (_path, relative, key), entry in zip(unread, read, strict=True):
                    self._memo[relative] = (key, entry)
                    found.append(entry)
        found.sort(key=lambda entry: entry.name)
        return found

    def entries(self) -> list[ArchiveEntry]:
        """The inventory of the last scan, sorted by name.

        Empty before the first :meth:`scan`. Reads no file.
        """
        return self._inventory

    def search(self, query: str) -> list[ArchiveEntry]:
        """The inventory entries whose corpus matches every token of
        ``query``.

        A token matches by substring first, then by fuzzy ratio. Entries
        without a summary match on the file name only. An empty query
        matches everything. Reads no file.
        """
        tokens = tokenize(query)
        return [
            entry
            for entry in self._inventory
            if all(self._matches(entry, field, text) for field, text in tokens)
        ]

    def intervals(self, name: str) -> list[Interval]:
        """One video's intervals, with the log's own corrections applied.

        A log with a current appendix answers from its cached
        intervals, one tail read, no derivation. A log without one, or
        with a stale ``intervals_version``, derives live from the
        body. Results are memoized by name, mtime, and size.

        Raises:
            FileNotFoundError: no finished log named ``name``.
        """
        path = self.directory / f"{name}{LOG_SUFFIX}"
        stat = path.stat()
        key = (stat.st_mtime, stat.st_size)
        memoized = self._intervals.get(name)
        if memoized is not None and memoized[0] == key:
            return memoized[1]
        _first, second_last, _last = _edge_lines(path)
        intervals = _cached_intervals(second_last)
        if intervals is None:
            log = load(path)
            catalog = self._catalog(log_locale(log.header))
            patches = correct(log, rules_from_settings(log.header.corrections), catalog)
            intervals = interval_view(log, patches, catalog)
        self._intervals[name] = (key, intervals)
        return intervals

    def video_path(self, name: str) -> Path | None:
        """The video file for ``name``, next to its log, or None.

        Looks for the filename the log's header records, then for the
        log's own basename with a common video suffix.
        """
        memoized = self._memo.get(f"{name}{LOG_SUFFIX}") or self._memo.get(
            f"{name}{PARTIAL_LOG_SUFFIX}"
        )
        candidates = []
        if memoized is not None:
            candidates.append((self.directory / name).parent / memoized[1].header.video.filename)
        candidates.extend(self.directory / f"{name}{suffix}" for suffix in VIDEO_SUFFIXES)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _catalog(self, locale: str) -> Catalog:
        if locale not in self._catalogs:
            self._catalogs[locale] = Catalog.load(locale=locale)
        return self._catalogs[locale]

    def _read_entry(self, path: Path, relative: str) -> ArchiveEntry:
        first, _second_last, last = _edge_lines(path)
        header = decode_header(first)
        name = relative.removesuffix(LOG_SUFFIX).removesuffix(PARTIAL_LOG_SUFFIX)
        footer = None if path.name.endswith(PARTIAL_LOG_SUFFIX) else _maybe_footer(last)
        if footer is None:
            return ArchiveEntry(
                name=name,
                path=path,
                state=EntryState.RUNNING,
                header=header,
                progress=self._progress(header, last),
            )
        summary = Summary.from_footer(footer)
        if summary is None:
            return ArchiveEntry(
                name=name, path=path, state=EntryState.OUTDATED, header=header, footer=footer
            )
        return ArchiveEntry(
            name=name,
            path=path,
            state=EntryState.COMPLETE,
            header=header,
            footer=footer,
            summary=summary,
        )

    def _progress(self, header: Header, last: bytes) -> float | None:
        duration = header.video.duration_s
        time_s = _maybe_time_s(last)
        if duration is None or duration <= 0 or time_s is None:
            return None
        return min(1.0, round(time_s / duration, 4))

    def _matches(self, entry: ArchiveEntry, field: SearchField | None, text: str) -> bool:
        corpus = self._corpus(entry)
        if field is None:
            values = [value for values in corpus.values() for value in values]
        elif field is SearchField.TEXT:
            values = [*corpus[SearchField.POPUP], *corpus[SearchField.BANNER]]
        else:
            values = corpus[field]
        needle = text.casefold()
        if any(needle in value.casefold() for value in values):
            return True
        return any(
            fuzz.partial_ratio(needle, value.casefold()) >= FUZZY_MATCH_THRESHOLD
            for value in values
        )

    def _corpus(self, entry: ArchiveEntry) -> dict[SearchField, list[str]]:
        corpus: dict[SearchField, list[str]] = {
            SearchField.NAME: [entry.name],
            SearchField.INSTRUMENT: [],
            SearchField.ACTION: [],
            SearchField.POPUP: [],
            SearchField.BANNER: [],
        }
        if entry.summary is None:
            return corpus
        for use in entry.summary.instruments:
            corpus[SearchField.INSTRUMENT].extend((use.name, use.type))
        corpus[SearchField.ACTION].extend(entry.summary.presses_by_action)
        corpus[SearchField.POPUP].extend(entry.summary.popup_texts)
        corpus[SearchField.BANNER].extend(entry.summary.banner_texts)
        return corpus
