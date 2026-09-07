"""Serves one archive directory over HTTP for ``hudini serve``.

Every route is one thin call into :class:`hudini.archive.Archive`. The
server binds to ``127.0.0.1`` unless told otherwise, because surgical
video must not become visible on a network by accident. Serialization
happens here, at the edge. This module does not import torch or
paddleocr.
"""

import logging
import socket
import sys
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, unquote, urlsplit

import msgspec

from hudini.archive import LOG_SUFFIX, Archive, ArchiveEntry

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8410
STREAM_CHUNK_BYTES = 2**20

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
}
DEFAULT_CONTENT_TYPE = "application/octet-stream"

_encoder = msgspec.json.Encoder()


def video_row(entry: ArchiveEntry) -> dict:
    """One ``/api/videos`` row: the header and footer facts the pages
    show, the entry state, and the summary when the entry has one."""
    row: dict = {
        "name": entry.name,
        "state": entry.state,
        "duration_s": entry.header.video.duration_s,
        "parsed_at": entry.header.created_at,
        "version": entry.header.version,
        "log_version": entry.header.log_version,
        "achieved_fps": entry.footer.achieved_fps if entry.footer is not None else None,
        "progress": entry.progress,
    }
    if entry.summary is not None:
        row["summary"] = entry.summary.to_wire()
    return row


def confined_name(name: str) -> bool:
    """Whether ``name`` stays inside the archive: a non-empty relative
    path with no ``..`` steps. Request names must pass this check
    before they touch the filesystem."""
    return bool(name) and not name.startswith("/") and ".." not in PurePosixPath(name).parts


class ArchiveServer(ThreadingHTTPServer):
    """The archive's HTTP server. One instance serves one directory.

    Construction starts the first archive scan on a background thread,
    so the server binds and serves the page shell immediately. The
    listing route waits for that scan; a rescan happens only on the
    refresh route.
    """

    def __init__(self, address: tuple[str, int], archive: Archive) -> None:
        super().__init__(address, ArchiveRequestHandler)
        self.archive = archive
        self.first_scan = threading.Thread(target=archive.scan, name="hudini-scan", daemon=True)
        self.first_scan.start()

    def handle_error(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: object
    ) -> None:
        """Keep client hang-ups quiet.

        A browser aborts video Range requests on every seek, so a reset
        or broken pipe mid-response is normal operation, not an error.
        Everything else still gets the standard traceback.
        """
        kind = sys.exc_info()[0]
        if kind is not None and issubclass(kind, (BrokenPipeError, ConnectionResetError)):
            logger.debug("client %s hung up mid-response", client_address)
            return
        super().handle_error(request, client_address)


class ArchiveRequestHandler(BaseHTTPRequestHandler):
    """Routes GET requests onto the archive, plus one POST: the rescan."""

    server: ArchiveServer

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("%s %s", self.address_string(), format % args)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/refresh":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        entries = self.server.archive.scan()
        self._listing(entries)

    def do_GET(self) -> None:
        parts = urlsplit(self.path)
        segments = [unquote(part) for part in parts.path.split("/") if part]
        if not segments:
            self._static("app.html")
        elif segments[0] == "static" and len(segments) == 2:
            self._static(segments[1])
        elif segments == ["api", "videos"]:
            query = parse_qs(parts.query).get("q", [""])[0]
            self._api_videos(query)
        elif segments[:2] == ["api", "intervals"] and len(segments) >= 3:
            self._api_intervals("/".join(segments[2:]))
        elif segments[0] == "video" and len(segments) >= 2:
            self._video("/".join(segments[1:]))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _api_videos(self, query: str) -> None:
        self.server.first_scan.join()
        archive = self.server.archive
        self._listing(archive.search(query) if query else archive.entries())

    def _listing(self, entries: list[ArchiveEntry]) -> None:
        self._json(
            {
                "directory": str(self.server.archive.directory),
                "videos": [video_row(entry) for entry in entries],
            }
        )

    def _api_intervals(self, name: str) -> None:
        archive = self.server.archive
        if not confined_name(name) or not (archive.directory / f"{name}{LOG_SUFFIX}").is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._json([asdict(interval) for interval in archive.intervals(name)])

    def _json(self, payload: object) -> None:
        body = _encoder.encode(payload)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name: str) -> None:
        path = (STATIC / name).resolve()
        if path.parent != STATIC.resolve() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix, DEFAULT_CONTENT_TYPE))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _video(self, name: str) -> None:
        archive = self.server.archive
        path = archive.video_path(name) if confined_name(name) else None
        if path is None or not path.resolve().is_relative_to(archive.directory):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        start, end = self._requested_range(size)
        if start is None or end is None:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        partial = self.headers.get("Range") is not None
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix, DEFAULT_CONTENT_TYPE))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        self._stream(path, start, end)

    def _requested_range(self, size: int) -> tuple[int | None, int | None]:
        """The byte range to send: the whole file without a Range header,
        the requested slice with one, (None, None) when unsatisfiable."""
        header = self.headers.get("Range")
        if header is None:
            return 0, size - 1
        unit, equals, spans = header.partition("=")
        if unit != "bytes" or not equals:
            return None, None
        first, dash, last = spans.split(",")[0].strip().partition("-")
        if not dash or not (first or last):
            return None, None
        if not first:
            start = max(0, size - int(last))
            return start, size - 1
        start = int(first)
        end = int(last) if last else size - 1
        if start >= size or start > end:
            return None, None
        return start, min(end, size - 1)

    def _stream(self, path: Path, start: int, end: int) -> None:
        remaining = end - start + 1
        with path.open("rb") as file:
            file.seek(start)
            while remaining > 0:
                chunk = file.read(min(STREAM_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def make_server(
    directory: Path | str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> ArchiveServer:
    """The archive server, bound but not yet serving.

    Port 0 binds an ephemeral port; ``server_address`` states the real
    one. The caller owns ``serve_forever`` and shutdown.
    """
    return ArchiveServer((host, port), Archive(directory))


def serve(directory: Path | str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Serve one archive directory until interrupted."""
    with make_server(directory, host=host, port=port) as server:
        bound_host, bound_port = server.server_address[:2]
        logger.info("serving %s at http://%s:%s/", directory, bound_host, bound_port)
        server.serve_forever()
