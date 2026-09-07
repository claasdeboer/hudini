"""Tests for hudini.web.server: a live server over a tmp-path archive."""

import json
import threading
from collections.abc import Callable, Iterator
from http.client import HTTPConnection
from pathlib import Path

import pytest

from hudini.schema import Header, Observation, PedalColor, PodStatus, Signal, Status, VideoInfo
from hudini.storage import LogWriter
from hudini.views import SUMMARY_VERSION, RunTotals, Summary
from hudini.web.server import ArchiveServer, make_server

VIDEO_BYTES = b"not really mp4, but 64 bytes of payload for range requests!!!!!!"


def _summary() -> Summary:
    return Summary(
        instruments=(),
        instrument_changes=0,
        presses={color: RunTotals(count=0, duration_s=0.0) for color in PedalColor},
        presses_by_action={"coag": 2},
        laser=RunTotals(count=0, duration_s=0.0),
        offscreen=(),
        tool_association=(),
        popup_count=0,
        popup_texts=(),
        banner_duration_s=0.0,
        banner_texts=(),
        warning_duration_s=0.0,
        no_ui_duration_s=0.0,
    )


def _write_log(directory: Path, stem: str) -> None:
    header = Header(
        log_version=1,
        version="1.0.0",
        created_at="2026-08-14T12:00:00+02:00",
        video=VideoInfo(
            filename=f"{stem}.mp4", width=1920, height=1080, codec="h264", duration_s=100.0
        ),
        requested=("status",),
        signals=(Signal.LAYOUT, Signal.STATUS),
        rates={Signal.LAYOUT: 1.0, Signal.STATUS: 1.0},
        batch_size=1,
        device="cpu",
    )
    with LogWriter(directory / f"{stem}.hudini.jsonl.gz") as writer:
        writer.write_header(header)
        writer.write(
            Observation(time_s=0.0, frame=0, signal=Signal.LAYOUT, key=(), value=None, score=1.0)
        )
        writer.write(
            Observation(
                time_s=0.0,
                frame=0,
                signal=Signal.STATUS,
                key=(2,),
                value=PodStatus(status=Status.ACTIVE),
                score=0.9,
            )
        )
        writer.write(
            Observation(time_s=1.0, frame=1, signal=Signal.LAYOUT, key=(), value=None, score=1.0)
        )
        writer.set_summary(SUMMARY_VERSION, _summary().to_wire())


@pytest.fixture
def server(tmp_path) -> Iterator[ArchiveServer]:
    _write_log(tmp_path, "case_002")
    (tmp_path / "case_002.mp4").write_bytes(VIDEO_BYTES)
    instance = make_server(tmp_path, port=0)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    thread.join(timeout=5)
    instance.server_close()


@pytest.fixture
def get(server) -> Callable[..., tuple[int, dict[str, str], bytes]]:
    """One GET against the live server: (status, headers, body)."""

    def _get(path: str, headers: dict[str, str] | None = None):
        _host, port = server.server_address[:2]
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        body = response.read()
        collected = {name.lower(): value for name, value in response.getheaders()}
        connection.close()
        return response.status, collected, body

    return _get


@pytest.fixture
def post(server) -> Callable[..., tuple[int, bytes]]:
    """One POST against the live server: (status, body)."""

    def _post(path: str):
        _host, port = server.server_address[:2]
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("POST", path)
        response = connection.getresponse()
        body = response.read()
        connection.close()
        return response.status, body

    return _post


class TestRefresh:
    def test_a_new_log_appears_only_after_a_refresh(self, get, post, tmp_path):
        _status, _headers, body = get("/api/videos")
        assert len(json.loads(body)["videos"]) == 1
        _write_log(tmp_path, "case_017")
        _status, _headers, body = get("/api/videos")
        assert len(json.loads(body)["videos"]) == 1
        status, body = post("/api/refresh")
        assert status == 200
        assert len(json.loads(body)["videos"]) == 2

    def test_an_unknown_post_route_is_404(self, post):
        status, _body = post("/api/anything")
        assert status == 404


class TestApiVideos:
    def test_rows_carry_the_index_columns(self, get, tmp_path):
        status, _headers, body = get("/api/videos")
        assert status == 200
        listing = json.loads(body)
        assert listing["directory"] == str(Path(tmp_path).resolve())
        (row,) = listing["videos"]
        assert row["name"] == "case_002"
        assert row["state"] == "complete"
        assert row["duration_s"] == 100.0
        assert row["achieved_fps"] == 1.0
        assert row["summary"]["presses_by_action"] == {"coag": 2}

    def test_a_query_filters(self, get):
        status, _headers, body = get("/api/videos?q=coag")
        assert status == 200
        assert len(json.loads(body)["videos"]) == 1
        status, _headers, body = get("/api/videos?q=nomatch_xyz_123")
        assert json.loads(body)["videos"] == []


class TestApiIntervals:
    def test_intervals_serialize_as_json(self, get):
        status, _headers, body = get("/api/intervals/case_002")
        assert status == 200
        intervals = json.loads(body)
        assert intervals
        assert {"lane", "arm", "column", "start_s", "end_s", "value"} <= set(intervals[0])

    def test_an_unknown_name_is_404(self, get):
        status, _headers, _body = get("/api/intervals/missing")
        assert status == 404

    def test_a_nested_name_reaches_its_log(self, get, tmp_path, server):
        nested = tmp_path / "sub"
        nested.mkdir()
        _write_log(nested, "case_017")
        for path in ("/api/intervals/sub/case_017", "/api/intervals/sub%2Fcase_017"):
            status, _headers, body = get(path)
            assert status == 200
            assert json.loads(body)

    @pytest.mark.parametrize(
        "path",
        [
            "/api/intervals/..%2Fcase_002",
            "/api/intervals/%2Fetc%2Fpasswd",
            "/video/..%2F..%2Fcase_002",
        ],
    )
    def test_a_traversal_name_is_404(self, get, path):
        status, _headers, _body = get(path)
        assert status == 404


class TestVideo:
    def test_a_full_request_streams_the_file(self, get):
        status, headers, body = get("/video/case_002")
        assert status == 200
        assert body == VIDEO_BYTES
        assert headers["accept-ranges"] == "bytes"

    def test_a_range_request_returns_the_slice(self, get):
        status, headers, body = get("/video/case_002", headers={"Range": "bytes=4-9"})
        assert status == 206
        assert body == VIDEO_BYTES[4:10]
        assert headers["content-range"] == f"bytes 4-9/{len(VIDEO_BYTES)}"

    def test_an_open_ended_range_reads_to_the_end(self, get):
        status, _headers, body = get("/video/case_002", headers={"Range": "bytes=60-"})
        assert status == 206
        assert body == VIDEO_BYTES[60:]

    def test_an_unsatisfiable_range_is_416(self, get):
        status, _headers, _body = get(
            "/video/case_002", headers={"Range": f"bytes={len(VIDEO_BYTES)}-"}
        )
        assert status == 416

    def test_a_missing_video_is_404(self, get):
        status, _headers, _body = get("/video/missing")
        assert status == 404


class TestHandleError:
    def test_a_client_hangup_is_quiet(self, server, capsys):
        try:
            raise ConnectionResetError(104, "Connection reset by peer")
        except ConnectionResetError:
            server.handle_error(None, ("127.0.0.1", 1234))
        assert capsys.readouterr().err == ""

    def test_other_errors_still_report(self, server, capsys):
        try:
            raise ValueError("a real bug")
        except ValueError:
            server.handle_error(None, ("127.0.0.1", 1234))
        assert "a real bug" in capsys.readouterr().err


class TestStatic:
    def test_a_static_asset_is_served(self, get):
        status, headers, _body = get("/static/shared.js")
        assert status == 200
        assert headers["content-type"].startswith("text/javascript")

    def test_the_root_serves_the_app_shell(self, get):
        status, headers, body = get("/")
        assert status == 200
        assert headers["content-type"].startswith("text/html")
        assert b"HUDINI_EMBEDDED" in body

    def test_a_traversal_attempt_is_404(self, get):
        status, _headers, _body = get("/static/%2e%2e%2fserver.py")
        assert status == 404

    def test_an_unknown_route_is_404(self, get):
        status, _headers, _body = get("/nope")
        assert status == 404
