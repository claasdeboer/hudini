"""Unit tests for hudini.storage.

Real files under tmp_path — the file is the unit here.
"""

import gzip
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from hudini.schema import (
    Footer,
    Header,
    Observation,
    PedalColor,
    PodStatus,
    Role,
    Signal,
    Status,
    VideoInfo,
    encode,
)
from hudini.storage import LogWriter, load, pack_log, sha256_of, tail_lengths, write_frames


@pytest.fixture
def make_header() -> Callable[..., Header]:
    def _make() -> Header:
        return Header(
            log_version=1,
            version="1.0.0",
            created_at="2026-08-07T14:12:03+02:00",
            video=VideoInfo(filename="case_002.mp4", width=1920, height=1080, codec="h264"),
            requested=("status",),
            signals=(Signal.LAYOUT, Signal.STATUS),
            rates={Signal.LAYOUT: 1.0, Signal.STATUS: 1.0},
            batch_size=1,
            device="cpu",
        )

    return _make


@pytest.fixture
def make_observation() -> Callable[..., Observation]:
    def _make(time_s: float = 0.0, signal: Signal = Signal.STATUS) -> Observation:
        value = None if signal is Signal.LAYOUT else PodStatus(status=Status.ACTIVE)
        return Observation(
            time_s=time_s, frame=int(time_s * 30), signal=signal, key=(2,), value=value, score=0.9
        )

    return _make


@pytest.fixture
def written_log(tmp_path, make_header, make_observation) -> Path:
    """A finished log written through LogWriter: 2 layout ticks, 1 status."""
    path = tmp_path / "case.hudini.jsonl.gz"
    with LogWriter(path) as writer:
        writer.write_header(make_header())
        writer.write(make_observation(time_s=0.0, signal=Signal.LAYOUT))
        writer.write(make_observation(time_s=0.0))
        writer.write(make_observation(time_s=0.5, signal=Signal.LAYOUT))
    return path


class TestLoad:
    def test_a_finished_log_round_trips(self, written_log, make_header, make_observation):
        log = load(written_log)
        assert log.header == make_header()
        assert log.observations[1] == make_observation(time_s=0.0)
        assert log.footer is not None

    def test_a_partial_file_loads_with_no_footer(self, tmp_path, make_header, make_observation):
        path = tmp_path / "case.hudini.jsonl.partial"
        lines = [encode(make_header()), encode(make_observation())]
        path.write_bytes(b"\n".join(lines) + b"\n")
        log = load(path)
        assert len(log.observations) == 1
        assert log.footer is None

    def test_a_gz_file_without_footer_is_a_partial_parse(self, tmp_path, make_header):
        path = tmp_path / "case.hudini.jsonl.gz"
        with gzip.open(path, "wb") as file:
            file.write(encode(make_header()) + b"\n")
        assert load(path).footer is None

    def test_an_empty_file_raises(self, tmp_path):
        path = tmp_path / "case.hudini.jsonl.partial"
        path.write_bytes(b"")
        with pytest.raises(ValueError):
            load(path)

    def test_a_file_not_starting_with_a_header_raises(self, tmp_path, make_observation):
        path = tmp_path / "case.hudini.jsonl.partial"
        path.write_bytes(encode(make_observation()) + b"\n")
        with pytest.raises(ValueError):
            load(path)


class TestContainer:
    def test_the_finished_file_decompresses_to_the_plain_document(self, written_log, make_header):
        lines = gzip.decompress(written_log.read_bytes()).splitlines()
        assert lines[0] == encode(make_header())
        assert json.loads(lines[-1])["hudini"] == "footer"

    def test_pack_log_is_deterministic(self):
        packed = pack_log(b'{"h":1}', b"body\n", b'{"a":1}', b'{"f":1}')
        assert packed == pack_log(b'{"h":1}', b"body\n", b'{"a":1}', b'{"f":1}')

    def test_tail_lengths_locate_the_tail_members(self):
        packed = pack_log(b'{"h":1}', b"body\n", b'{"a":1}', b'{"f":1}')
        appendix_length, footer_length = tail_lengths(packed)
        assert gzip.decompress(packed[-footer_length:]).strip() == b'{"f":1}'
        start = len(packed) - footer_length - appendix_length
        assert gzip.decompress(packed[start : start + appendix_length]).strip() == b'{"a":1}'

    def test_a_log_without_an_appendix_has_zero_appendix_length(self):
        packed = pack_log(b'{"h":1}', b"", None, b'{"f":1}')
        appendix_length, _footer_length = tail_lengths(packed)
        assert appendix_length == 0

    def test_a_plain_gzip_file_has_no_tail_pointer(self):
        assert tail_lengths(gzip.compress(b"h\nbody\nf\n")) is None


class TestLogWriter:
    def test_the_footer_digest_matches_the_body(self, written_log):
        log = load(written_log)
        layout = [obs for obs in log.observations if obs.signal is Signal.LAYOUT]
        assert log.footer.observations == len(log.observations)
        assert log.footer.frames_sampled == len(layout)
        assert log.footer.achieved_fps == pytest.approx((len(layout) - 1) / 0.5)

    def test_the_partial_file_is_live_and_the_rename_commits(
        self, tmp_path, make_header, make_observation
    ):
        path = tmp_path / "case.hudini.jsonl.gz"
        with LogWriter(path) as writer:
            writer.write_header(make_header())
            writer.write(make_observation())
            assert not path.exists()
            live = load(writer.partial_path)
            assert len(live.observations) == 1
            assert live.footer is None
        assert path.exists()
        assert not writer.partial_path.exists()

    def test_an_exception_keeps_the_partial_and_writes_no_final(
        self, tmp_path, make_header, make_observation
    ):
        path = tmp_path / "case.hudini.jsonl.gz"
        with pytest.raises(RuntimeError), LogWriter(path) as writer:
            writer.write_header(make_header())
            writer.write(make_observation())
            raise RuntimeError("model died")
        assert writer.partial_path.exists()
        assert not path.exists()
        assert load(writer.partial_path).footer is None

    def test_a_single_layout_tick_gives_zero_achieved_fps(
        self, tmp_path, make_header, make_observation
    ):
        path = tmp_path / "case.hudini.jsonl.gz"
        with LogWriter(path) as writer:
            writer.write_header(make_header())
            writer.write(make_observation(signal=Signal.LAYOUT))
        footer = load(path).footer
        assert footer == Footer(
            observations=1,
            frames_sampled=1,
            achieved_fps=0.0,
            runtime_s=footer.runtime_s,
        )

    def test_a_set_summary_lands_in_the_footer_and_round_trips(
        self, tmp_path, make_header, make_observation
    ):
        path = tmp_path / "case.hudini.jsonl.gz"
        with LogWriter(path) as writer:
            writer.write_header(make_header())
            writer.write(make_observation(time_s=0.0))
            writer.set_summary(1, {"popup_count": 3})
        loaded = load(path)
        assert loaded.footer is not None
        assert loaded.footer.summary_version == 1
        assert loaded.footer.summary == {"popup_count": 3}

    def test_a_footer_without_a_summary_stays_summary_free(
        self, tmp_path, make_header, make_observation
    ):
        path = tmp_path / "case.hudini.jsonl.gz"
        with LogWriter(path) as writer:
            writer.write_header(make_header())
            writer.write(make_observation(time_s=0.0))
        loaded = load(path)
        assert loaded.footer is not None
        assert loaded.footer.summary_version is None
        assert loaded.footer.summary is None

    def test_set_intervals_lands_in_the_appendix_and_round_trips(
        self, tmp_path, make_header, make_observation
    ):
        path = tmp_path / "case.hudini.jsonl.gz"
        wire = [
            {
                "lane": "status",
                "arm": 2,
                "column": 2,
                "start_s": 0.0,
                "end_s": 4.0,
                "value": "active",
            }
        ]
        with LogWriter(path) as writer:
            writer.write_header(make_header())
            writer.write(make_observation(time_s=0.0))
            writer.set_intervals(1, wire)
        loaded = load(path)
        assert loaded.appendix is not None
        assert loaded.appendix.intervals_version == 1
        assert loaded.appendix.intervals == wire
        assert loaded.footer is not None
        assert len(loaded.observations) == 1

    def test_a_log_without_intervals_has_no_appendix(self, tmp_path, make_header, make_observation):
        path = tmp_path / "case.hudini.jsonl.gz"
        with LogWriter(path) as writer:
            writer.write_header(make_header())
            writer.write(make_observation(time_s=0.0))
        assert load(path).appendix is None

    def test_writing_before_the_header_raises(self, tmp_path, make_header, make_observation):
        with LogWriter(tmp_path / "case.hudini.jsonl.gz") as writer:
            with pytest.raises(ValueError):
                writer.write(make_observation())
            writer.write_header(make_header())

    def test_a_path_without_gz_suffix_raises(self, tmp_path):
        with pytest.raises(ValueError):
            LogWriter(tmp_path / "case.hudini.jsonl")

    def test_sealing_without_a_header_raises(self, tmp_path):
        with pytest.raises(ValueError), LogWriter(tmp_path / "case.hudini.jsonl.gz"):
            pass


def test_sha256_of_hashes_the_file_content(tmp_path):
    path = tmp_path / "weights.pt"
    path.write_bytes(b"weights")
    assert sha256_of(path) == hashlib.sha256(b"weights").hexdigest()


class TestWriteFrames:
    def test_the_export_self_describes_and_round_trips(self, tmp_path):
        records = [
            {
                "frame_index": 0,
                "time_s": 0.0,
                "pods": [
                    {
                        "column": 1,
                        "role": Role.CAMERA,
                        "status": {"value": Status.ACTIVE, "score": 0.9},
                        "pedals": {PedalColor.YELLOW: {"value": True, "action": None}},
                    }
                ],
            },
            {"frame_index": 30, "time_s": 1.0, "pods": []},
        ]
        path = tmp_path / "case.frames.jsonl.gz"
        write_frames(
            path,
            records,
            video=VideoInfo(filename="case.mp4", width=1920, height=1080, codec="h264"),
            corrections=[{"name": "debounce:status", "min_duration_s": 0.5}],
        )
        with gzip.open(path, "rt") as file:
            meta_line, *record_lines = [json.loads(line) for line in file]
        assert meta_line["meta"]["video"] == {"filename": "case.mp4", "sha256": None}
        assert meta_line["meta"]["corrections"] == [
            {"name": "debounce:status", "min_duration_s": 0.5}
        ]
        assert meta_line["meta"]["version"]
        assert record_lines[0]["pods"][0] == {
            "column": 1,
            "role": "camera",
            "status": {"value": "active", "score": 0.9},
            "pedals": {"yellow": {"value": True, "action": None}},
        }
        assert record_lines[1] == {"frame_index": 30, "time_s": 1.0, "pods": []}

    def test_an_export_without_records_still_carries_the_meta(self, tmp_path):
        path = tmp_path / "case.frames.jsonl.gz"
        write_frames(
            path,
            [],
            video=VideoInfo(filename="case.mp4", width=1920, height=1080, codec="h264"),
            corrections=[],
        )
        with gzip.open(path, "rt") as file:
            lines = file.readlines()
        assert len(lines) == 1
        assert "meta" in json.loads(lines[0])
