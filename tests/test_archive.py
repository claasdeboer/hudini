"""Unit tests for hudini.archive.

Real files under tmp_path — the directory is the unit here.
"""

import gzip
from collections.abc import Callable
from pathlib import Path

import pytest

from hudini.archive import (
    TAIL_WINDOW_BYTES,
    Archive,
    EntryState,
    SearchField,
    log_locale,
    tokenize,
)
from hudini.schema import (
    Header,
    Observation,
    PedalColor,
    PodStatus,
    Signal,
    Status,
    VideoInfo,
    encode,
)
from hudini.storage import LogWriter
from hudini.views import (
    INTERVALS_VERSION,
    SUMMARY_VERSION,
    InstrumentUse,
    RunTotals,
    Summary,
)


@pytest.fixture
def make_summary() -> Callable[..., Summary]:
    """Build a summary with overridable instruments, actions, and texts."""

    def _make(
        instruments: tuple[InstrumentUse, ...] = (),
        presses_by_action: dict[str, int] | None = None,
        popup_texts: tuple[str, ...] = (),
        banner_texts: tuple[str, ...] = (),
    ) -> Summary:
        return Summary(
            instruments=instruments,
            instrument_changes=0,
            presses={color: RunTotals(count=0, duration_s=0.0) for color in PedalColor},
            presses_by_action=presses_by_action or {},
            laser=RunTotals(count=0, duration_s=0.0),
            offscreen=(),
            tool_association=(),
            popup_count=len(popup_texts),
            popup_texts=popup_texts,
            banner_duration_s=0.0,
            banner_texts=banner_texts,
            warning_duration_s=0.0,
            no_ui_duration_s=0.0,
        )

    return _make


@pytest.fixture
def make_header() -> Callable[..., Header]:
    def _make(filename: str = "case.mp4", duration_s: float | None = 100.0) -> Header:
        return Header(
            log_version=1,
            version="1.0.0",
            created_at="2026-08-14T12:00:00+02:00",
            video=VideoInfo(
                filename=filename,
                width=1920,
                height=1080,
                codec="h264",
                duration_s=duration_s,
            ),
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
            time_s=time_s, frame=int(time_s), signal=signal, key=(2,), value=value, score=0.9
        )

    return _make


@pytest.fixture
def write_log(make_header, make_observation, make_summary) -> Callable[..., Path]:
    """Write one finished log into a directory, with a summary by default."""

    def _write(
        directory: Path,
        stem: str,
        summary: Summary | None = None,
        summary_version: int = SUMMARY_VERSION,
        with_summary: bool = True,
    ) -> Path:
        path = directory / f"{stem}.hudini.jsonl.gz"
        with LogWriter(path) as writer:
            writer.write_header(make_header(filename=f"{stem}.mp4"))
            writer.write(make_observation(time_s=0.0, signal=Signal.LAYOUT))
            writer.write(make_observation(time_s=1.0, signal=Signal.LAYOUT))
            if with_summary:
                writer.set_summary(summary_version, (summary or make_summary()).to_wire())
        return path

    return _write


@pytest.fixture
def write_partial(make_header, make_observation) -> Callable[..., Path]:
    """Write one .partial log: header plus observations, no footer."""

    def _write(directory: Path, stem: str, last_time_s: float = 50.0) -> Path:
        path = directory / f"{stem}.hudini.jsonl.partial"
        lines = [
            encode(make_header(filename=f"{stem}.mp4")),
            encode(make_observation(time_s=0.0, signal=Signal.LAYOUT)),
            encode(make_observation(time_s=last_time_s, signal=Signal.LAYOUT)),
        ]
        path.write_bytes(b"\n".join(lines) + b"\n")
        return path

    return _write


class TestEntries:
    def test_states_and_summaries_per_file_kind(
        self, tmp_path, write_log, write_partial, make_summary
    ):
        write_log(tmp_path, "b_complete", summary=make_summary())
        write_log(tmp_path, "c_outdated", summary_version=SUMMARY_VERSION + 1)
        write_partial(tmp_path, "a_running", last_time_s=50.0)
        entries = Archive(tmp_path).scan()
        assert [(entry.name, entry.state) for entry in entries] == [
            ("a_running", EntryState.RUNNING),
            ("b_complete", EntryState.COMPLETE),
            ("c_outdated", EntryState.OUTDATED),
        ]
        assert entries[0].summary is None
        assert entries[1].summary is not None
        assert entries[2].summary is None

    def test_a_partial_log_reports_progress_from_header_duration(self, tmp_path, write_partial):
        write_partial(tmp_path, "case", last_time_s=50.0)
        (entry,) = Archive(tmp_path).scan()
        assert entry.progress == 0.5

    def test_a_partial_larger_than_the_tail_window_still_reports_progress(
        self, tmp_path, make_header, make_observation
    ):
        path = tmp_path / "case.hudini.jsonl.partial"
        lines = [encode(make_header(filename="case.mp4", duration_s=50000.0))]
        lines.extend(
            encode(make_observation(time_s=float(t), signal=Signal.LAYOUT)) for t in range(25000)
        )
        path.write_bytes(b"\n".join(lines) + b"\n")
        assert path.stat().st_size > TAIL_WINDOW_BYTES
        (entry,) = Archive(tmp_path).scan()
        assert entry.progress == round(24999 / 50000, 4)

    def test_a_sealed_log_without_a_summary_is_outdated(self, tmp_path, write_log):
        write_log(tmp_path, "case", with_summary=False)
        (entry,) = Archive(tmp_path).scan()
        assert entry.state is EntryState.OUTDATED

    def test_non_log_files_are_ignored(self, tmp_path, write_log):
        write_log(tmp_path, "case")
        (tmp_path / "case.mp4").write_bytes(b"")
        (tmp_path / "notes.txt").write_text("not a log")
        assert [entry.name for entry in Archive(tmp_path).scan()] == ["case"]

    def test_a_nested_log_is_named_by_its_relative_path(self, tmp_path, write_log):
        nested = tmp_path / "2026-08" / "case_002"
        nested.mkdir(parents=True)
        write_log(nested, "part_002")
        write_log(tmp_path, "flat")
        assert [entry.name for entry in Archive(tmp_path).scan()] == [
            "2026-08/case_002/part_002",
            "flat",
        ]

    def test_an_unchanged_file_is_not_read_again(self, tmp_path, write_log):
        write_log(tmp_path, "case")
        archive = Archive(tmp_path)
        (first,) = archive.scan()
        (second,) = archive.scan()
        assert second is first

    def test_a_changed_file_is_read_again(self, tmp_path, write_log, make_summary):
        write_log(tmp_path, "case")
        archive = Archive(tmp_path)
        (before,) = archive.scan()
        summary = make_summary(popup_texts=("Move grip to match instrument",))
        write_log(tmp_path, "case", summary=summary)
        (after,) = archive.scan()
        assert before.summary.popup_texts == ()
        assert after.summary.popup_texts == ("Move grip to match instrument",)

    def test_entries_reads_the_inventory_without_a_scan(self, tmp_path, write_log):
        write_log(tmp_path, "case")
        archive = Archive(tmp_path)
        assert archive.entries() == []
        archive.scan()
        (entry,) = archive.entries()
        assert entry.name == "case"

    def test_a_legacy_single_stream_log_still_reads(self, tmp_path, write_log):
        path = write_log(tmp_path, "case")
        path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes())))
        archive = Archive(tmp_path)
        (entry,) = archive.scan()
        assert entry.state is EntryState.COMPLETE
        assert archive.intervals("case")

    def test_a_deleted_log_stays_listed_until_the_next_scan(self, tmp_path, write_log):
        path = write_log(tmp_path, "case")
        archive = Archive(tmp_path)
        archive.scan()
        path.unlink()
        assert len(archive.entries()) == 1
        assert archive.scan() == []


class TestSearch:
    @pytest.fixture
    def stocked(self, tmp_path, write_log, write_partial, make_summary) -> Archive:
        write_log(
            tmp_path,
            "case_002",
            summary=make_summary(
                instruments=(
                    InstrumentUse(
                        name="Fenestrated Bipolar Forceps",
                        type="grasper",
                        duration_s=10.0,
                        arms=(2,),
                    ),
                ),
                presses_by_action={"coag": 3},
                popup_texts=("Move grip to match instrument",),
            ),
        )
        write_log(
            tmp_path,
            "case_017",
            summary=make_summary(
                instruments=(
                    InstrumentUse(
                        name="Vessel Sealer Extend",
                        type="vessel_sealer",
                        duration_s=5.0,
                        arms=(4,),
                    ),
                ),
                presses_by_action={"seal": 8},
            ),
        )
        write_partial(tmp_path, "case_154")
        archive = Archive(tmp_path)
        archive.scan()
        return archive

    def test_an_empty_query_matches_everything(self, stocked):
        assert len(stocked.search("")) == 3

    def test_a_bare_token_searches_every_field(self, stocked):
        assert [entry.name for entry in stocked.search("bipolar")] == ["case_002"]

    def test_every_token_must_match(self, stocked):
        assert stocked.search("bipolar seal") == []

    def test_a_prefix_narrows_a_token_to_one_field(self, stocked):
        assert [entry.name for entry in stocked.search("action:seal")] == ["case_017"]
        assert stocked.search("instrument:coag") == []

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ('popup:"move grip"', ["case_002"]),
            ('text:"move grip"', ["case_002"]),
            ('banner:"move grip"', []),
        ],
    )
    def test_popup_and_banner_narrow_and_text_is_their_union(self, stocked, query, expected):
        assert [entry.name for entry in stocked.search(query)] == expected

    def test_a_quoted_phrase_is_one_token(self, stocked):
        assert [entry.name for entry in stocked.search('"move grip"')] == ["case_002"]

    def test_a_fuzzy_token_still_finds_its_entry(self, stocked):
        assert [entry.name for entry in stocked.search("fenestratd bipolr")] == ["case_002"]

    def test_a_running_entry_matches_on_the_name_only(self, stocked):
        assert [entry.name for entry in stocked.search("154")] == ["case_154"]
        assert all(entry.name != "case_154" for entry in stocked.search("bipolar"))


class TestTokenize:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("bipolar coag", [(None, "bipolar"), (None, "coag")]),
            ('"move grip" coag', [(None, "move grip"), (None, "coag")]),
            ("action:coag", [(SearchField.ACTION, "coag")]),
            ('text:"table motion"', [(SearchField.TEXT, "table motion")]),
            ("  ", []),
            ("unknown:token", [(None, "unknown:token")]),
        ],
    )
    def test_queries_tokenize(self, query, expected):
        assert tokenize(query) == expected


class TestIntervals:
    def test_intervals_come_from_the_views_with_header_rules(self, tmp_path, write_log):
        write_log(tmp_path, "case")
        intervals = Archive(tmp_path).intervals("case")
        assert intervals
        assert all(interval.end_s > interval.start_s for interval in intervals)

    def test_a_missing_log_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Archive(tmp_path).intervals("missing")

    def test_a_nested_name_reaches_its_log(self, tmp_path, write_log):
        nested = tmp_path / "sub"
        nested.mkdir()
        write_log(nested, "case")
        assert Archive(tmp_path).intervals("sub/case")

    def test_a_current_appendix_answers_without_derivation(
        self, tmp_path, make_header, make_observation
    ):
        cached = [
            {
                "lane": "banner",
                "arm": None,
                "column": None,
                "start_s": 1.0,
                "end_s": 2.0,
                "value": "from the appendix, not the body",
            }
        ]
        path = tmp_path / "case.hudini.jsonl.gz"
        with LogWriter(path) as writer:
            writer.write_header(make_header())
            writer.write(make_observation(time_s=0.0, signal=Signal.LAYOUT))
            writer.set_intervals(INTERVALS_VERSION, cached)
        (interval,) = Archive(tmp_path).intervals("case")
        assert interval.value == "from the appendix, not the body"

    def test_a_stale_appendix_version_falls_back_to_the_body(
        self, tmp_path, make_header, make_observation
    ):
        cached = [
            {
                "lane": "banner",
                "arm": None,
                "column": None,
                "start_s": 1.0,
                "end_s": 2.0,
                "value": "stale",
            }
        ]
        path = tmp_path / "case.hudini.jsonl.gz"
        with LogWriter(path) as writer:
            writer.write_header(make_header())
            writer.write(make_observation(time_s=0.0, signal=Signal.LAYOUT))
            writer.set_intervals(INTERVALS_VERSION + 1, cached)
        intervals = Archive(tmp_path).intervals("case")
        assert all(interval.value != "stale" for interval in intervals)

    def test_intervals_memoize_until_the_file_changes(self, tmp_path, write_log):
        write_log(tmp_path, "case")
        archive = Archive(tmp_path)
        first = archive.intervals("case")
        assert archive.intervals("case") is first
        write_log(tmp_path, "case")
        assert archive.intervals("case") is not first


class TestVideoPath:
    def test_the_header_filename_wins(self, tmp_path, write_log):
        write_log(tmp_path, "case")
        (tmp_path / "case.mp4").write_bytes(b"")
        archive = Archive(tmp_path)
        archive.scan()
        assert archive.video_path("case") == tmp_path / "case.mp4"

    def test_a_name_suffix_probe_works_without_a_scan(self, tmp_path, write_log):
        write_log(tmp_path, "case")
        (tmp_path / "case.mkv").write_bytes(b"")
        assert Archive(tmp_path).video_path("case") == tmp_path / "case.mkv"

    def test_no_video_gives_none(self, tmp_path, write_log):
        write_log(tmp_path, "case")
        assert Archive(tmp_path).video_path("case") is None

    def test_a_nested_video_sits_next_to_its_log(self, tmp_path, write_log):
        nested = tmp_path / "sub"
        nested.mkdir()
        write_log(nested, "case")
        (nested / "case.mp4").write_bytes(b"")
        archive = Archive(tmp_path)
        archive.scan()
        assert archive.video_path("sub/case") == nested / "case.mp4"


def test_log_locale_defaults_to_en(make_header):
    assert log_locale(make_header()) == "en"
