"""Unit tests for hudini.cli.

The parse handler's model layer is patched at its import sources; query
and catalog run against real files under tmp_path.
"""

import argparse
import gzip
import json
import warnings
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hudini.catalog import CatalogEntry, PedalAction, PedalPair
from hudini.cli import (
    GROUPS,
    configure_logging,
    main,
    model_argument,
    rate_argument,
    signal_selection,
    time_argument,
)
from hudini.privacy import PopupTemplate
from hudini.schema import (
    Box,
    Footer,
    FrameGeometry,
    Header,
    Log,
    Observation,
    PodGeometry,
    PodStatus,
    Role,
    Signal,
    Status,
    VideoInfo,
)
from hudini.storage import LogWriter


class TestArgumentTypes:
    def test_a_group_token_selects_the_study(self):
        assert signal_selection("pedals") == GROUPS["pedals"]

    def test_subtraction_removes_a_group(self):
        assert signal_selection("all,-detectors") == GROUPS["all"] - GROUPS["detectors"]

    def test_plain_names_select_signals(self):
        assert signal_selection("status,arm") == {Signal.STATUS, Signal.ARM}

    @pytest.mark.parametrize("text", ["bogus", "detectors,-detectors"])
    def test_bad_selections_raise(self, text):
        with pytest.raises(argparse.ArgumentTypeError):
            signal_selection(text)

    @pytest.mark.parametrize(
        ("text", "seconds"),
        [("12.5", 12.5), ("98:23", 5903.0), ("1:02:03.5", 3723.5)],
    )
    def test_time_arguments_accept_seconds_and_clock_forms(self, text, seconds):
        assert time_argument(text) == seconds

    def test_rate_and_model_arguments_split_on_equals(self):
        assert rate_argument("pedals=30") == (Signal.PEDALS, 30.0)
        name, path = model_argument("arm=weights.pt")
        assert (name, path.name) == ("arm", "weights.pt")

    @pytest.mark.parametrize("text", ["pedals", "bogus=1"])
    def test_a_malformed_rate_raises(self, text):
        with pytest.raises(argparse.ArgumentTypeError):
            rate_argument(text)


def make_parse_log() -> Log:
    header = Header(
        log_version=1,
        version="1.0.0",
        created_at="2026-08-10T12:00:00+02:00",
        video=VideoInfo(filename="case.mp4", width=1920, height=1080, codec="h264"),
        requested=("pedals",),
        signals=(Signal.PEDALS,),
        rates={Signal.PEDALS: 30.0},
        batch_size=1,
        device="cpu",
    )
    layout = Observation(
        time_s=0.0, frame=0, signal=Signal.LAYOUT, key=(), value=GEOMETRY, score=1.0
    )
    footer = Footer(observations=1, frames_sampled=1, achieved_fps=0.0, runtime_s=1.0)
    return Log(header=header, observations=(layout,), footer=footer)


def run_parse(tmp_path, extra: list[str]) -> tuple[int, MagicMock, MagicMock]:
    parser = MagicMock()
    parser.parse_video.return_value = make_parse_log()
    with (
        patch("hudini.parser.Parser", return_value=parser) as parser_class,
        patch(
            "hudini.video.probe_video",
            return_value=VideoInfo(
                filename="case.mp4", width=1920, height=1080, codec="h264", duration_s=10.0
            ),
        ),
    ):
        code = main(["parse", "case.mp4", "-o", str(tmp_path), "-q", *extra])
    return code, parser, parser_class


def test_parse_wires_the_facade_and_writes_only_the_log(tmp_path, capsys):
    code, parser, parser_class = run_parse(
        tmp_path, ["--signals", "pedals", "--rate", "pedals=30", "--fast"]
    )
    assert code == 0
    assert parser_class.call_args.kwargs["signals"] == GROUPS["pedals"]
    call = parser.parse_video.call_args
    assert call.kwargs["rates"] == {Signal.PEDALS: 30.0}
    assert call.kwargs["fast"] is True
    printed = capsys.readouterr().out.strip().splitlines()
    assert printed == [str(tmp_path / "case.hudini.jsonl.gz")]
    assert not (tmp_path / "case.html").exists()


def test_parse_progress_bar_fills_to_the_total(tmp_path, capsys):
    parser = MagicMock()
    parser.parse_video.return_value = make_parse_log()
    with (
        patch("hudini.parser.Parser", return_value=parser),
        patch(
            "hudini.video.probe_video",
            return_value=VideoInfo(
                filename="case.mp4", width=1920, height=1080, codec="h264", duration_s=10.0
            ),
        ),
    ):
        assert main(["parse", "case.mp4", "-o", str(tmp_path)]) == 0
    assert "10.00/10.00" in capsys.readouterr().err


class TestConfigureLogging:
    def test_warnings_are_suppressed_by_default(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            configure_logging(verbose=False, quiet=False)
            warnings.warn("noise", FutureWarning)
        assert caught == []

    def test_verbose_keeps_the_warnings(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            configure_logging(verbose=True, quiet=False)
            warnings.warn("signal", FutureWarning)
        assert [str(entry.message) for entry in caught] == ["signal"]


def test_frame_prints_the_record_of_one_image(tmp_path, capsys):
    import cv2

    image = tmp_path / "frame.png"
    cv2.imwrite(str(image), np.zeros((40, 60, 3), dtype=np.uint8))
    parser = MagicMock()
    parser.parse_frame.return_value = [
        Observation(time_s=0.0, frame=0, signal=Signal.LAYOUT, key=(), value=GEOMETRY, score=1.0),
        Observation(
            time_s=0.0,
            frame=0,
            signal=Signal.STATUS,
            key=(2,),
            value=PodStatus(status=Status.ACTIVE),
            score=0.9,
        ),
    ]
    with (
        patch("hudini.parser.Parser", return_value=parser),
        patch("hudini.cli.Catalog"),
    ):
        code = main(["frame", str(image), "--signals", "status", "-q"])
    assert code == 0
    record = json.loads(capsys.readouterr().out)
    assert record["frame_index"] == 0
    assert record["pods"][0]["status"] == {"value": "active", "score": 0.9}
    (rgb,) = parser.parse_frame.call_args.args
    assert rgb.shape == (40, 60, 3)


def test_frame_refuses_an_unreadable_image(tmp_path, capsys):
    with patch("hudini.parser.Parser"):
        code = main(["frame", str(tmp_path / "missing.png"), "-q"])
    assert code == 2
    assert "cannot read" in capsys.readouterr().err


GEOMETRY = FrameGeometry(
    region=Box(x=0, y=0, w=400, h=400),
    pods=(PodGeometry(column=2, role=Role.INSTRUMENT, box=Box(x=100, y=380, w=100, h=20)),),
)


@pytest.fixture
def written_log(tmp_path) -> Callable[..., str]:
    """A finished two-frame log: status active at t=0, cleared at t=1."""

    def _write() -> str:
        header = Header(
            log_version=1,
            version="1.0.0",
            created_at="2026-08-07T14:12:03+02:00",
            video=VideoInfo(filename="case.mp4", width=1920, height=1080, codec="h264"),
            requested=("status",),
            signals=(Signal.STATUS,),
            rates={Signal.STATUS: 1.0},
            batch_size=1,
            device="cpu",
        )
        path = tmp_path / "case.hudini.jsonl.gz"
        with LogWriter(path) as writer:
            writer.write_header(header)
            for time_s, value in ((0.0, PodStatus(status=Status.ACTIVE)), (1.0, None)):
                writer.write(
                    Observation(
                        time_s=time_s,
                        frame=int(time_s * 30),
                        signal=Signal.LAYOUT,
                        key=(),
                        value=GEOMETRY,
                        score=1.0,
                    )
                )
                writer.write(
                    Observation(
                        time_s=time_s,
                        frame=int(time_s * 30),
                        signal=Signal.STATUS,
                        key=(2,),
                        value=value,
                        score=0.9,
                    )
                )
        return str(path)

    return _write


class TestQuery:
    def test_state_carries_forward_and_absence_clears(self, written_log, capsys):
        path = written_log()
        assert main(["query", path, "--at", "0:00:00.5", "--signal", "status"]) == 0
        halfway = json.loads(capsys.readouterr().out)
        assert halfway["state"] == [
            {
                "t": 0.0,
                "f": 0,
                "s": "status",
                "k": [2],
                "v": {"type": "status", "status": "active"},
                "c": 0.9,
            }
        ]
        assert main(["query", path, "--signal", "status"]) == 0
        end = json.loads(capsys.readouterr().out)
        assert end == {"at_s": 1.0, "state": []}

    def test_fresh_shows_the_frame_observations(self, written_log, capsys):
        assert main(["query", written_log(), "--at", "0", "--show", "fresh"]) == 0
        fresh = json.loads(capsys.readouterr().out)["fresh"]
        assert [row["s"] for row in fresh] == ["layout", "status"]

    def test_lanes_project_the_frame(self, written_log, capsys):
        path = written_log()
        with patch("hudini.cli.Catalog") as catalog:
            catalog.load.return_value.pedal_action.return_value = None
            assert main(["query", path, "--at", "0", "--show", "lanes"]) == 0
        rows = json.loads(capsys.readouterr().out)["lanes"]
        assert any(row["lane"] == "status" and row["value"] == "active" for row in rows)


class TestCatalog:
    ENTRIES = (
        CatalogEntry(
            name="monopolar curved scissors",
            display_name="monopolar curved scissors",
            type="monopolar_cautery",
            pedals=PedalPair(
                yellow=PedalAction(labels=("coag",)), blue=PedalAction(labels=("cut",))
            ),
        ),
        CatalogEntry(
            name="force bipolar",
            display_name="force bipolar",
            type="bipolar_grasper",
            pedals=PedalPair(
                yellow=PedalAction(labels=("grip", "strong")), blue=PedalAction(labels=("coag",))
            ),
        ),
        CatalogEntry(
            name="round tip scissors",
            display_name="round tip scissors",
            type="cold_scissors",
            pedals=None,
        ),
        CatalogEntry(
            name="small clip applier",
            display_name="small clip applier",
            type="clip_applier",
            pedals=PedalPair(yellow=PedalAction(labels=()), blue=PedalAction(labels=())),
        ),
        CatalogEntry(
            name="arm stowed",
            display_name="arm stowed",
            type="system_message",
            pedals=PedalPair(yellow=PedalAction(labels=()), blue=PedalAction(labels=())),
        ),
    )

    def _pods(self, capsys, arguments: list[str]) -> str:
        with patch("hudini.cli.Catalog") as catalog:
            catalog.load.return_value.entries = self.ENTRIES
            assert main(["catalog", *arguments]) == 0
        return capsys.readouterr().out

    def test_pods_table_sorts_by_type_and_separates_system_messages(self, capsys):
        lines = self._pods(capsys, []).splitlines()
        assert lines[0].split() == ["Instrument", "Type", "Yellow", "Blue"]
        assert set(lines[1]) == {"─", " "}
        names = [line.split("  ")[0] for line in lines[2:6]]
        assert names == [
            "force bipolar",
            "small clip applier",
            "round tip scissors",
            "monopolar curved scissors",
        ]
        tail = lines[lines.index("") + 1 :]
        assert tail[0] == "System message"
        assert "arm stowed" in tail

    def test_pods_table_marks_every_pedal_case(self, capsys):
        out = self._pods(capsys, [])
        assert "grip→strong" in out
        assert "?" in out
        assert "—" in out
        assert "→  the label can switch from the first to the second" in out
        assert "no pedal data in the catalog" in out

    def test_pods_json_carries_the_pedal_contract(self, capsys):
        rows = json.loads(self._pods(capsys, ["pods", "--json"]))
        by_name = {row["name"]: row for row in rows}
        assert by_name["force bipolar"]["pedals"]["yellow"] == {
            "labels": ["grip", "strong"],
            "idle_label": "grip",
            "press_dependent": True,
        }
        assert by_name["round tip scissors"]["pedals"] is None

    TEMPLATES = (
        PopupTemplate(
            id="energy_preset_applied",
            locale="de",
            text="{surgeon}s '{x}' energievoreinstellung angewendet",
            tail="energievoreinstellung angewendet",
        ),
        PopupTemplate(
            id="energy_preset_applied",
            locale="en",
            text="{surgeon}'s '{x}' energy preset applied",
            tail="energy preset applied",
        ),
    )

    def test_popups_prints_one_locale_with_placeholders(self, capsys):
        with patch("hudini.cli.load_templates", return_value=self.TEMPLATES):
            assert main(["catalog", "popups", "--locale", "de"]) == 0
        out = capsys.readouterr().out
        assert "{surgeon}s '{x}' energievoreinstellung angewendet" in out
        assert "energy preset applied" not in out

    def test_popups_json_carries_id_text_and_tail(self, capsys):
        with patch("hudini.cli.load_templates", return_value=self.TEMPLATES):
            assert main(["catalog", "popups", "--locale", "en", "--json"]) == 0
        rows = json.loads(capsys.readouterr().out)
        assert rows == [
            {
                "id": "energy_preset_applied",
                "text": "{surgeon}'s '{x}' energy preset applied",
                "tail": "energy preset applied",
            }
        ]

    def test_popups_with_an_unknown_locale_fails_on_stderr(self, capsys):
        with patch("hudini.cli.load_templates", return_value=self.TEMPLATES):
            assert main(["catalog", "popups", "--locale", "fr"]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "no popup templates for locale 'fr'" in captured.err


def test_fetch_downloads_every_checkpoint_and_the_ocr_pair(capsys):
    with (
        patch(
            "hudini.sensors.resolve_checkpoint",
            side_effect=lambda file: Path("/cache") / file,
        ),
        patch(
            "hudini.ocr.ensure_models",
            return_value=[Path("/cache/ocr_det"), Path("/cache/ocr_rec")],
        ) as ensure,
    ):
        assert main(["fetch", "--ocr-model-size", "small"]) == 0
    assert ensure.call_args.args == ("small",)
    assert capsys.readouterr().out.splitlines() == [
        "/cache/arm_digit_cnn.pt",
        "/cache/camera_state_cnn.pt",
        "/cache/offscreen_digit_cnn.pt",
        "/cache/offscreen_state_rfdetr.pt",
        "/cache/tool_association_rfdetr.pt",
        "/cache/ocr_det",
        "/cache/ocr_rec",
    ]


def test_version_is_a_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("hudini ")


class TestServe:
    def test_a_busy_port_reports_instead_of_crashing(self, tmp_path, capsys):
        from hudini.web.server import make_server

        with make_server(tmp_path, port=0) as blocker:
            port = blocker.server_address[1]
            code = main(["serve", str(tmp_path), "--port", str(port), "--no-browser"])
        assert code == 2
        assert "already in use" in capsys.readouterr().err


class TestTimeline:
    def test_timeline_writes_the_html_and_the_frames_export(self, written_log, tmp_path, capsys):
        path = written_log()
        assert main(["timeline", path, "--frames"]) == 0
        printed = capsys.readouterr().out.strip().splitlines()
        frames_file = tmp_path / "case.frames.jsonl.gz"
        html_file = tmp_path / "case.html"
        assert printed == [str(frames_file), str(html_file)]
        with gzip.open(frames_file, "rt") as file:
            meta_line, *record_lines = [json.loads(line) for line in file]
        assert meta_line["meta"]["video"]["filename"] == "case.mp4"
        assert record_lines[0]["pods"][0]["status"]["value"] == "active"
        page = html_file.read_text()
        assert '"name": "case"' in page

    def test_no_timeline_skips_the_html(self, written_log, tmp_path):
        assert main(["timeline", written_log(), "--frames", "--no-timeline"]) == 0
        assert not (tmp_path / "case.html").exists()
        assert (tmp_path / "case.frames.jsonl.gz").exists()

    def test_timeline_refuses_a_partial_file(self, written_log, tmp_path, capsys):
        finished = Path(written_log())
        partial = tmp_path / "case.hudini.jsonl.partial"
        with gzip.open(finished, "rb") as file:
            lines = file.read().splitlines()
        partial.write_bytes(b"\n".join(lines[:-1]) + b"\n")
        assert main(["timeline", str(partial)]) == 2
        assert "partial" in capsys.readouterr().err


def test_serve_refuses_a_missing_directory(tmp_path, capsys):
    from hudini.cli import main as cli_main

    assert cli_main(["serve", str(tmp_path / "nope")]) == 2
    assert "not a directory" in capsys.readouterr().err
