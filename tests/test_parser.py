"""Unit tests for hudini.parser.

The registry is replaced by fake sensor classes that carry the real
specs, so selection, expansion, and rate resolution are exercised against
the production declarations without loading any model. Engines and video
IO are patched at the parser namespace; LogWriter runs for real.
"""

from collections.abc import Callable
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hudini.parser import Parser
from hudini.schema import Observation, PodStatus, Signal, Status, VideoInfo
from hudini.sensors import REGISTRY, Engines, LoadConfig, Result, SensorSpec, Task
from hudini.storage import load
from hudini.views import FrameState

VIDEO_INFO = VideoInfo(filename="case.mp4", width=1920, height=1080, codec="h264")


class FakeSensor:
    """A stand-in sensor: the real spec, no models."""

    spec: ClassVar[SensorSpec]
    buildable: ClassVar[bool] = True

    @classmethod
    def build(cls, engines: Engines, config: LoadConfig) -> "FakeSensor | None":
        if not cls.buildable:
            return None
        return cls()

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        return []

    def read(self, tasks: list[Task]) -> list[Result | None]:
        return []


def sensor_class(spec: SensorSpec, buildable: bool = True) -> type[FakeSensor]:
    return type(f"Fake_{spec.name}", (FakeSensor,), {"spec": spec, "buildable": buildable})


@pytest.fixture
def fake_registry() -> dict[Signal, type[FakeSensor]]:
    return {name: sensor_class(sensor.spec) for name, sensor in REGISTRY.items()}


@pytest.fixture
def build_parser(fake_registry) -> Callable[..., tuple[Parser, MagicMock]]:
    """Construct a Parser against the fake registry; returns it with the
    OCREngine mock."""

    def _build(
        registry: dict[Signal, type[FakeSensor]] | None = None, **kwargs
    ) -> tuple[Parser, MagicMock]:
        with (
            patch("hudini.parser.REGISTRY", registry or fake_registry),
            patch("hudini.parser.OCREngine") as ocr,
            patch("hudini.parser.Catalog") as catalog,
            patch("hudini.parser.resolve_device", return_value="cpu"),
        ):
            ocr.return_value.model_size = "tiny"
            catalog.load.return_value.locale = kwargs.get("locale", "en")
            parser = Parser(**kwargs)
        return parser, ocr

    return _build


def make_observation(time_s: float, signal: Signal) -> Observation:
    value = PodStatus(status=Status.ACTIVE) if signal is Signal.STATUS else None
    return Observation(time_s=time_s, frame=0, signal=signal, key=(), value=value, score=1.0)


class TestSelection:
    def test_default_selects_every_signal(self, build_parser):
        parser, _ = build_parser()
        assert set(parser.signals) == set(Signal) - {Signal.LAYOUT}

    def test_after_edges_expand_the_selection(self, build_parser):
        parser, _ = build_parser(signals=["pedal_label"])
        assert set(parser.signals) == {Signal.PEDAL_LABEL, Signal.PEDALS, Signal.INSTRUMENT}

    def test_an_unknown_signal_raises_with_the_valid_names(self):
        with pytest.raises(ValueError, match="pedals"):
            Parser(signals=["bogus"])

    def test_an_unknown_checkpoint_name_raises(self):
        with pytest.raises(ValueError, match="unknown checkpoint"):
            Parser(signals=["arm"], models={"bogus": "x.pt"})

    def test_ocr_loads_only_for_a_signal_that_requires_it(self, build_parser):
        _, ocr = build_parser(signals=["arm"])
        ocr.assert_not_called()
        _, ocr = build_parser(signals=["instrument"], ocr_model_size="small")
        ocr.assert_called_once_with(model_size="small", device="cpu")

    def test_a_failed_build_drops_the_signal_and_its_dependents(self, build_parser, fake_registry):
        registry = dict(fake_registry)
        registry[Signal.INSTRUMENT] = sensor_class(
            REGISTRY[Signal.INSTRUMENT].spec, buildable=False
        )
        parser, _ = build_parser(registry=registry, signals=["pedal_label"])
        assert set(parser.signals) == {Signal.PEDALS}


class TestRates:
    def test_fast_uses_thrift_rates_and_an_override_wins(self, build_parser):
        parser, _ = build_parser(signals=["instrument", "pedals"])
        with patch("hudini.parser.Session") as session_class:
            parser.session(rates={"pedals": 30.0}, fast=True)
        rates = session_class.call_args.kwargs["rates"]
        assert rates[Signal.INSTRUMENT] == REGISTRY[Signal.INSTRUMENT].spec.thrift_rate
        assert rates[Signal.PEDALS] == 30.0

    @pytest.mark.parametrize(
        ("rates", "match"),
        [({"pedals": 10.0}, "unselected"), ({"status": -1.0}, "positive")],
    )
    def test_an_invalid_rate_raises(self, build_parser, rates, match):
        parser, _ = build_parser(signals=["status"])
        with pytest.raises(ValueError, match=match):
            parser.session(rates=rates)


class TestParseVideo:
    def test_writes_the_log_and_returns_what_load_gives(self, tmp_path, build_parser):
        parser, _ = build_parser(signals=["status"])
        observations = [
            make_observation(0.0, Signal.LAYOUT),
            make_observation(0.0, Signal.STATUS),
            make_observation(1.0, Signal.LAYOUT),
        ]
        session = MagicMock()
        session.iter_observations.return_value = iter(observations)
        weights = tmp_path / "camera_state_cnn.pt"
        weights.write_bytes(b"weights")
        with (
            patch("hudini.parser.probe_video", return_value=VIDEO_INFO),
            patch("hudini.parser.iter_frames") as frames,
            patch("hudini.parser.Session", return_value=session) as session_class,
            patch("hudini.parser.resolve_checkpoint", return_value=weights),
        ):
            log = parser.parse_video("case.mp4", output=tmp_path)
        loaded = load(tmp_path / "case.hudini.jsonl.gz")
        assert loaded.header == log.header
        assert loaded.observations == tuple(observations)
        assert loaded.footer == log.footer
        assert log.footer.frames_sampled == 2
        assert frames.call_args.kwargs["fps"] == 1.0
        assert session_class.call_args.kwargs["rates"] == {
            Signal.LAYOUT: 1.0,
            Signal.STATUS: 1.0,
        }

    def test_the_header_states_the_run(self, tmp_path, build_parser):
        parser, _ = build_parser(signals=["status"])
        session = MagicMock()
        session.iter_observations.return_value = iter([make_observation(0.0, Signal.LAYOUT)])
        weights = tmp_path / "camera_state_cnn.pt"
        weights.write_bytes(b"weights")
        with (
            patch("hudini.parser.probe_video", return_value=VIDEO_INFO),
            patch("hudini.parser.iter_frames"),
            patch("hudini.parser.Session", return_value=session),
            patch("hudini.parser.resolve_checkpoint", return_value=weights),
        ):
            log = parser.parse_video("case.mp4", output=tmp_path)
        header = log.header
        assert header.requested == ("status",)
        assert header.signals == (Signal.STATUS,)
        assert Signal.LAYOUT not in header.rates
        assert header.video == VIDEO_INFO
        assert header.models["status"].file == "camera_state_cnn.pt"
        assert header.engines == {}
        assert [rule["name"] for rule in header.corrections] == [
            "complete_layout",
            "debounce:status",
        ]
        assert header.corrections[1]["min_duration_s"] == 0.15

    def test_progress_sees_every_decoded_frame(self, tmp_path, build_parser):
        parser, _ = build_parser(signals=["status"])
        frame = MagicMock(time_s=1.5)
        session = MagicMock()
        session.iter_observations.side_effect = lambda frames: (
            make_observation(0.0, Signal.LAYOUT) for _frame in frames
        )
        seen = []
        with (
            patch("hudini.parser.probe_video", return_value=VIDEO_INFO),
            patch("hudini.parser.iter_frames", return_value=iter([frame])),
            patch("hudini.parser.Session", return_value=session),
            patch("hudini.parser.resolve_checkpoint", return_value=tmp_path / "none.pt"),
        ):
            parser.parse_video("case.mp4", output=tmp_path, progress=seen.append)
        assert seen == [1.5]


def test_parse_frame_feeds_one_frame_at_time_zero(build_parser):
    parser, _ = build_parser(signals=["status"])
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    session = MagicMock()
    session.iter_observations.return_value = iter([make_observation(0.0, Signal.STATUS)])
    with patch("hudini.parser.Session", return_value=session):
        observations = parser.parse_frame(rgb)
    assert observations == [make_observation(0.0, Signal.STATUS)]
    (frames,) = session.iter_observations.call_args.args
    (frame,) = frames
    assert (frame.idx, frame.time_s) == (0, 0.0)
    assert frame.rgb is rgb
