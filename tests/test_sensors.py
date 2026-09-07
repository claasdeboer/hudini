"""Unit tests for hudini.sensors.

Model calls are patched at the sensors namespace; OCR engines and
detectors are fakes. The registry validations themselves ran at import.
"""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hudini.catalog import Catalog, _CatalogFile, _RawEntry, _RawPedals
from hudini.detection import Detection
from hudini.hsv import PressRead
from hudini.ocr import OCRResult
from hudini.schema import (
    Box,
    FrameGeometry,
    Instrument,
    Laser,
    Observation,
    OffscreenBarStatus,
    PedalColor,
    PodGeometry,
    PodStatus,
    PopupMessage,
    PopupStack,
    Press,
    Role,
    Signal,
    Status,
)
from hudini.sensors import (
    HF_REPO,
    HF_REVISION,
    LANE_PROJECTORS,
    REGISTRY,
    ArmSensor,
    BannerSensor,
    Crop,
    Engines,
    InstrumentSensor,
    LaserSensor,
    LayoutSensor,
    LoadConfig,
    OffscreenSensor,
    PedalLabelSensor,
    PedalsSensor,
    PopupsSensor,
    Result,
    SensorSpec,
    StatusSensor,
    Task,
    ToolAssociationSensor,
    execution_levels,
    resolve_checkpoint,
    wake_on_new_banner,
    wake_on_popup_change,
)
from hudini.views import FrameState, iter_frame_states

GEOMETRY = FrameGeometry(
    region=Box(x=50, y=20, w=400, h=400),
    pods=(
        PodGeometry(column=1, role=Role.CAMERA, box=Box(x=50, y=380, w=100, h=40)),
        PodGeometry(
            column=2,
            role=Role.INSTRUMENT,
            box=Box(x=150, y=390, w=100, h=30),
            popups=(Box(x=150, y=360, w=100, h=25),),
        ),
    ),
    banner=Box(x=50, y=20, w=400, h=30),
)


class FakeOCR:
    """OCR double: maps each crop to a preset read, through the real join."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)

    def read(self, images: list[np.ndarray]) -> list[OCRResult]:
        return [
            OCRResult.from_lines([(text, 0.9)] if text else [])
            for text, _image in zip(self._texts, images, strict=True)
        ]

    def read_line(self, images: list[np.ndarray]) -> list[OCRResult]:
        return self.read(images)


class FakeDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def detect(self, images: list[np.ndarray]) -> list[list[Detection]]:
        return [self._detections for _image in images]


@pytest.fixture
def catalog() -> Catalog:
    entries = (
        _RawEntry(
            name="force instrument",
            type="grasper",
            pedals=_RawPedals(yellow=("grip", "strong"), blue=("bipolar",)),
        ),
        _RawEntry(
            name="plain instrument",
            type="grasper",
            pedals=_RawPedals(yellow=(), blue=("coag",)),
        ),
    )
    return Catalog(_CatalogFile(color_locales={"en": {}}, entries=entries))


@pytest.fixture
def make_observation() -> Callable[..., Observation]:
    def _make(signal: Signal, key: tuple, value: object) -> Observation:
        return Observation(time_s=1.0, frame=0, signal=signal, key=key, value=value, score=1.0)

    return _make


@pytest.fixture
def frame_state(make_observation) -> Callable[..., FrameState]:
    """A FrameState holding the shared geometry plus extra observations."""

    def _make(*extra: tuple) -> FrameState:
        log = [make_observation(Signal.LAYOUT, (), GEOMETRY)]
        log += [make_observation(signal, key, value) for signal, key, value in extra]
        (fs,) = list(iter_frame_states(log))
        return fs

    return _make


RGB = np.zeros((440, 500, 3), dtype=np.uint8)


class TestRegistry:
    def test_every_signal_has_a_sensor_and_a_projector(self):
        assert set(REGISTRY) == set(Signal)
        assert set(LANE_PROJECTORS) == set(Signal)

    def test_the_dag_levels_match_the_design(self):
        assert execution_levels(set(Signal)) == [
            [Signal.LAYOUT],
            [
                Signal.ARM,
                Signal.BANNER,
                Signal.INSTRUMENT,
                Signal.LASER,
                Signal.OFFSCREEN,
                Signal.PEDALS,
                Signal.POPUPS,
                Signal.STATUS,
                Signal.TOOL_ASSOCIATION,
            ],
            [Signal.PEDAL_LABEL],
        ]

    def test_a_selection_keeps_its_own_levels(self):
        selected = {Signal.LAYOUT, Signal.INSTRUMENT, Signal.PEDALS, Signal.PEDAL_LABEL}
        assert execution_levels(selected) == [
            [Signal.LAYOUT],
            [Signal.INSTRUMENT, Signal.PEDALS],
            [Signal.PEDAL_LABEL],
        ]

    @pytest.mark.parametrize(
        ("rate", "thrift_rate"),
        [(0.0, 1.0), (-1.0, 1.0), (1.0, 2.0)],
    )
    def test_an_invalid_spec_raises(self, rate, thrift_rate):
        with pytest.raises(ValueError):
            SensorSpec(name=Signal.STATUS, payload=PodStatus, rate=rate, thrift_rate=thrift_rate)


def _frame_task(rgb: np.ndarray, region: Box | None) -> Task:
    """The session's level-0 convention for the layout sensor."""
    height, width = rgb.shape[:2]
    return Task(
        key=(), crops=(Crop(rgb=rgb, box=Box(x=0, y=0, w=width, h=height)),), context=(region,)
    )


class TestLayoutSensor:
    def test_an_all_black_frame_reads_as_absent(self):
        sensor = LayoutSensor()
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        (result,) = sensor.read([_frame_task(black, None)])
        assert result == Result.absent()

    def test_a_cached_region_skips_detection(self):
        sensor = LayoutSensor()
        frame = np.zeros((440, 500, 3), dtype=np.uint8)
        frame[20:420, 50:450] = 80
        with patch("hudini.sensors.detect_region") as detect:
            (result,) = sensor.read([_frame_task(frame, Box(x=50, y=20, w=400, h=400))])
        detect.assert_not_called()
        assert result == Result.absent()  # a plain gray region has no pods

    def test_a_frame_with_pods_reads_as_geometry(self):
        sensor = LayoutSensor()
        frame = np.zeros((440, 500, 3), dtype=np.uint8)
        frame[20:420, 50:450] = 80
        frame[20 + 350, 50:450] = 255  # a status border across all columns
        (result,) = sensor.read([_frame_task(frame, None)])
        assert isinstance(result.value, FrameGeometry)
        assert result.value.region == Box(x=50, y=20, w=400, h=400)


class FakeStatusClassifier:
    """CNN double: every pod reads active."""

    def classify(self, pods: list[np.ndarray]) -> list[tuple[Status, float]]:
        return [(Status.ACTIVE, 0.9) for _pod in pods]


class TestStatusSensor:
    def test_the_camera_pod_goes_to_the_cnn_and_instruments_to_the_rule(self, frame_state):
        sensor = StatusSensor(classifier=FakeStatusClassifier())
        tasks = sensor.plan(frame_state(), RGB)
        assert [task.key for task in tasks] == [(1,), (2,)]
        assert [task.context for task in tasks] == [(Role.CAMERA,), (Role.INSTRUMENT,)]
        with (
            patch("hudini.sensors.warning_score", return_value=None),
            patch(
                "hudini.sensors.classify_instrument_status",
                return_value=(Status.INACTIVE, 0.1),
            ) as rule,
        ):
            results = sensor.read(tasks)
        assert rule.call_count == 1
        assert results[0].value == PodStatus(status=Status.ACTIVE)
        assert results[1].value == PodStatus(status=Status.INACTIVE)

    def test_a_camera_warning_beats_the_cnn(self, frame_state):
        sensor = StatusSensor(classifier=FakeStatusClassifier())
        tasks = [task for task in sensor.plan(frame_state(), RGB) if task.key == (1,)]
        with patch("hudini.sensors.warning_score", return_value=0.3):
            (result,) = sensor.read(tasks)
        assert result == Result(value=PodStatus(status=Status.WARNING), score=0.3)


class FakeArmClassifier:
    def classify(self, pods: list[np.ndarray]) -> list[tuple[int, float]]:
        return [(3, 0.8) for _pod in pods]


class TestArmSensor:
    def test_reads_the_digit_of_every_pod(self, frame_state):
        sensor = ArmSensor(classifier=FakeArmClassifier())
        tasks = sensor.plan(frame_state(), RGB)
        results = sensor.read(tasks)
        assert [task.key for task in tasks] == [(1,), (2,)]
        assert results[0].value.digit == 3


class TestInstrumentSensor:
    def test_skips_the_camera_pod(self, frame_state, catalog):
        sensor = InstrumentSensor(ocr=FakeOCR([]), catalog=catalog)
        tasks = sensor.plan(frame_state(), RGB)
        assert [task.key for task in tasks] == [(2,)]

    def test_a_match_updates_with_the_catalog_name(self, frame_state, catalog):
        sensor = InstrumentSensor(
            ocr=FakeOCR(["force instrument", "force instrument"]), catalog=catalog
        )
        (result,) = sensor.read(sensor.plan(frame_state(), RGB))
        assert isinstance(result.value, Instrument)
        assert result.value.name == "force instrument"
        assert result.value.match == 100

    def test_unmatched_text_holds(self, frame_state, catalog):
        sensor = InstrumentSensor(ocr=FakeOCR(["zzzzzzzz", "zzzzzzzz"]), catalog=catalog)
        (result,) = sensor.read(sensor.plan(frame_state(), RGB))
        assert result is None

    def test_an_empty_pod_clears(self, frame_state, catalog):
        sensor = InstrumentSensor(ocr=FakeOCR(["", ""]), catalog=catalog)
        (result,) = sensor.read(sensor.plan(frame_state(), RGB))
        assert result == Result.absent()


class TestPedalsSensor:
    def test_reads_both_colors_of_every_instrument_pod(self, frame_state):
        sensor = PedalsSensor()
        tasks = sensor.plan(frame_state(), RGB)
        assert [task.key for task in tasks] == [(2, PedalColor.YELLOW), (2, PedalColor.BLUE)]
        reads = {
            PedalColor.YELLOW: PressRead(pressed=True, score=0.9),
            PedalColor.BLUE: PressRead(pressed=False, score=0.1),
        }
        with patch("hudini.sensors.classify_pedals", return_value=reads):
            results = sensor.read(tasks)
        assert results[0].value == Press(pressed=True)
        assert results[1].value == Press(pressed=False)


class TestLaserSensor:
    def test_plans_only_the_camera_pod(self, frame_state):
        sensor = LaserSensor()
        tasks = sensor.plan(frame_state(), RGB)
        assert [task.key for task in tasks] == [(1,)]

    def test_reads_the_rule_verdict(self, frame_state):
        sensor = LaserSensor()
        tasks = sensor.plan(frame_state(), RGB)
        with patch("hudini.sensors.classify_laser", return_value=(True, 0.8)):
            (result,) = sensor.read(tasks)
        assert result.value == Laser(on=True)
        assert result.score == 0.8


class TestPedalLabelSensor:
    def _press(self, color: PedalColor = PedalColor.YELLOW) -> tuple:
        return (Signal.PEDALS, (2, color), Press(pressed=True))

    def _instrument(self, name: str) -> tuple:
        value = Instrument(name=name, match=95.0, raw=name, window="narrow")
        return (Signal.INSTRUMENT, (2,), value)

    def test_a_press_dependent_pedal_gets_a_task(self, frame_state, catalog):
        sensor = PedalLabelSensor(ocr=FakeOCR([]), catalog=catalog)
        fs = frame_state(self._press(), self._instrument("force instrument"))
        assert [task.key for task in sensor.plan(fs, RGB)] == [(2, PedalColor.YELLOW)]

    def test_a_catalog_answered_pedal_gets_none(self, frame_state, catalog):
        sensor = PedalLabelSensor(ocr=FakeOCR([]), catalog=catalog)
        fs = frame_state(self._press(PedalColor.BLUE), self._instrument("plain instrument"))
        assert sensor.plan(fs, RGB) == []

    def test_an_unknown_instrument_still_reads(self, frame_state, catalog):
        sensor = PedalLabelSensor(ocr=FakeOCR([]), catalog=catalog)
        fs = frame_state(self._press())
        assert len(sensor.plan(fs, RGB)) == 1

    def test_an_unpressed_pedal_gets_none(self, frame_state, catalog):
        sensor = PedalLabelSensor(ocr=FakeOCR([]), catalog=catalog)
        fs = frame_state((Signal.PEDALS, (2, PedalColor.YELLOW), Press(pressed=False)))
        assert sensor.plan(fs, RGB) == []

    def test_read_slices_the_cell_and_abstains_on_empty_text(self, catalog):
        sensor = PedalLabelSensor(ocr=FakeOCR(["STRONG", ""]), catalog=catalog)
        pod = Crop(rgb=np.zeros((30, 100, 3), dtype=np.uint8), box=Box(x=0, y=0, w=100, h=30))
        tasks = [
            Task(key=(2, PedalColor.YELLOW), crops=(pod,)),
            Task(key=(3, PedalColor.YELLOW), crops=(pod,)),
        ]
        with patch(
            "hudini.sensors.crop_pedal_action_label",
            return_value=np.zeros((10, 10, 3), dtype=np.uint8),
        ):
            first, second = sensor.read(tasks)
        assert first.value.text == "strong"
        assert second is None

    def test_an_unreadable_cell_abstains(self, catalog):
        sensor = PedalLabelSensor(ocr=FakeOCR([]), catalog=catalog)
        pod = Crop(rgb=np.zeros((30, 100, 3), dtype=np.uint8), box=Box(x=0, y=0, w=100, h=30))
        with patch(
            "hudini.sensors.crop_pedal_action_label",
            return_value=np.zeros((0, 0, 3), dtype=np.uint8),
        ):
            (result,) = sensor.read([Task(key=(2, PedalColor.YELLOW), crops=(pod,))])
        assert result is None


class TestPopupsSensor:
    def test_one_task_per_column_carries_the_segment_crops(self, frame_state):
        sensor = PopupsSensor(ocr=FakeOCR(["check arm 2"]))
        tasks = sensor.plan(frame_state(), RGB)
        assert [task.key for task in tasks] == [(1,), (2,)]
        assert [len(task.crops) for task in tasks] == [0, 1]
        results = sensor.read(tasks)
        assert results[0].value == PopupStack()
        assert results[1].value == PopupStack(
            messages=(PopupMessage(text="check arm 2", box=Box(x=150, y=360, w=100, h=25)),)
        )

    def test_a_segment_that_reads_as_nothing_is_dropped(self, frame_state):
        sensor = PopupsSensor(ocr=FakeOCR([""]))
        results = sensor.read(sensor.plan(frame_state(), RGB))
        assert results[1].value == PopupStack()


class TestBannerSensor:
    def test_reads_the_banner_when_present(self, frame_state):
        sensor = BannerSensor(ocr=FakeOCR(["table motion"]))
        (result,) = sensor.read(sensor.plan(frame_state(), RGB))
        assert result.value.text == "table motion"

    def test_a_missing_banner_reads_as_absence_without_pixels(self, make_observation):
        geometry = FrameGeometry(region=GEOMETRY.region, pods=GEOMETRY.pods, banner=None)
        (fs,) = list(iter_frame_states([make_observation(Signal.LAYOUT, (), geometry)]))
        sensor = BannerSensor(ocr=FakeOCR([]))
        tasks = sensor.plan(fs, RGB)
        assert [task.crops for task in tasks] == [()]
        assert sensor.read(tasks) == [Result.absent()]


class TestDetectorSensors:
    DETECTION: ClassVar[Detection] = Detection(
        label="active", score=0.91, box=Box(x=10, y=100, w=20, h=300)
    )

    def test_offscreen_bars_come_back_in_full_frame_pixels(self, frame_state):
        digit_classifier = MagicMock()
        digit_classifier.classify.return_value = [(4, 0.8)]
        sensor = OffscreenSensor(
            detector=FakeDetector([self.DETECTION]), digit_classifier=digit_classifier
        )
        tasks = sensor.plan(frame_state(), RGB)
        (result,) = sensor.read(tasks)
        (bar,) = result.value.bars
        assert bar.box == Box(x=60, y=120, w=20, h=300)
        assert bar.status is OffscreenBarStatus.ACTIVE
        assert bar.arm == 4

    def test_tool_badges_parse_the_arm_from_the_class(self, frame_state):
        detection = Detection(label="3", score=0.88, box=Box(x=1, y=2, w=3, h=4))
        sensor = ToolAssociationSensor(detector=FakeDetector([detection]))
        (result,) = sensor.read(sensor.plan(frame_state(), RGB))
        (badge,) = result.value.badges
        assert badge.arm == 3
        assert badge.box == Box(x=51, y=22, w=3, h=4)

    def test_build_rejects_a_checkpoint_without_a_required_class(self):
        detector = MagicMock()
        detector.labels = ("offscreen", "active")
        with (
            patch("hudini.sensors.hf_hub_download", return_value="/cache/model.pt"),
            patch("hudini.sensors.load_detector", return_value=detector),
            pytest.raises(ValueError, match="inactive"),
        ):
            OffscreenSensor.build(engines=Engines(catalog=MagicMock()), config=LoadConfig())

    def test_build_accepts_a_checkpoint_teaching_extra_classes(self):
        detector = MagicMock()
        detector.labels = ("badge", "1", "2", "3", "4", "5")
        with (
            patch("hudini.sensors.hf_hub_download", return_value="/cache/model.pt"),
            patch("hudini.sensors.load_detector", return_value=detector),
        ):
            sensor = ToolAssociationSensor.build(
                engines=Engines(catalog=MagicMock()), config=LoadConfig()
            )
        assert isinstance(sensor, ToolAssociationSensor)


class TestWakes:
    def test_the_popup_wake_fires_on_a_count_move(self, frame_state, make_observation):
        before = frame_state()
        changed = FrameGeometry(region=GEOMETRY.region, pods=GEOMETRY.pods[:1], banner=None)
        (after,) = list(iter_frame_states([make_observation(Signal.LAYOUT, (), changed)]))
        assert wake_on_popup_change(after, before) is True
        assert wake_on_popup_change(before, before) is False
        assert wake_on_popup_change(before, None) is False

    def test_the_banner_wake_fires_only_on_the_transition(self, frame_state, make_observation):
        without = FrameGeometry(region=GEOMETRY.region, pods=GEOMETRY.pods, banner=None)
        (before,) = list(iter_frame_states([make_observation(Signal.LAYOUT, (), without)]))
        with_banner = frame_state()
        assert wake_on_new_banner(with_banner, before) is True
        assert wake_on_new_banner(with_banner, with_banner) is False
        assert wake_on_new_banner(before, with_banner) is False


def test_resolve_checkpoint_downloads_pinned_and_returns_the_path():
    with patch(
        "hudini.sensors.hf_hub_download",
        return_value="/cache/camera_state_cnn.pt",
    ) as download:
        path = resolve_checkpoint("camera_state_cnn.pt")
    assert path == Path("/cache/camera_state_cnn.pt")
    download.assert_called_once_with(
        repo_id=HF_REPO, filename="camera_state_cnn.pt", revision=HF_REVISION
    )
