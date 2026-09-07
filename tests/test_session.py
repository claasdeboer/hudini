"""Unit tests for hudini.session.

Sensors are scriptable fakes with real specs; frames carry their script
marker in the pixel values. The batch-invariance and ordering laws are
asserted directly.
"""

from collections.abc import Callable

import numpy as np
import pytest

from hudini.schema import (
    Box,
    FrameGeometry,
    Instrument,
    PodGeometry,
    PodStatus,
    Role,
    Signal,
    Status,
)
from hudini.sensors import Result, SensorSpec, Task
from hudini.session import REGION_RECHECK_SECONDS, UI_LOSS_RESET_SECONDS, Session
from hudini.video import Frame
from hudini.views import FrameState

GEOMETRY = FrameGeometry(
    region=Box(x=0, y=0, w=400, h=400),
    pods=(
        PodGeometry(column=1, role=Role.INSTRUMENT, box=Box(x=0, y=350, w=100, h=50)),
        PodGeometry(column=2, role=Role.INSTRUMENT, box=Box(x=100, y=350, w=100, h=50)),
        PodGeometry(column=3, role=Role.CAMERA, box=Box(x=200, y=350, w=100, h=50)),
        PodGeometry(column=4, role=Role.INSTRUMENT, box=Box(x=300, y=350, w=100, h=50)),
    ),
)

INCOMPLETE_GEOMETRY = FrameGeometry(
    region=Box(x=0, y=0, w=500, h=400),
    pods=(
        PodGeometry(column=1, role=Role.INSTRUMENT, box=None),
        PodGeometry(column=2, role=Role.INSTRUMENT, box=Box(x=100, y=350, w=100, h=50)),
        PodGeometry(column=3, role=Role.INSTRUMENT, box=None),
        PodGeometry(column=4, role=Role.INSTRUMENT, box=Box(x=300, y=350, w=100, h=50)),
    ),
)


def make_frame(index: int, time_s: float, marker: int) -> Frame:
    return Frame(idx=index, time_s=time_s, rgb=np.full((4, 4, 3), marker, dtype=np.uint8))


class FakeLayout:
    """Level 0: maps each frame's pixel marker to a scripted geometry."""

    spec = SensorSpec(
        name=Signal.LAYOUT, payload=FrameGeometry, rate=1.0, thrift_rate=1.0, greedy=True
    )

    def __init__(self, geometries: dict[int, FrameGeometry | None]) -> None:
        self._geometries = geometries
        self.contexts: list[Box | None] = []

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        return []

    def read(self, tasks: list[Task]) -> list[Result | None]:
        results: list[Result | None] = []
        for task in tasks:
            self.contexts.append(task.context[0])
            geometry = self._geometries.get(int(task.crops[0].rgb[0, 0, 0]))
            results.append(
                Result(value=geometry, score=1.0) if geometry is not None else Result.absent()
            )
        return results


class FakeStatus:
    """Level 1, greedy: one reading per frame with geometry."""

    spec = SensorSpec(
        name=Signal.STATUS,
        payload=PodStatus,
        rate=1.0,
        thrift_rate=1.0,
        greedy=True,
        after=(Signal.LAYOUT,),
    )

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        if (Signal.LAYOUT,) not in fs.state:
            return []
        return [Task(key=(1,), crops=())]

    def read(self, tasks: list[Task]) -> list[Result | None]:
        return [Result(value=PodStatus(status=Status.ACTIVE), score=0.9) for _task in tasks]


class FakeInstrument:
    """Level 1, gated: reads only when its clock ticks or its wake fires."""

    def __init__(self, wake=None) -> None:
        self.spec = SensorSpec(
            name=Signal.INSTRUMENT,
            payload=Instrument,
            rate=1.0,
            thrift_rate=1.0,
            after=(Signal.LAYOUT,),
            wake=wake,
        )

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        if (Signal.LAYOUT,) not in fs.state:
            return []
        return [Task(key=(1,), crops=())]

    def read(self, tasks: list[Task]) -> list[Result | None]:
        value = Instrument(name="fake", match=100.0, raw="fake", window="narrow")
        return [Result(value=value, score=1.0) for _task in tasks]


class FakeLabel:
    """Level 2: records the frame states its plan received."""

    spec = SensorSpec(
        name=Signal.PEDAL_LABEL,
        payload=Instrument,
        rate=1.0,
        thrift_rate=1.0,
        greedy=True,
        after=(Signal.LAYOUT, Signal.STATUS),
    )

    def __init__(self) -> None:
        self.seen: list[FrameState] = []

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        self.seen.append(fs)
        return []

    def read(self, tasks: list[Task]) -> list[Result | None]:
        return []


@pytest.fixture
def ui_frames() -> Callable[..., list[Frame]]:
    """Frames at a fixed rate whose markers all map to the shared geometry."""

    def _make(count: int, fps: float = 2.0) -> list[Frame]:
        return [make_frame(index, index / fps, marker=1) for index in range(count)]

    return _make


def _session(*extra_sensors, batch_size: int = 1, geometries: dict | None = None) -> Session:
    sensors = {Signal.LAYOUT: FakeLayout(geometries if geometries is not None else {1: GEOMETRY})}
    rates = {Signal.LAYOUT: 2.0}
    for sensor in extra_sensors:
        sensors[sensor.spec.name] = sensor
        rates[sensor.spec.name] = sensor.spec.rate
    return Session(sensors, rates, batch_size=batch_size)


class TestFrameClock:
    def test_every_frame_emits_exactly_one_layout_observation(self, ui_frames):
        session = _session(geometries={1: GEOMETRY, 2: None})
        frames = [make_frame(0, 0.0, 1), make_frame(5, 0.5, 2), make_frame(10, 1.0, 1)]
        observations = list(session.iter_observations(frames))
        layout = [obs for obs in observations if obs.signal is Signal.LAYOUT]
        assert [obs.frame for obs in layout] == [0, 5, 10]
        assert [obs.value is None for obs in layout] == [False, True, False]

    def test_observations_are_stamped_with_the_frames_identity(self, ui_frames):
        session = _session(FakeStatus())
        (frame,) = ui_frames(1)
        status = [obs for obs in session.iter_observations([frame]) if obs.signal is Signal.STATUS]
        assert status[0].time_s == frame.time_s
        assert status[0].frame == frame.idx
        assert status[0].key == (1,)
        assert status[0].score == 0.9


class TestScheduling:
    def test_a_gated_sensor_reads_on_its_own_ticks(self, ui_frames):
        session = _session(FakeInstrument())
        frames = ui_frames(4, fps=2.5)  # 0.0, 0.4, 0.8, 1.2
        times = [
            obs.time_s
            for obs in session.iter_observations(frames)
            if obs.signal is Signal.INSTRUMENT
        ]
        assert times == [0.0, 1.2]

    def test_a_wake_reads_off_cadence_without_advancing_the_clock(self, ui_frames):
        wake = lambda fs, previous: fs.time_s == 0.5
        session = _session(FakeInstrument(wake=wake))
        frames = ui_frames(3, fps=2.0)  # 0.0, 0.5, 1.0
        times = [
            obs.time_s
            for obs in session.iter_observations(frames)
            if obs.signal is Signal.INSTRUMENT
        ]
        # 0.5 is the wake; 1.0 still ticks, so the wake did not advance.
        assert times == [0.0, 0.5, 1.0]

    def test_a_greedy_sensor_reads_every_frame(self, ui_frames):
        session = _session(FakeStatus())
        frames = ui_frames(4)
        count = sum(1 for obs in session.iter_observations(frames) if obs.signal is Signal.STATUS)
        assert count == 4


class TestLevels:
    def test_a_level_two_plan_sees_the_same_frames_level_one_reading(self, ui_frames):
        label = FakeLabel()
        session = _session(FakeStatus(), label)
        list(session.iter_observations(ui_frames(2)))
        for fs in label.seen:
            fresh_signals = {obs.signal for obs in fs.fresh}
            assert Signal.STATUS in fresh_signals
            assert (Signal.STATUS, 1) in fs.state

    def test_observations_come_out_sorted_by_level_then_signal(self, ui_frames):
        session = _session(FakeStatus(), FakeInstrument(), FakeLabel())
        observations = list(session.iter_observations(ui_frames(2, fps=1.0)))
        per_frame: dict[int, list[Signal]] = {}
        for obs in observations:
            per_frame.setdefault(obs.frame, []).append(obs.signal)
        for signals in per_frame.values():
            assert signals == [Signal.LAYOUT, Signal.INSTRUMENT, Signal.STATUS]


class TestBatchInvariance:
    def test_the_batch_size_never_changes_the_output(self, ui_frames):
        frames = ui_frames(6)
        single = list(_session(FakeStatus(), FakeInstrument()).iter_observations(frames))
        batched_run = list(
            _session(FakeStatus(), FakeInstrument(), batch_size=3).iter_observations(frames)
        )
        assert single == batched_run

    def test_two_runs_are_identical(self, ui_frames):
        frames = ui_frames(5)
        first = list(_session(FakeStatus()).iter_observations(frames))
        second = list(_session(FakeStatus()).iter_observations(frames))
        assert first == second


class TestRegionCacheAndUiLoss:
    def _frames(self) -> list[Frame]:
        frames = [make_frame(0, 0.0, 1), make_frame(1, 0.5, 1)]
        lost = np.arange(1.0, 1.0 + UI_LOSS_RESET_SECONDS + 1.0, 0.5)
        frames += [make_frame(10 + i, float(t), 2) for i, t in enumerate(lost)]
        return frames

    def test_the_region_is_cached_only_from_ui_frames(self):
        layout = FakeLayout({1: GEOMETRY, 2: None})
        session = Session({Signal.LAYOUT: layout}, {Signal.LAYOUT: 2.0})
        list(session.iter_observations(self._frames()[:4]))
        assert layout.contexts[0] is None
        assert layout.contexts[1] == GEOMETRY.region
        assert layout.contexts[2] == GEOMETRY.region  # a short loss keeps the cache

    def test_a_long_ui_loss_clears_held_keys_and_drops_the_cache(self):
        layout = FakeLayout({1: GEOMETRY, 2: None})
        status = FakeStatus()
        session = Session(
            {Signal.LAYOUT: layout, Signal.STATUS: status},
            {Signal.LAYOUT: 2.0, Signal.STATUS: 2.0},
        )
        observations = list(session.iter_observations(self._frames()))
        clears = [obs for obs in observations if obs.signal is Signal.STATUS and obs.value is None]
        assert len(clears) == 1
        assert clears[0].time_s >= 1.0 + UI_LOSS_RESET_SECONDS
        assert layout.contexts[-1] is None  # the cache was dropped

    def test_an_incomplete_layout_never_seeds_the_cache(self):
        layout = FakeLayout({1: INCOMPLETE_GEOMETRY})
        session = Session({Signal.LAYOUT: layout}, {Signal.LAYOUT: 2.0})
        frames = [make_frame(i, i * 0.5, 1) for i in range(4)]
        list(session.iter_observations(frames))
        assert all(context is None for context in layout.contexts)

    def test_a_persistent_incomplete_layout_drops_the_cache(self):
        layout = FakeLayout({1: GEOMETRY, 2: INCOMPLETE_GEOMETRY})
        session = Session({Signal.LAYOUT: layout}, {Signal.LAYOUT: 2.0})
        frames = [make_frame(0, 0.0, 1)] + [
            make_frame(10 + i, 1.0 + i, 2) for i in range(int(REGION_RECHECK_SECONDS) + 2)
        ]
        list(session.iter_observations(frames))
        assert layout.contexts[1] == GEOMETRY.region
        assert layout.contexts[-1] is None

    def test_a_short_incomplete_stretch_keeps_the_cache(self):
        layout = FakeLayout({1: GEOMETRY, 2: INCOMPLETE_GEOMETRY})
        session = Session({Signal.LAYOUT: layout}, {Signal.LAYOUT: 2.0})
        frames = [make_frame(0, 0.0, 1)] + [make_frame(1 + i, 1.0 + i, 2) for i in range(3)]
        frames.append(make_frame(9, 5.0, 1))
        list(session.iter_observations(frames))
        assert all(context == GEOMETRY.region for context in layout.contexts[1:])

    def test_no_reset_without_a_cached_region(self):
        layout = FakeLayout({2: None})
        session = Session({Signal.LAYOUT: layout}, {Signal.LAYOUT: 2.0})
        frames = [make_frame(i, i * 1.0, 2) for i in range(8)]
        observations = list(session.iter_observations(frames))
        assert all(obs.signal is Signal.LAYOUT for obs in observations)


class TestValidation:
    def test_a_session_without_layout_raises(self):
        with pytest.raises(ValueError):
            Session({Signal.STATUS: FakeStatus()}, {Signal.STATUS: 1.0})

    def test_cyclic_dependencies_raise(self):
        instrument = FakeInstrument()
        instrument.spec = SensorSpec(
            name=Signal.INSTRUMENT,
            payload=Instrument,
            rate=1.0,
            thrift_rate=1.0,
            after=(Signal.STATUS,),
        )

        class CyclicStatus(FakeStatus):
            spec = SensorSpec(
                name=Signal.STATUS,
                payload=PodStatus,
                rate=1.0,
                thrift_rate=1.0,
                after=(Signal.INSTRUMENT,),
            )

        with pytest.raises(ValueError):
            Session(
                {
                    Signal.LAYOUT: FakeLayout({1: GEOMETRY}),
                    Signal.INSTRUMENT: instrument,
                    Signal.STATUS: CyclicStatus(),
                },
                {Signal.LAYOUT: 1.0, Signal.INSTRUMENT: 1.0, Signal.STATUS: 1.0},
            )


class NoisyStatus(FakeStatus):
    """A status reading with raw model floats, for the stamp contract."""

    def read(self, tasks: list[Task]) -> list[Result | None]:
        return [
            Result(value=PodStatus(status=Status.ACTIVE), score=0.999977707862854)
            for _task in tasks
        ]


def test_the_stamp_normalizes_scores_and_times():
    session = Session(
        {Signal.LAYOUT: FakeLayout({1: GEOMETRY}), Signal.STATUS: NoisyStatus()},
        {Signal.STATUS: 1.0},
    )
    observations = list(session.iter_observations([make_frame(0, 1 / 3, marker=1)]))
    status = next(obs for obs in observations if obs.signal is Signal.STATUS)
    assert status.score == 1.0
    assert status.time_s == 0.333
