"""Turns the frames of one video into observations in one pass.

:class:`Session` owns the mutable state of one video: the sensor clocks,
the cached active region, and the running key state. Each batch of
frames goes through three phases: schedule, read one level at a time,
and emit. Batch size never changes the output. For that, a sensor's
``plan`` and ``wake`` read only state of lower levels.

After :data:`UI_LOSS_RESET_SECONDS` without UI, the session marks every
held key as absent and clears the cached region. Only a complete layout
seeds the region cache, and :data:`REGION_RECHECK_SECONDS` of incomplete
layouts clear it again.
"""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from itertools import batched

from hudini.layout import layout_complete
from hudini.schema import Box, FrameGeometry, Observation, Signal
from hudini.sensors import Crop, Result, Sensor, Task, topological_levels
from hudini.video import Frame
from hudini.views import FrameState

UI_LOSS_RESET_SECONDS = 5.0

REGION_RECHECK_SECONDS = 10.0
"""Drop the cached region after this long of parsed but incomplete layouts."""


def _apply(state: dict[tuple, Observation], obs: Observation) -> None:
    """The three-way contract on a state dict: update, or clear on None."""
    entry = (obs.signal, *obs.key)
    if obs.value is None:
        state.pop(entry, None)
    else:
        state[entry] = obs


@dataclass(slots=True)
class _PendingFrame:
    """One frame mid-pass: what was decided for it and read so far.

    Attributes:
        due: the signals to read on this frame.
        ui_lost: whether this frame crossed the UI-loss threshold.
        observations: everything read so far, unsorted until emission.
    """

    frame: Frame
    due: set[Signal]
    ui_lost: bool
    observations: list[Observation] = field(default_factory=list)

    def frame_state(self, before: dict[tuple, Observation]) -> FrameState:
        """This frame's view for ``plan``: the state before it, plus its
        own observations so far."""
        state = dict(before)
        for obs in self.observations:
            _apply(state, obs)
        return FrameState(
            time_s=self.frame.time_s,
            frame=self.frame.idx,
            fresh=tuple(self.observations),
            state=state,
        )


class Session:
    """One stateful pass of loaded sensors over one frame source.

    Args:
        sensors: the built sensor instances by signal. The layout sensor
            is required, because it is the frame clock.
        rates: the configured rate per signal, in frames per second.
        batch_size: frames handed to the model levels at once.

    Raises:
        ValueError: no layout sensor, or cyclic sensor dependencies.
    """

    def __init__(
        self,
        sensors: Mapping[Signal, Sensor],
        rates: Mapping[Signal, float],
        batch_size: int = 1,
    ) -> None:
        if Signal.LAYOUT not in sensors:
            raise ValueError("a session needs the layout sensor")
        self._sensors = dict(sensors)
        self._batch_size = batch_size
        self._levels = topological_levels(
            {name: sensor.spec.after for name, sensor in self._sensors.items()}
        )
        self._level_of = {name: index for index, level in enumerate(self._levels) for name in level}
        self._intervals = {
            name: 1.0 / rates[name]
            for name, sensor in self._sensors.items()
            if not sensor.spec.greedy and name is not Signal.LAYOUT
        }
        self._last_tick: dict[Signal, float] = {}
        self._state: dict[tuple, Observation] = {}
        self._cached_region: Box | None = None
        self._no_ui_since: float | None = None
        self._incomplete_since: float | None = None
        self._previous_fs: FrameState | None = None

    def iter_observations(self, frames: Iterable[Frame]) -> Iterator[Observation]:
        """One observation per reading, sorted by time, level, and signal."""
        for chunk in batched(frames, self._batch_size):
            pending_frames = self._schedule_frames(chunk)
            for level in self._levels[1:]:
                self._read_level(level, pending_frames)
            yield from self._emit_observations(pending_frames)

    # Phase 1, schedule: layout, the region cache, and what is due.

    def _schedule_frames(self, chunk: tuple[Frame, ...]) -> list[_PendingFrame]:
        """Run level 0 for each frame in order and decide what to read.

        Sequential on purpose: the region cache must update between
        frames, or the batch size would change what layout sees.
        """
        pending_frames: list[_PendingFrame] = []
        state = dict(self._state)
        for frame in chunk:
            obs, ui_lost = self._observe_layout(frame)
            _apply(state, obs)
            fs = FrameState(time_s=frame.time_s, frame=frame.idx, fresh=(obs,), state=dict(state))
            pending = _PendingFrame(
                frame=frame, due=self._due_signals(fs, frame.time_s), ui_lost=ui_lost
            )
            pending.observations.append(obs)
            pending_frames.append(pending)
            self._previous_fs = fs
        return pending_frames

    def _observe_layout(self, frame: Frame) -> tuple[Observation, bool]:
        """One frame's layout reading, and whether the UI has been lost
        long enough to reset."""
        height, width = frame.rgb.shape[:2]
        task = Task(
            key=(),
            crops=(Crop(rgb=frame.rgb, box=Box(x=0, y=0, w=width, h=height)),),
            context=(self._cached_region,),
        )
        (result,) = self._sensors[Signal.LAYOUT].read([task])
        if result is None:
            result = Result.absent()
        ui_lost = False
        if isinstance(result.value, FrameGeometry):
            self._no_ui_since = None
            # Only a complete layout may seed the cache. A fade-in or a
            # mis-detected region yields a partial pod row at best.
            if layout_complete(result.value):
                self._cached_region = result.value.region
                self._incomplete_since = None
            elif self._cached_region is None:
                self._incomplete_since = None
            elif self._incomplete_since is None:
                self._incomplete_since = frame.time_s
            elif frame.time_s - self._incomplete_since >= REGION_RECHECK_SECONDS:
                self._cached_region = None
                self._incomplete_since = None
        elif self._no_ui_since is None:
            self._no_ui_since = frame.time_s
            self._incomplete_since = None
        elif (
            self._cached_region is not None
            and frame.time_s - self._no_ui_since >= UI_LOSS_RESET_SECONDS
        ):
            self._cached_region = None
            ui_lost = True
        return self._stamp(frame, Signal.LAYOUT, (), result), ui_lost

    def _due_signals(self, fs: FrameState, time_s: float) -> set[Signal]:
        """The signals to read on this frame: greedy, ticked, or woken.

        Consumes the ticked sensors' clocks; a wake reads out of cadence
        and leaves the clock alone.
        """
        due: set[Signal] = set()
        for name, sensor in self._sensors.items():
            if name is Signal.LAYOUT:
                continue
            if sensor.spec.greedy or self._consume_tick(name, time_s) or self._woken(sensor, fs):
                due.add(name)
        return due

    def _woken(self, sensor: Sensor, fs: FrameState) -> bool:
        """Whether the sensor's wake condition fires on this frame."""
        return sensor.spec.wake is not None and sensor.spec.wake(fs, self._previous_fs)

    def _consume_tick(self, name: Signal, time_s: float) -> bool:
        """Take one sensor's pending clock tick, advancing the clock."""
        last = self._last_tick.get(name)
        if last is None or time_s - last >= self._intervals[name]:
            self._last_tick[name] = time_s
            return True
        return False

    # Phase 2, read one level: plan per frame, read batched per sensor.

    def _read_level(self, level: list[Signal], pending_frames: list[_PendingFrame]) -> None:
        """Plan every frame's tasks for one level, then read each sensor
        once over the whole batch.

        Each frame plans against the state before it plus its own lower
        levels, so batch and single-frame processing see the same inputs.
        """
        jobs: dict[Signal, list[tuple[_PendingFrame, Task]]] = {name: [] for name in level}
        rolling = dict(self._state)
        for pending in pending_frames:
            fs = pending.frame_state(rolling)
            for name in level:
                if name in pending.due:
                    for task in self._sensors[name].plan(fs, pending.frame.rgb):
                        jobs[name].append((pending, task))
            for obs in pending.observations:
                _apply(rolling, obs)
        for name in level:
            if not jobs[name]:
                continue
            results = self._sensors[name].read([task for _pending, task in jobs[name]])
            for (pending, task), result in zip(jobs[name], results, strict=True):
                if result is not None:
                    pending.observations.append(self._stamp(pending.frame, name, task.key, result))

    # Phase 3, emit: order, apply, and yield.

    def _emit_observations(self, pending_frames: list[_PendingFrame]) -> Iterator[Observation]:
        """Sort each frame's observations, apply them to the session
        state, and yield them, the UI-loss absences included."""
        for pending in pending_frames:
            if pending.ui_lost:
                pending.observations.extend(self._held_key_absences(pending.frame))
            pending.observations.sort(key=self._sort_key)
            for obs in pending.observations:
                _apply(self._state, obs)
                yield obs

    def _held_key_absences(self, frame: Frame) -> list[Observation]:
        """Confirmed absences for every held key, after a long UI loss."""
        return [
            Observation(
                time_s=round(frame.time_s, 3),
                frame=frame.idx,
                signal=entry[0],
                key=entry[1:],
                value=None,
                score=0.0,
            )
            for entry in self._state
            if entry[0] is not Signal.LAYOUT
        ]

    def _stamp(self, frame: Frame, name: Signal, key: tuple, result: Result) -> Observation:
        """A reading as an observation: the frame's identity added.

        The stamp normalizes the wire numbers: scores round to four
        decimals and times to milliseconds.
        """
        return Observation(
            time_s=round(frame.time_s, 3),
            frame=frame.idx,
            signal=name,
            key=key,
            value=result.value,
            score=round(result.score, 4),
        )

    def _sort_key(self, obs: Observation) -> tuple:
        """The within-frame order: level, then signal, then key."""
        return (self._level_of[obs.signal], obs.signal, obs.key)
