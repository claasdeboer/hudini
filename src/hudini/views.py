"""Derives views from the observation log without changing it.

Every function here derives from observations and never changes them.
The raw lenses are :func:`snapshot` and :func:`iter_frame_states`. The
interval view is :func:`interval_view` over :func:`lanes` and
:func:`encode_runs`, and :func:`summarize` folds intervals into the case
summary. The corrections feed :class:`Patch` values through
:func:`apply_patches` and :class:`LogView`. This module does not import
torch or paddleocr.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import msgspec
from msgspec.structs import replace

from hudini.catalog import ResolvedAlias
from hudini.schema import (
    Appendix,
    ArmDigit,
    Banner,
    Box,
    Footer,
    FrameGeometry,
    Instrument,
    Laser,
    Observation,
    OffscreenBars,
    OffscreenBarStatus,
    PedalColor,
    PedalLabel,
    PodGeometry,
    PodStatus,
    PopupStack,
    Press,
    Role,
    Signal,
    Status,
    ToolBadges,
    Value,
)


def payload[P: msgspec.Struct](obs: Observation, kind: type[P]) -> P:
    """The observation's value as its known payload type.

    Raises:
        TypeError: the value is not of that type, a mis-keyed observation.
    """
    if not isinstance(obs.value, kind):
        raise TypeError(
            f"{obs.signal} observation holds {type(obs.value).__name__}, not {kind.__name__}"
        )
    return obs.value


class LaneName(StrEnum):
    """The output lanes: what the intervals and the timeline track."""

    STATUS = "status"
    INSTRUMENT = "instrument"
    PEDAL_YELLOW = "pedal_yellow"
    PEDAL_BLUE = "pedal_blue"
    LASER = "laser"
    POPUP = "popup"
    BANNER = "banner"
    OFFSCREEN = "offscreen"
    TOOL_ASSOCIATION = "tool_association"
    ROLE = "role"
    NO_UI = "no_ui"


@dataclass(frozen=True, slots=True)
class Lane:
    """One strip of the output: one kind of fact about one identity.

    Attributes:
        arm: the arm identity the strip tracks, for pod-scoped lanes.
        column: the screen column the pod sits in, for pod-scoped lanes.
        key: identity extension, e.g. ``(text,)`` for a popup message.
    """

    name: LaneName
    arm: int | None = None
    column: int | None = None
    key: tuple = ()


@dataclass(frozen=True, slots=True)
class Interval:
    """One temporal run of one lane. ``[start_s, end_s)``, half-open.

    Attributes:
        value: what the lane showed, or None when no value ever resolved.
    """

    lane: LaneName
    arm: int | None
    column: int | None
    start_s: float
    end_s: float
    value: str | None


class PatchAction(StrEnum):
    """What a patch does to the observations it covers."""

    DELETE = "delete"
    SET = "set"


@dataclass(frozen=True, slots=True)
class Patch:
    """A correction's rewrite of what a sensor "would have said".

    ``DELETE`` skips covered observations, as if the sensor did not read.
    ``SET`` replaces their value: None reads as confirmed absence, a
    payload reads as that value. A patch never changes an observation's
    time or score.

    Attributes:
        start_s: span start, inclusive.
        end_s: span end, exclusive. Equal to ``start_s`` for a point
            patch, which covers exactly that time.
        by: the rule that fired, e.g. ``"debounce:status"``.
        value: SET only.

    Raises:
        ValueError: a DELETE patch carries a value.
    """

    signal: Signal
    key: tuple
    start_s: float
    end_s: float
    action: PatchAction
    by: str
    value: Value | None = None

    def __post_init__(self) -> None:
        if self.action is PatchAction.DELETE and self.value is not None:
            raise ValueError("a DELETE patch cannot carry a value")

    def covers(self, time_s: float) -> bool:
        """Whether an observation at ``time_s`` falls inside this patch."""
        if self.start_s == self.end_s:
            return time_s == self.start_s
        return self.start_s <= time_s < self.end_s


class PatchSweep:
    """The patches of one (signal, key), queried in time order.

    Call ``covering`` with times that do not decrease. A patch becomes
    active when the query time reaches its start. It retires when it can
    cover no later time. So one pass over a log touches each patch once.
    """

    def __init__(self, patches: list[tuple[int, Patch]]) -> None:
        self._pending = sorted(patches, key=lambda entry: entry[1].start_s)
        self._next = 0
        self._active: dict[int, Patch] = {}

    def covering(self, time_s: float) -> list[Patch]:
        """The patches covering ``time_s``, in application order."""
        pending = self._pending
        while self._next < len(pending) and pending[self._next][1].start_s <= time_s:
            index, patch = pending[self._next]
            self._active[index] = patch
            self._next += 1
        expired = [index for index, patch in self._active.items() if not patch.covers(time_s)]
        for index in expired:
            del self._active[index]
        return [patch for _index, patch in sorted(self._active.items())]


def apply_patches(log: Iterable[Observation], patches: Iterable[Patch]) -> list[Observation]:
    """The corrected observations: ``patches`` applied in order.

    Later patches see the effects of earlier patches, the order of the
    correction pass. ``log`` must be time-sorted.
    """
    grouped: dict[tuple, list[tuple[int, Patch]]] = {}
    for index, patch in enumerate(patches):
        grouped.setdefault((patch.signal, patch.key), []).append((index, patch))
    sweeps = {entry: PatchSweep(group) for entry, group in grouped.items()}
    corrected: list[Observation] = []
    for obs in log:
        sweep = sweeps.get((obs.signal, obs.key))
        keep = True
        for patch in sweep.covering(obs.time_s) if sweep is not None else ():
            if patch.action is PatchAction.DELETE:
                keep = False
                break
            obs = replace(obs, value=patch.value)
        if keep:
            corrected.append(obs)
    return corrected


def snapshot(log: Iterable[Observation], at_s: float) -> dict[tuple, Observation]:
    """The state of every key at one time: carry-forward, as a function.

    A valued observation updates its key, a None value clears it, and keys
    never observed are absent. ``log`` must be time-sorted.
    """
    state: dict[tuple, Observation] = {}
    for obs in log:
        if obs.time_s > at_s:
            break
        entry = (obs.signal, *obs.key)
        if obs.value is None:
            state.pop(entry, None)
        else:
            state[entry] = obs
    return state


@dataclass(frozen=True, slots=True)
class FrameState:
    """One sampled frame, through the frame lens.

    Attributes:
        fresh: the observations of this frame, what happened.
        state: the snapshot at this time, this frame included, what is
            true. A per-frame copy, so a change affects nothing else.
    """

    time_s: float
    frame: int
    fresh: tuple[Observation, ...]
    state: dict[tuple, Observation]

    @property
    def geometry(self) -> FrameGeometry | None:
        """The frame's geometry from its state, or None when the UI was
        not parsed."""
        held = self.state.get((Signal.LAYOUT,))
        if held is None or not isinstance(held.value, FrameGeometry):
            return None
        return held.value


def iter_frame_states(observations: Iterable[Observation]) -> Iterator[FrameState]:
    """One :class:`FrameState` per sampled frame, in time order.

    ``observations`` must be time-sorted; frames are delimited by the
    ``frame`` field changing.
    """
    state: dict[tuple, Observation] = {}
    fresh: list[Observation] = []
    current: tuple[int, float] | None = None
    for obs in observations:
        marker = (obs.frame, obs.time_s)
        if current is not None and marker != current:
            yield FrameState(
                time_s=current[1], frame=current[0], fresh=tuple(fresh), state=dict(state)
            )
            fresh = []
        current = marker
        fresh.append(obs)
        entry = (obs.signal, *obs.key)
        if obs.value is None:
            state.pop(entry, None)
        else:
            state[entry] = obs
    if current is not None:
        yield FrameState(time_s=current[1], frame=current[0], fresh=tuple(fresh), state=dict(state))


@dataclass(frozen=True, slots=True)
class LogView:
    """The correction rules' read-only view of the corrected-so-far log."""

    observations: tuple[Observation, ...]

    def series(self, signal: Signal) -> dict[tuple, tuple[Observation, ...]]:
        """Each key's observations of one signal, through time."""
        grouped: dict[tuple, list[Observation]] = {}
        for obs in self.observations:
            if obs.signal is signal:
                grouped.setdefault(obs.key, []).append(obs)
        return {key: tuple(entries) for key, entries in grouped.items()}

    def frames(self) -> Iterator[FrameState]:
        """One :class:`FrameState` per frame."""
        return iter_frame_states(self.observations)


class ActionCatalog(Protocol):
    """What the lane projection needs from the catalog."""

    def pedal_action(self, *, instrument: str, color: PedalColor, label: str | None) -> str | None:
        """The action a pressed pedal delivers, or None when unknown."""
        ...


Projector = Callable[
    [FrameState, Mapping[int, int], ActionCatalog], Iterable[tuple[Lane, str | None]]
]

_PEDAL_LANES = {PedalColor.YELLOW: LaneName.PEDAL_YELLOW, PedalColor.BLUE: LaneName.PEDAL_BLUE}


def _state_entries(fs: FrameState, signal: Signal) -> Iterator[tuple[tuple, Observation]]:
    for entry, obs in fs.state.items():
        if entry[0] is signal:
            yield entry[1:], obs


def _arm_by_column(fs: FrameState) -> dict[int, int]:
    return {key[0]: payload(obs, ArmDigit).digit for key, obs in _state_entries(fs, Signal.ARM)}


def _project_layout(
    fs: FrameState, arms: Mapping[int, int], catalog: ActionCatalog
) -> Iterator[tuple[Lane, str | None]]:
    geometry = fs.geometry
    if geometry is None:
        return
    for pod in geometry.pods:
        if pod.role is Role.CAMERA:
            column = pod.column
            lane = Lane(name=LaneName.ROLE, arm=arms.get(column, column), column=column)
            yield lane, Role.CAMERA


def _project_status(
    fs: FrameState, arms: Mapping[int, int], catalog: ActionCatalog
) -> Iterator[tuple[Lane, str | None]]:
    for key, obs in _state_entries(fs, Signal.STATUS):
        column = key[0]
        lane = Lane(name=LaneName.STATUS, arm=arms.get(column, column), column=column)
        yield lane, payload(obs, PodStatus).status


def _project_instrument(
    fs: FrameState, arms: Mapping[int, int], catalog: ActionCatalog
) -> Iterator[tuple[Lane, str | None]]:
    for key, obs in _state_entries(fs, Signal.INSTRUMENT):
        column = key[0]
        lane = Lane(name=LaneName.INSTRUMENT, arm=arms.get(column, column), column=column)
        yield lane, payload(obs, Instrument).name


def _project_pedals(
    fs: FrameState, arms: Mapping[int, int], catalog: ActionCatalog
) -> Iterator[tuple[Lane, str | None]]:
    for key, obs in _state_entries(fs, Signal.PEDALS):
        if not payload(obs, Press).pressed:
            continue
        column, color = key
        instrument = fs.state.get((Signal.INSTRUMENT, column))
        label = fs.state.get((Signal.PEDAL_LABEL, column, color))
        action = None
        if instrument is not None:
            action = catalog.pedal_action(
                instrument=payload(instrument, Instrument).name,
                color=color,
                label=payload(label, PedalLabel).text if label is not None else None,
            )
        lane = Lane(name=_PEDAL_LANES[color], arm=arms.get(column, column), column=column)
        yield lane, action


def _project_laser(
    fs: FrameState, arms: Mapping[int, int], catalog: ActionCatalog
) -> Iterator[tuple[Lane, str | None]]:
    for key, obs in _state_entries(fs, Signal.LASER):
        if not payload(obs, Laser).on:
            continue
        column = key[0]
        yield Lane(name=LaneName.LASER, arm=arms.get(column, column), column=column), None


def _project_popups(
    fs: FrameState, arms: Mapping[int, int], catalog: ActionCatalog
) -> Iterator[tuple[Lane, str | None]]:
    for key, obs in _state_entries(fs, Signal.POPUPS):
        column = key[0]
        for message in payload(obs, PopupStack).messages:
            lane = Lane(
                name=LaneName.POPUP,
                arm=arms.get(column, column),
                column=column,
                key=(message.text,),
            )
            yield lane, message.text


def _project_banner(
    fs: FrameState, arms: Mapping[int, int], catalog: ActionCatalog
) -> Iterator[tuple[Lane, str | None]]:
    for _key, obs in _state_entries(fs, Signal.BANNER):
        yield Lane(name=LaneName.BANNER), payload(obs, Banner).text


def _project_offscreen(
    fs: FrameState, arms: Mapping[int, int], catalog: ActionCatalog
) -> Iterator[tuple[Lane, str | None]]:
    for _key, obs in _state_entries(fs, Signal.OFFSCREEN):
        for bar in payload(obs, OffscreenBars).bars:
            if bar.arm is not None:
                yield Lane(name=LaneName.OFFSCREEN, arm=bar.arm), bar.status


def _project_tool_association(
    fs: FrameState, arms: Mapping[int, int], catalog: ActionCatalog
) -> Iterator[tuple[Lane, str | None]]:
    for _key, obs in _state_entries(fs, Signal.TOOL_ASSOCIATION):
        for badge in payload(obs, ToolBadges).badges:
            if badge.arm is not None:
                yield Lane(name=LaneName.TOOL_ASSOCIATION, arm=badge.arm), None


# One projector per signal; None means the signal paints no lanes of its
# own. The registry checks this table is complete.
LANE_PROJECTORS: dict[Signal, Projector | None] = {
    Signal.LAYOUT: _project_layout,
    Signal.STATUS: _project_status,
    Signal.ARM: None,
    Signal.INSTRUMENT: _project_instrument,
    Signal.PEDALS: _project_pedals,
    Signal.PEDAL_LABEL: None,
    Signal.LASER: _project_laser,
    Signal.POPUPS: _project_popups,
    Signal.BANNER: _project_banner,
    Signal.OFFSCREEN: _project_offscreen,
    Signal.TOOL_ASSOCIATION: _project_tool_association,
}


def lanes(fs: FrameState, catalog: ActionCatalog) -> dict[Lane, str | None]:
    """The complete lane snapshot of one frame: what a timeline cursor shows.

    A frame without geometry (the UI was not parsed) paints only the
    ``NO_UI`` lane. A None value is a lane whose value has not resolved
    yet.
    """
    if (Signal.LAYOUT,) not in fs.state:
        return {Lane(name=LaneName.NO_UI): None}
    arms = _arm_by_column(fs)
    painted: dict[Lane, str | None] = {}
    for projector in LANE_PROJECTORS.values():
        if projector is not None:
            painted.update(projector(fs, arms, catalog))
    return painted


def encode_runs(frames: Iterable[tuple[float, dict[Lane, str | None]]]) -> list[Interval]:
    """Run-length encoding over lane snapshots, domain-free.

    A new lane opens an interval, a missing lane closes it, and a change
    between two real values splits. A None value is pending: the first
    real value fills the run backwards, and None itself never splits.
    Runs are half-open; the last open runs close one sample spacing past
    the final frame.
    """
    intervals: list[Interval] = []
    open_runs: dict[Lane, dict] = {}

    def close(lane: Lane, run: dict, end_s: float) -> None:
        intervals.append(
            Interval(
                lane=lane.name,
                arm=lane.arm,
                column=lane.column,
                start_s=run["start_s"],
                end_s=end_s,
                value=run["value"],
            )
        )

    previous_s: float | None = None
    spacing = 0.0
    for time_s, snap in frames:
        if previous_s is not None:
            spacing = time_s - previous_s
        for lane in [lane for lane in open_runs if lane not in snap]:
            close(lane, open_runs.pop(lane), end_s=time_s)
        for lane, value in snap.items():
            run = open_runs.get(lane)
            if run is None:
                open_runs[lane] = {"start_s": time_s, "value": value}
            elif value is None or value == run["value"]:
                continue
            elif run["value"] is None:
                run["value"] = value
            else:
                close(lane, run, end_s=time_s)
                open_runs[lane] = {"start_s": time_s, "value": value}
        previous_s = time_s
    if previous_s is not None:
        for lane, run in open_runs.items():
            close(lane, run, end_s=previous_s + spacing)
    intervals.sort(key=lambda entry: (entry.start_s, entry.lane, entry.arm or 0, entry.column or 0))
    return intervals


def interval_view(
    log: Iterable[Observation], patches: Iterable[Patch], catalog: ActionCatalog
) -> list[Interval]:
    """The intervals: one entry per temporal run of one lane."""
    corrected = apply_patches(log, patches)
    frames = ((fs.time_s, lanes(fs, catalog)) for fs in iter_frame_states(corrected))
    return encode_runs(frames)


SUMMARY_VERSION = 1
"""Version of the :class:`Summary` shape a log footer embeds."""

INTERVALS_VERSION = 1
"""Version of the interval shape and lane vocabulary a log appendix
embeds."""


def intervals_to_wire(intervals: list[Interval]) -> list[dict]:
    """The intervals as the plain JSON objects an appendix carries.

    The shape is exactly what a JSON round trip gives, so an appendix
    built in memory equals the same appendix read back from disk.
    """
    return msgspec.json.decode(msgspec.json.encode(intervals))


def intervals_from_appendix(appendix: Appendix | None) -> list[Interval] | None:
    """The appendix's intervals, when present at the current version.

    Returns None for a missing appendix or one of another version, so a
    stale cache degrades to a live derivation, never to an error.
    """
    if appendix is None or appendix.intervals_version != INTERVALS_VERSION:
        return None
    return msgspec.convert(appendix.intervals, type=list[Interval], strict=False)


UNKNOWN_INSTRUMENT_TYPE = "unknown"
"""The class of an instrument name the catalog does not know."""


class SummaryCatalog(ActionCatalog, Protocol):
    """What the summary fold needs from the catalog."""

    def resolve(self, alias: str) -> ResolvedAlias | None:
        """The facts of an exact alias, or None for an unknown one."""
        ...


class InstrumentUse(msgspec.Struct, frozen=True):
    """One instrument's use in the case.

    Attributes:
        type: the catalog class, or ``unknown`` for a name the catalog
            does not know.
        duration_s: total mounted time in seconds.
        arms: the arms that carried it, sorted.
    """

    name: str
    type: str
    duration_s: float
    arms: tuple[int, ...]


class RunTotals(msgspec.Struct, frozen=True):
    """Run count and total duration for one on-off lane.

    Counts count runs, never detections, so the numbers do not change
    with the sampling rate.
    """

    count: int
    duration_s: float


class OffscreenRuns(msgspec.Struct, frozen=True):
    """Off-screen indicator runs for one arm and one bar status.

    Attributes:
        arm: None when the arm digit was not resolved.
        status: what the bar showed, or None when the detector could not
            tell.
    """

    arm: int | None
    status: OffscreenBarStatus | None
    count: int
    duration_s: float


class ToolAssociationRuns(msgspec.Struct, frozen=True):
    """Association-badge runs for one arm.

    Attributes:
        arm: None when the arm digit was not resolved.
    """

    arm: int | None
    count: int
    duration_s: float


class Summary(msgspec.Struct, frozen=True):
    """What happened in the case, folded from the corrected intervals.

    Case facts only: run facts such as filename, duration, and fps live
    in the header and the footer beside the summary. Durations are
    seconds, rounded to milliseconds. Every tuple has a deterministic
    order, so the same log gives the same summary bytes.

    Attributes:
        actions_by_pedal: press run counts per action, split by pedal
            color. Empty for a summary written before the field existed.
    """

    instruments: tuple[InstrumentUse, ...]
    instrument_changes: int
    presses: dict[PedalColor, RunTotals]
    presses_by_action: dict[str, int]
    laser: RunTotals
    offscreen: tuple[OffscreenRuns, ...]
    tool_association: tuple[ToolAssociationRuns, ...]
    popup_count: int
    popup_texts: tuple[str, ...]
    banner_duration_s: float
    banner_texts: tuple[str, ...]
    warning_duration_s: float
    no_ui_duration_s: float
    actions_by_pedal: dict[PedalColor, dict[str, int]] = {}

    def to_wire(self) -> dict:
        """The summary as the plain JSON object a footer embeds.

        The shape is exactly what a JSON round trip gives, so a footer
        built in memory equals the same footer read back from disk.
        """
        return msgspec.json.decode(msgspec.json.encode(self))

    @classmethod
    def from_footer(cls, footer: Footer) -> "Summary | None":
        """The footer's summary, when present at the current version.

        Returns None for a footer without a summary or with a summary
        of another version. The value is never converted to the typed
        shape in that case, so an old shape cannot fail the read.
        """
        if footer.summary is None or footer.summary_version != SUMMARY_VERSION:
            return None
        return msgspec.convert(footer.summary, type=cls, strict=False)


def summarize(intervals: Iterable[Interval], catalog: SummaryCatalog) -> Summary:
    """Fold intervals into the case summary.

    Instruments are listed by first appearance. An instrument run whose
    name never resolved is not attributed. ``instrument_changes`` counts
    a mount that follows a different mount on the same arm.
    """
    instruments: dict[str, dict] = {}
    changes = 0
    last_mount: dict[int | None, str] = {}
    press_runs: dict[PedalColor, list[float]] = {color: [0, 0.0] for color in PedalColor}
    actions: dict[str, int] = {}
    pedal_actions: dict[PedalColor, dict[str, int]] = {color: {} for color in PedalColor}
    laser_runs = [0, 0.0]
    offscreen_runs: dict[tuple[int | None, OffscreenBarStatus | None], list[float]] = {}
    association_runs: dict[int | None, list[float]] = {}
    popup_count = 0
    popup_texts: set[str] = set()
    banner_duration = 0.0
    banner_texts: set[str] = set()
    warning_duration = 0.0
    no_ui_duration = 0.0

    for entry in sorted(intervals, key=lambda run: run.start_s):
        span = entry.end_s - entry.start_s
        if entry.lane is LaneName.INSTRUMENT and entry.value is not None:
            use = instruments.setdefault(entry.value, {"duration": 0.0, "arms": set()})
            use["duration"] += span
            if entry.arm is not None:
                use["arms"].add(entry.arm)
            if last_mount.get(entry.arm) not in (None, entry.value):
                changes += 1
            last_mount[entry.arm] = entry.value
        elif entry.lane in (LaneName.PEDAL_YELLOW, LaneName.PEDAL_BLUE):
            color = PedalColor.YELLOW if entry.lane is LaneName.PEDAL_YELLOW else PedalColor.BLUE
            press_runs[color][0] += 1
            press_runs[color][1] += span
            if entry.value is not None:
                actions[entry.value] = actions.get(entry.value, 0) + 1
                counts = pedal_actions[color]
                counts[entry.value] = counts.get(entry.value, 0) + 1
        elif entry.lane is LaneName.LASER:
            laser_runs[0] += 1
            laser_runs[1] += span
        elif entry.lane is LaneName.OFFSCREEN:
            status = None if entry.value is None else OffscreenBarStatus(entry.value)
            totals = offscreen_runs.setdefault((entry.arm, status), [0, 0.0])
            totals[0] += 1
            totals[1] += span
        elif entry.lane is LaneName.TOOL_ASSOCIATION:
            totals = association_runs.setdefault(entry.arm, [0, 0.0])
            totals[0] += 1
            totals[1] += span
        elif entry.lane is LaneName.POPUP:
            popup_count += 1
            if entry.value is not None:
                popup_texts.add(entry.value)
        elif entry.lane is LaneName.BANNER:
            banner_duration += span
            if entry.value is not None:
                banner_texts.add(entry.value)
        elif entry.lane is LaneName.STATUS and entry.value == Status.WARNING:
            warning_duration += span
        elif entry.lane is LaneName.NO_UI:
            no_ui_duration += span

    def instrument_type(name: str) -> str:
        resolved = catalog.resolve(name)
        return resolved.entry.type if resolved is not None else UNKNOWN_INSTRUMENT_TYPE

    def row_order(key: tuple[int | OffscreenBarStatus | None, ...]) -> tuple:
        return tuple((part is None, part if part is not None else "") for part in key)

    return Summary(
        instruments=tuple(
            InstrumentUse(
                name=name,
                type=instrument_type(name),
                duration_s=round(use["duration"], 3),
                arms=tuple(sorted(use["arms"])),
            )
            for name, use in instruments.items()
        ),
        instrument_changes=changes,
        presses={
            color: RunTotals(count=int(count), duration_s=round(duration, 3))
            for color, (count, duration) in press_runs.items()
        },
        presses_by_action={action: actions[action] for action in sorted(actions)},
        laser=RunTotals(count=int(laser_runs[0]), duration_s=round(laser_runs[1], 3)),
        offscreen=tuple(
            OffscreenRuns(arm=arm, status=status, count=int(count), duration_s=round(duration, 3))
            for (arm, status), (count, duration) in sorted(
                offscreen_runs.items(), key=lambda item: row_order(item[0])
            )
        ),
        tool_association=tuple(
            ToolAssociationRuns(arm=arm, count=int(count), duration_s=round(duration, 3))
            for arm, (count, duration) in sorted(
                association_runs.items(), key=lambda item: row_order((item[0],))
            )
        ),
        popup_count=popup_count,
        popup_texts=tuple(sorted(popup_texts)),
        banner_duration_s=round(banner_duration, 3),
        banner_texts=tuple(sorted(banner_texts)),
        warning_duration_s=round(warning_duration, 3),
        no_ui_duration_s=round(no_ui_duration, 3),
        actions_by_pedal={
            color: {action: counts[action] for action in sorted(counts)}
            for color, counts in pedal_actions.items()
        },
    )


def summary_view(
    log: Iterable[Observation], patches: Iterable[Patch], catalog: SummaryCatalog
) -> Summary:
    """The summary: one struct of case facts per video."""
    return summarize(interval_view(log, patches, catalog), catalog)


RecordValue = str | int | float | bool | None


def _shaped(obs: Observation, at_s: float, value: RecordValue) -> dict:
    """The uniform record entry: value, rounded score, and
    ``observed_at_s`` when the value came from an earlier frame."""
    entry: dict = {"value": value, "score": round(obs.score, 4)}
    if obs.time_s < at_s:
        entry["observed_at_s"] = round(obs.time_s, 3)
    return entry


def _box_record(box: Box) -> dict:
    return {"x": box.x, "y": box.y, "w": box.w, "h": box.h}


def _pedal_action_of(
    fs: FrameState,
    column: int,
    color: PedalColor,
    instrument_name: str | None,
    catalog: ActionCatalog,
) -> str | None:
    if instrument_name is None:
        return None
    label_obs = fs.state.get((Signal.PEDAL_LABEL, column, color))
    label = payload(label_obs, PedalLabel).text if label_obs is not None else None
    return catalog.pedal_action(instrument=instrument_name, color=color, label=label)


def _pedal_entries(
    fs: FrameState, column: int, instrument_name: str | None, catalog: ActionCatalog
) -> dict:
    entries: dict = {}
    for color in (PedalColor.YELLOW, PedalColor.BLUE):
        press = fs.state.get((Signal.PEDALS, column, color))
        if press is None:
            continue
        entry = _shaped(press, fs.time_s, payload(press, Press).pressed)
        entry["action"] = _pedal_action_of(fs, column, color, instrument_name, catalog)
        entries[color] = entry
    return entries


def _popup_entries(fs: FrameState, column: int) -> list[dict] | None:
    stack_obs = fs.state.get((Signal.POPUPS, column))
    if stack_obs is None:
        return None
    return [
        {**_shaped(stack_obs, fs.time_s, message.text), "box": _box_record(message.box)}
        for message in payload(stack_obs, PopupStack).messages
    ]


def _pod_record(fs: FrameState, pod: PodGeometry, catalog: ActionCatalog) -> dict:
    record: dict = {"column": pod.column, "role": pod.role}
    if pod.box is not None:
        record["box"] = _box_record(pod.box)
    status = fs.state.get((Signal.STATUS, pod.column))
    if status is not None:
        record["status"] = _shaped(status, fs.time_s, payload(status, PodStatus).status)
    arm = fs.state.get((Signal.ARM, pod.column))
    if arm is not None:
        record["arm"] = _shaped(arm, fs.time_s, payload(arm, ArmDigit).digit)
    laser = fs.state.get((Signal.LASER, pod.column))
    if laser is not None:
        record["laser"] = _shaped(laser, fs.time_s, payload(laser, Laser).on)
    instrument_name = None
    instrument_obs = fs.state.get((Signal.INSTRUMENT, pod.column))
    if instrument_obs is not None:
        instrument = payload(instrument_obs, Instrument)
        instrument_name = instrument.name
        entry = _shaped(instrument_obs, fs.time_s, instrument.name)
        entry["ocr"] = {"raw": instrument.raw, "window": instrument.window}
        if instrument.reload_color is not None:
            entry["reload_color"] = instrument.reload_color
        record["instrument"] = entry
    pedals = _pedal_entries(fs, pod.column, instrument_name, catalog)
    if pedals:
        record["pedals"] = pedals
    popups = _popup_entries(fs, pod.column)
    if popups is not None:
        record["popups"] = popups
    return record


def _banner_record(fs: FrameState) -> dict | None:
    geometry = fs.geometry
    box = geometry.banner if geometry is not None else None
    obs = fs.state.get((Signal.BANNER,))
    if box is None and obs is None:
        return None
    record: dict = {}
    if box is not None:
        record["box"] = _box_record(box)
    if obs is not None:
        record.update(_shaped(obs, fs.time_s, payload(obs, Banner).text))
    return record


def _offscreen_record(fs: FrameState) -> dict | None:
    obs = fs.state.get((Signal.OFFSCREEN,))
    if obs is None:
        return None
    detections = [
        {
            "status": bar.status,
            "arm": bar.arm,
            "score": round(bar.score, 4),
            "box": _box_record(bar.box),
        }
        for bar in payload(obs, OffscreenBars).bars
    ]
    record: dict = {"detections": detections}
    if obs.time_s < fs.time_s:
        record["observed_at_s"] = round(obs.time_s, 3)
    return record


def _tool_association_record(fs: FrameState) -> dict | None:
    obs = fs.state.get((Signal.TOOL_ASSOCIATION,))
    if obs is None:
        return None
    detections = [
        {"arm": badge.arm, "score": round(badge.score, 4), "box": _box_record(badge.box)}
        for badge in payload(obs, ToolBadges).badges
    ]
    record: dict = {"detections": detections}
    if obs.time_s < fs.time_s:
        record["observed_at_s"] = round(obs.time_s, 3)
    return record


def _frame_record(fs: FrameState, catalog: ActionCatalog) -> dict:
    record: dict = {"frame_index": fs.frame, "time_s": round(fs.time_s, 3), "pods": []}
    geometry = fs.geometry
    if geometry is not None:
        record["pods"] = [_pod_record(fs, pod, catalog) for pod in geometry.pods]
    banner = _banner_record(fs)
    if banner is not None:
        record["banner"] = banner
    offscreen = _offscreen_record(fs)
    if offscreen is not None:
        record["offscreen"] = offscreen
    badges = _tool_association_record(fs)
    if badges is not None:
        record["tool_association"] = badges
    return record


def frame_view(
    log: Iterable[Observation], patches: Iterable[Patch], catalog: ActionCatalog
) -> list[dict]:
    """Per-frame records: what hudini believed at each sampled frame.

    One dict per sampled frame, for evaluation against frame-keyed
    ground truth. Each record holds ``frame_index``, ``time_s``, and
    ``pods``. It also holds ``banner``, ``offscreen``, and
    ``tool_association`` when they are known. A pod entry holds
    ``column``, ``role``, ``box``, and one ``{"value", "score"}`` entry
    per signal: ``status``, ``arm``, ``instrument`` (with ``ocr`` and
    ``reload_color``), ``pedals`` (with the catalog-derived ``action``),
    and ``popups``. An entry carries ``observed_at_s`` exactly when its
    value came from an earlier frame. Fields appear only when they
    apply. A frame without parsed UI has ``"pods": []``. Scores round to
    four decimals and times to milliseconds. This is the one place where
    presentation rounding occurs.
    """
    corrected = apply_patches(log, patches)
    return [_frame_record(fs, catalog) for fs in iter_frame_states(corrected)]
