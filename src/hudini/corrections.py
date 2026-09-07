"""Judges the observation log with correction rules and computes patches for the views.

A rule judges the whole log and never changes it. Each rule computes
:class:`~hudini.views.Patch` values, and the views apply the patches on
read. :func:`correct` applies the rules in order, and each rule sees the
patches of the rules before it. Series rules measure in seconds, never
in frame counts. This module does not import torch or paddleocr.
"""

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from msgspec.structs import replace

from hudini.catalog import Catalog
from hudini.layout import layout_complete
from hudini.schema import (
    ArmDigit,
    CorrectionSetting,
    FrameGeometry,
    Instrument,
    Observation,
    PedalColor,
    Press,
    Signal,
)
from hudini.views import (
    FrameState,
    LogView,
    Patch,
    PatchAction,
    apply_patches,
    payload,
)


class CorrectionSpec(Protocol):
    """One correction rule: its fields declare, its one method computes."""

    @property
    def name(self) -> str:
        """The rule's name, as patches and the header record it."""
        ...

    def patches(self, view: LogView, catalog: Catalog) -> Iterable[Patch]:
        """The patches this rule emits over the corrected-so-far log."""
        ...

    def settings(self) -> dict[str, CorrectionSetting]:
        """The rule's header entry: its name plus its declared fields."""
        ...


def correct(
    log: Iterable[Observation], specs: Iterable[CorrectionSpec], catalog: Catalog
) -> list[Patch]:
    """Run the correction pass: every spec, in the given order, as a fold.

    Each rule computes over the log with the earlier rules' patches
    applied, so the gates clear impossible presses before debounce judges
    what is left.
    """
    observations = list(log)
    patches: list[Patch] = []
    for spec in specs:
        view = LogView(observations=tuple(apply_patches(observations, patches)))
        patches.extend(spec.patches(view, catalog))
    return patches


def _run_end(times: list[float], stop: int) -> float:
    """When the run ending before index ``stop`` is over, half-open.

    The last run has no next sample, so it borrows the preceding spacing.
    """
    if stop < len(times):
        return times[stop]
    if len(times) >= 2:
        return times[-1] + (times[-1] - times[-2])
    return times[-1]


@dataclass(frozen=True, slots=True)
class Debounce:
    """Drop runs of a non-neutral value shorter than ``min_duration_s``.

    Runs are consecutive observations of one key whose payload ``field``
    holds an equal, non-neutral value. A suppressed run reads as neutral:
    the payload with ``field`` set to ``neutral``, or confirmed absence
    when ``neutral`` is None.
    """

    signal: Signal
    field: str
    min_duration_s: float
    neutral: Any = None

    @property
    def name(self) -> str:
        return f"debounce:{self.signal}"

    def settings(self) -> dict[str, CorrectionSetting]:
        return {"name": self.name, **asdict(self)}

    def patches(self, view: LogView, catalog: Catalog) -> Iterator[Patch]:
        for key, series in view.series(self.signal).items():
            times = [obs.time_s for obs in series]
            values = [
                getattr(obs.value, self.field) if obs.value is not None else None for obs in series
            ]
            start = 0
            while start < len(series):
                value = values[start]
                if value is None or value == self.neutral:
                    start += 1
                    continue
                stop = start + 1
                while stop < len(series) and values[stop] == value:
                    stop += 1
                end_s = _run_end(times, stop)
                if end_s - times[start] < self.min_duration_s:
                    source = series[start].value
                    if self.neutral is None or source is None:
                        patched = None
                    else:
                        patched = replace(source, **{self.field: self.neutral})
                    yield Patch(
                        signal=self.signal,
                        key=key,
                        start_s=times[start],
                        end_s=end_s,
                        action=PatchAction.SET,
                        by=self.name,
                        value=patched,
                    )
                start = stop


@dataclass(frozen=True, slots=True)
class FillGap:
    """Bridge short neutral gaps between equal neighbors.

    A gap is a run of observations whose payload ``field`` holds
    ``neutral`` (confirmed absence counts as neutral). It is bridged only
    when the values on both sides are equal by ``field`` and the gap is at
    most ``max_duration_s`` long, never at the start or end of a series.
    The gap reads as the value before it.
    """

    signal: Signal
    field: str
    max_duration_s: float
    neutral: Any = None

    @property
    def name(self) -> str:
        return f"fill:{self.signal}"

    def settings(self) -> dict[str, CorrectionSetting]:
        return {"name": self.name, **asdict(self)}

    def patches(self, view: LogView, catalog: Catalog) -> Iterator[Patch]:
        for key, series in view.series(self.signal).items():
            times = [obs.time_s for obs in series]
            values = [
                getattr(obs.value, self.field) if obs.value is not None else None for obs in series
            ]
            start = 0
            while start < len(series):
                if not (values[start] is None or values[start] == self.neutral):
                    start += 1
                    continue
                stop = start
                while stop < len(series) and (values[stop] is None or values[stop] == self.neutral):
                    stop += 1
                bracketed = 0 < start and stop < len(series)
                if bracketed and times[stop] - times[start] <= self.max_duration_s:
                    before, after = values[start - 1], values[stop]
                    if before is not None and before == after:
                        yield Patch(
                            signal=self.signal,
                            key=key,
                            start_s=times[start],
                            end_s=times[stop],
                            action=PatchAction.SET,
                            by=self.name,
                            value=series[start - 1].value,
                        )
                start = stop


@dataclass(frozen=True, slots=True)
class CompleteLayout:
    """Read an incomplete layout as no UI.

    A layout is complete when every pod has a box and one pod is the
    camera. Steel shafts and tissue edges in a video without UI make one
    or two pod boxes per frame, never a complete row. A run of incomplete
    layouts is kept when complete layouts sit on both sides of it and it
    is at most ``max_gap_s`` long: real UI drops a pod for a frame or two.
    """

    max_gap_s: float

    @property
    def name(self) -> str:
        return "complete_layout"

    def settings(self) -> dict[str, CorrectionSetting]:
        return {"name": self.name, **asdict(self)}

    def patches(self, view: LogView, catalog: Catalog) -> Iterator[Patch]:
        for key, series in view.series(Signal.LAYOUT).items():
            times = [obs.time_s for obs in series]
            complete = [
                isinstance(obs.value, FrameGeometry) and layout_complete(obs.value)
                for obs in series
            ]
            incomplete = [
                isinstance(obs.value, FrameGeometry) and not layout_complete(obs.value)
                for obs in series
            ]
            start = 0
            while start < len(series):
                if not incomplete[start]:
                    start += 1
                    continue
                stop = start
                while stop < len(series) and incomplete[stop]:
                    stop += 1
                bracketed = (
                    0 < start and stop < len(series) and complete[start - 1] and complete[stop]
                )
                if not (bracketed and times[stop] - times[start] <= self.max_gap_s):
                    yield Patch(
                        signal=Signal.LAYOUT,
                        key=key,
                        start_s=times[start],
                        end_s=_run_end(times, stop),
                        action=PatchAction.SET,
                        by=self.name,
                        value=None,
                    )
                start = stop


_RELEASED = Press(pressed=False)


def _fresh_presses(fs: FrameState) -> Iterator[Observation]:
    for obs in fs.fresh:
        if obs.signal is Signal.PEDALS and isinstance(obs.value, Press) and obs.value.pressed:
            yield obs


def _release(obs: Observation, by: str) -> Patch:
    return Patch(
        signal=Signal.PEDALS,
        key=obs.key,
        start_s=obs.time_s,
        end_s=obs.time_s,
        action=PatchAction.SET,
        by=by,
        value=_RELEASED,
    )


@dataclass(frozen=True, slots=True)
class PedalRequiresInstrument:
    """Drop presses on a pod with no instrument in state.

    The instrument carries forward, so no instrument means none has been
    seen yet, so there is nothing a pedal can actuate.
    """

    @property
    def name(self) -> str:
        return "pedal_requires_instrument"

    def settings(self) -> dict[str, CorrectionSetting]:
        return {"name": self.name, **asdict(self)}

    def patches(self, view: LogView, catalog: Catalog) -> Iterator[Patch]:
        for fs in view.frames():
            for obs in _fresh_presses(fs):
                column = obs.key[0]
                if (Signal.INSTRUMENT, column) not in fs.state:
                    yield _release(obs, by=self.name)


@dataclass(frozen=True, slots=True)
class PedalRequiresAction:
    """Drop presses on a pedal whose instrument declares no action for it.

    Gates only on knowledge: an unknown instrument or missing pedal data
    gives no opinion. Only a declared empty label set suppresses, and only
    the offending color.
    """

    @property
    def name(self) -> str:
        return "pedal_requires_action"

    def settings(self) -> dict[str, CorrectionSetting]:
        return {"name": self.name, **asdict(self)}

    def patches(self, view: LogView, catalog: Catalog) -> Iterator[Patch]:
        for fs in view.frames():
            for obs in _fresh_presses(fs):
                column, color = obs.key
                instrument = fs.state.get((Signal.INSTRUMENT, column))
                if instrument is None:
                    continue
                resolution = catalog.resolve(payload(instrument, Instrument).name)
                if resolution is None or resolution.entry.pedals is None:
                    continue
                if not resolution.entry.pedals.of(PedalColor(color)).labels:
                    yield _release(obs, by=self.name)


@dataclass(frozen=True, slots=True)
class ArmDigitUnique:
    """Keep an arm digit on at most one pod per frame.

    Arms are uniquely numbered, so two pods claiming one digit is
    impossible. The higher score keeps the digit; the losers read as
    confirmed absence and keep their scores.
    """

    @property
    def name(self) -> str:
        return "arm_digit_unique"

    def settings(self) -> dict[str, CorrectionSetting]:
        return {"name": self.name, **asdict(self)}

    def patches(self, view: LogView, catalog: Catalog) -> Iterator[Patch]:
        for fs in view.frames():
            by_digit: dict[int, list[Observation]] = {}
            for obs in fs.fresh:
                if obs.signal is Signal.ARM and isinstance(obs.value, ArmDigit):
                    by_digit.setdefault(obs.value.digit, []).append(obs)
            for claimants in by_digit.values():
                if len(claimants) <= 1:
                    continue
                claimants.sort(key=lambda obs: obs.score, reverse=True)
                for loser in claimants[1:]:
                    yield Patch(
                        signal=Signal.ARM,
                        key=loser.key,
                        start_s=loser.time_s,
                        end_s=loser.time_s,
                        action=PatchAction.SET,
                        by=self.name,
                        value=None,
                    )


_GATE_RULES = {
    "pedal_requires_instrument": PedalRequiresInstrument,
    "pedal_requires_action": PedalRequiresAction,
    "arm_digit_unique": ArmDigitUnique,
}


def _neutral(entry: dict[str, CorrectionSetting], key: str = "neutral") -> CorrectionSetting:
    value = entry[key]
    return tuple(str(item) for item in value) if isinstance(value, list | tuple) else value


def _seconds(entry: dict[str, CorrectionSetting], key: str) -> float:
    value = entry[key]
    if not isinstance(value, int | float):
        raise TypeError(f"correction setting {key!r} must be a number, but it is {value!r}")
    return float(value)


def rules_from_settings(
    settings: Iterable[dict[str, CorrectionSetting]],
) -> tuple[CorrectionSpec, ...]:
    """Rebuild correction rules from their header entries.

    The inverse of ``settings()``: a decoded header's ``corrections``
    list becomes the rule objects that produce it. So a report can apply
    exactly the rules of the run that wrote the log.

    Raises:
        ValueError: an entry names a rule this package does not know.
        TypeError: a seconds setting is not a number.
    """
    rules: list[CorrectionSpec] = []
    for entry in settings:
        name = str(entry["name"])
        kind = name.partition(":")[0]
        if kind == "debounce":
            rules.append(
                Debounce(
                    signal=Signal(str(entry["signal"])),
                    field=str(entry["field"]),
                    min_duration_s=_seconds(entry, "min_duration_s"),
                    neutral=_neutral(entry),
                )
            )
        elif kind == "fill":
            rules.append(
                FillGap(
                    signal=Signal(str(entry["signal"])),
                    field=str(entry["field"]),
                    max_duration_s=_seconds(entry, "max_duration_s"),
                    neutral=_neutral(entry),
                )
            )
        elif kind == "complete_layout":
            rules.append(CompleteLayout(max_gap_s=_seconds(entry, "max_gap_s")))
        elif kind in _GATE_RULES:
            rules.append(_GATE_RULES[kind]())
        else:
            valid = sorted(["debounce", "fill", "complete_layout", *_GATE_RULES])
            raise ValueError(f"unknown correction rule {name!r}; valid kinds: {valid}")
    return tuple(rules)
