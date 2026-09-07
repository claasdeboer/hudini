"""Unit tests for the correction rules in hudini.corrections."""

import json
from collections.abc import Callable

import pytest

from hudini.catalog import Catalog, _CatalogFile, _RawEntry, _RawPedals
from hudini.corrections import (
    ArmDigitUnique,
    CompleteLayout,
    Debounce,
    FillGap,
    PedalRequiresAction,
    PedalRequiresInstrument,
    correct,
    rules_from_settings,
)
from hudini.schema import (
    ArmDigit,
    Box,
    FrameGeometry,
    Instrument,
    Observation,
    OCRWindow,
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
from hudini.views import LogView, apply_patches

BOX = Box(x=0, y=0, w=10, h=5)


@pytest.fixture
def make_observation() -> Callable[..., Observation]:
    """Build an observation with overridable fields."""

    def _make(
        signal: Signal,
        key: tuple,
        value: object,
        time_s: float,
        frame: int | None = None,
        score: float = 1.0,
    ) -> Observation:
        return Observation(
            time_s=time_s,
            frame=frame if frame is not None else round(time_s * 30),
            signal=signal,
            key=key,
            value=value,
            score=score,
        )

    return _make


@pytest.fixture
def catalog() -> Catalog:
    """A small catalog: one yellow-less instrument, one without pedal data."""
    entries = (
        _RawEntry(
            name="maryland bipolar forceps",
            type="bipolar_grasper",
            pedals=_RawPedals(yellow=(), blue=("bipolar",)),
        ),
        _RawEntry(name="mystery instrument", type="grasper"),
    )
    return Catalog(_CatalogFile(color_locales={"en": {}}, entries=entries))


def _view(observations: list[Observation]) -> LogView:
    return LogView(observations=tuple(observations))


def _instrument(name: str) -> Instrument:
    return Instrument(name=name, match=95.0, raw=name, window=OCRWindow.NARROW)


class TestDebounce:
    def test_a_short_flicker_reads_as_neutral(self, make_observation, catalog):
        rule = Debounce(Signal.STATUS, field="status", min_duration_s=0.5, neutral=Status.INACTIVE)
        log = [
            make_observation(Signal.STATUS, (2,), PodStatus(status=Status.INACTIVE), 1.0),
            make_observation(Signal.STATUS, (2,), PodStatus(status=Status.ACTIVE), 1.2),
            make_observation(Signal.STATUS, (2,), PodStatus(status=Status.INACTIVE), 1.4),
        ]
        corrected = apply_patches(log, rule.patches(_view(log), catalog))
        assert corrected[1].value == PodStatus(status=Status.INACTIVE)

    def test_a_run_at_min_duration_s_survives(self, make_observation, catalog):
        # Dyadic times, so the duration is exact and the boundary is real.
        rule = Debounce(Signal.STATUS, field="status", min_duration_s=0.25, neutral=Status.INACTIVE)
        log = [
            make_observation(Signal.STATUS, (2,), PodStatus(status=Status.ACTIVE), 1.0),
            make_observation(Signal.STATUS, (2,), PodStatus(status=Status.INACTIVE), 1.25),
        ]
        assert list(rule.patches(_view(log), catalog)) == []

    def test_neutral_runs_are_left_alone(self, make_observation, catalog):
        rule = Debounce(Signal.STATUS, field="status", min_duration_s=9.0, neutral=Status.INACTIVE)
        log = [
            make_observation(Signal.STATUS, (2,), PodStatus(status=Status.INACTIVE), 1.0),
            make_observation(Signal.STATUS, (2,), PodStatus(status=Status.INACTIVE), 1.1),
        ]
        assert list(rule.patches(_view(log), catalog)) == []

    def test_a_none_neutral_reads_as_confirmed_absence(self, make_observation, catalog):
        rule = Debounce(Signal.ARM, field="digit", min_duration_s=1.0, neutral=None)
        log = [
            make_observation(Signal.ARM, (2,), ArmDigit(digit=3), 1.0),
            make_observation(Signal.ARM, (2,), ArmDigit(digit=4), 1.1),
            make_observation(Signal.ARM, (2,), ArmDigit(digit=3), 1.2),
        ]
        corrected = apply_patches(log, rule.patches(_view(log), catalog))
        assert corrected[1].value is None

    def test_the_last_run_borrows_the_preceding_spacing(self, make_observation, catalog):
        rule = Debounce(Signal.STATUS, field="status", min_duration_s=2.0, neutral=Status.INACTIVE)
        log = [
            make_observation(Signal.STATUS, (2,), PodStatus(status=Status.INACTIVE), 1.0),
            make_observation(Signal.STATUS, (2,), PodStatus(status=Status.INACTIVE), 2.0),
            make_observation(Signal.STATUS, (2,), PodStatus(status=Status.ACTIVE), 3.0),
        ]
        (patch,) = rule.patches(_view(log), catalog)
        assert (patch.start_s, patch.end_s) == (3.0, 4.0)

    def test_keys_debounce_independently(self, make_observation, catalog):
        rule = Debounce(Signal.PEDALS, field="pressed", min_duration_s=0.15, neutral=False)
        yellow = (2, PedalColor.YELLOW)
        blue = (2, PedalColor.BLUE)
        log = [
            make_observation(Signal.PEDALS, yellow, Press(pressed=True), 1.0),
            make_observation(Signal.PEDALS, yellow, Press(pressed=False), 1.05),
            make_observation(Signal.PEDALS, blue, Press(pressed=True), 1.0),
            make_observation(Signal.PEDALS, blue, Press(pressed=True), 1.5),
            make_observation(Signal.PEDALS, blue, Press(pressed=False), 2.0),
        ]
        (patch,) = rule.patches(_view(log), catalog)
        assert patch.key == yellow


class TestFillGap:
    RULE = FillGap(Signal.POPUPS, field="texts", max_duration_s=0.5, neutral=())

    def _stack(self, text: str, y: int = 0) -> PopupStack:
        return PopupStack(messages=(PopupMessage(text=text, box=Box(x=0, y=y, w=10, h=5)),))

    def test_a_short_gap_between_equal_texts_is_bridged(self, make_observation, catalog):
        log = [
            make_observation(Signal.POPUPS, (2,), self._stack("check arm 2", y=0), 1.0),
            make_observation(Signal.POPUPS, (2,), PopupStack(), 1.1),
            make_observation(Signal.POPUPS, (2,), self._stack("check arm 2", y=8), 1.2),
        ]
        corrected = apply_patches(log, self.RULE.patches(_view(log), catalog))
        # Bridged with the value before the gap, moved boxes notwithstanding.
        assert corrected[1].value == log[0].value

    def test_different_neighbors_are_a_real_transition(self, make_observation, catalog):
        log = [
            make_observation(Signal.POPUPS, (2,), self._stack("popup a"), 1.0),
            make_observation(Signal.POPUPS, (2,), PopupStack(), 1.1),
            make_observation(Signal.POPUPS, (2,), self._stack("popup b"), 1.2),
        ]
        assert list(self.RULE.patches(_view(log), catalog)) == []

    @pytest.mark.parametrize("gap_index", [0, 2])
    def test_a_gap_at_the_series_edge_is_never_bridged(self, make_observation, catalog, gap_index):
        entries = [self._stack("popup a"), self._stack("popup a"), self._stack("popup a")]
        entries[gap_index] = PopupStack()
        log = [
            make_observation(Signal.POPUPS, (2,), value, 1.0 + 0.1 * index)
            for index, value in enumerate(entries)
        ]
        assert list(self.RULE.patches(_view(log), catalog)) == []

    def test_a_long_gap_is_not_bridged(self, make_observation, catalog):
        log = [
            make_observation(Signal.POPUPS, (2,), self._stack("popup a"), 1.0),
            make_observation(Signal.POPUPS, (2,), PopupStack(), 1.1),
            make_observation(Signal.POPUPS, (2,), self._stack("popup a"), 2.0),
        ]
        assert list(self.RULE.patches(_view(log), catalog)) == []

    def test_confirmed_absence_counts_as_a_gap(self, make_observation, catalog):
        log = [
            make_observation(Signal.POPUPS, (2,), self._stack("popup a"), 1.0),
            make_observation(Signal.POPUPS, (2,), None, 1.1),
            make_observation(Signal.POPUPS, (2,), self._stack("popup a"), 1.2),
        ]
        corrected = apply_patches(log, self.RULE.patches(_view(log), catalog))
        assert corrected[1].value == log[0].value


def _geometry(boxed: int) -> FrameGeometry:
    """A layout with ``boxed`` pods segmented; the fourth pod is the camera."""
    pods = tuple(
        PodGeometry(
            column=column,
            role=Role.CAMERA if column == 4 else Role.INSTRUMENT,
            box=Box(x=column * 10, y=0, w=10, h=10) if column <= boxed else None,
            popups=(),
        )
        for column in range(1, 5)
    )
    return FrameGeometry(region=Box(x=0, y=0, w=40, h=20), pods=pods, banner=None)


class TestCompleteLayout:
    RULE = CompleteLayout(max_gap_s=1.0)

    def _log(self, make_observation, values: list) -> list[Observation]:
        return [
            make_observation(Signal.LAYOUT, (), value, 1.0 + 0.1 * index)
            for index, value in enumerate(values)
        ]

    def test_an_isolated_incomplete_run_reads_as_no_ui(self, make_observation, catalog):
        log = self._log(make_observation, [None, _geometry(1), _geometry(2), None])
        corrected = apply_patches(log, self.RULE.patches(_view(log), catalog))
        assert [obs.value for obs in corrected] == [None, None, None, None]

    def test_a_short_gap_between_complete_layouts_is_kept(self, make_observation, catalog):
        log = self._log(make_observation, [_geometry(4), _geometry(3), _geometry(4)])
        assert list(self.RULE.patches(_view(log), catalog)) == []

    def test_a_long_gap_between_complete_layouts_reads_as_no_ui(self, make_observation, catalog):
        gap = [_geometry(3)] * 12
        log = self._log(make_observation, [_geometry(4), *gap, _geometry(4)])
        corrected = apply_patches(log, self.RULE.patches(_view(log), catalog))
        assert [obs.value is None for obs in corrected] == [False, *([True] * 12), False]

    def test_a_gap_beside_absence_is_not_bracketed(self, make_observation, catalog):
        log = self._log(make_observation, [_geometry(4), _geometry(3), None])
        corrected = apply_patches(log, self.RULE.patches(_view(log), catalog))
        assert corrected[1].value is None

    def test_a_run_at_the_series_end_is_covered_to_its_last_frame(self, make_observation, catalog):
        log = self._log(make_observation, [_geometry(4), _geometry(1), _geometry(1)])
        (patch,) = self.RULE.patches(_view(log), catalog)
        assert (patch.start_s, patch.end_s) == (1.1, pytest.approx(1.3))
        assert patch.by == "complete_layout"

    def test_complete_layouts_and_absence_are_untouched(self, make_observation, catalog):
        log = self._log(make_observation, [_geometry(4), None, _geometry(4)])
        assert list(self.RULE.patches(_view(log), catalog)) == []


class TestPedalRequiresInstrument:
    def test_a_press_without_an_instrument_is_released(self, make_observation, catalog):
        rule = PedalRequiresInstrument()
        log = [make_observation(Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=True), 1.0)]
        corrected = apply_patches(log, rule.patches(_view(log), catalog))
        assert corrected[0].value == Press(pressed=False)

    def test_a_carried_instrument_keeps_the_press(self, make_observation, catalog):
        rule = PedalRequiresInstrument()
        log = [
            make_observation(
                Signal.INSTRUMENT, (2,), _instrument("maryland bipolar forceps"), 1.0, frame=0
            ),
            make_observation(
                Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=True), 2.0, frame=30
            ),
        ]
        assert list(rule.patches(_view(log), catalog)) == []

    def test_an_unpressed_pedal_is_left_alone(self, make_observation, catalog):
        rule = PedalRequiresInstrument()
        log = [make_observation(Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=False), 1.0)]
        assert list(rule.patches(_view(log), catalog)) == []


class TestPedalRequiresAction:
    def test_a_press_on_a_color_with_no_labels_is_released(self, make_observation, catalog):
        rule = PedalRequiresAction()
        log = [
            make_observation(
                Signal.INSTRUMENT, (2,), _instrument("maryland bipolar forceps"), 1.0, frame=0
            ),
            make_observation(
                Signal.PEDALS, (2, PedalColor.YELLOW), Press(pressed=True), 1.0, frame=0
            ),
            make_observation(
                Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=True), 1.0, frame=0
            ),
        ]
        corrected = apply_patches(log, rule.patches(_view(log), catalog))
        pressed = {obs.key: obs.value.pressed for obs in corrected if obs.signal is Signal.PEDALS}
        assert pressed == {(2, PedalColor.YELLOW): False, (2, PedalColor.BLUE): True}

    @pytest.mark.parametrize("name", ["unknown instrument", "mystery instrument"])
    def test_missing_knowledge_gives_no_opinion(self, make_observation, catalog, name):
        rule = PedalRequiresAction()
        log = [
            make_observation(Signal.INSTRUMENT, (2,), _instrument(name), 1.0, frame=0),
            make_observation(
                Signal.PEDALS, (2, PedalColor.YELLOW), Press(pressed=True), 1.0, frame=0
            ),
        ]
        assert list(rule.patches(_view(log), catalog)) == []


class TestArmDigitUnique:
    def test_the_lower_score_reads_as_absence_and_keeps_its_score(self, make_observation, catalog):
        rule = ArmDigitUnique()
        log = [
            make_observation(Signal.ARM, (1,), ArmDigit(digit=3), 1.0, frame=0, score=0.9),
            make_observation(Signal.ARM, (2,), ArmDigit(digit=3), 1.0, frame=0, score=0.8),
        ]
        corrected = apply_patches(log, rule.patches(_view(log), catalog))
        assert corrected[0].value == ArmDigit(digit=3)
        assert corrected[1].value is None
        assert corrected[1].score == 0.8

    def test_unique_digits_are_left_alone(self, make_observation, catalog):
        rule = ArmDigitUnique()
        log = [
            make_observation(Signal.ARM, (1,), ArmDigit(digit=1), 1.0, frame=0),
            make_observation(Signal.ARM, (2,), ArmDigit(digit=2), 1.0, frame=0),
        ]
        assert list(rule.patches(_view(log), catalog)) == []


class TestCorrect:
    def test_the_fold_lets_the_gate_silence_the_debounce(self, make_observation, catalog):
        # A short press with no instrument: the gate releases it, so the
        # debounce sees only neutral values and stays quiet.
        log = [
            make_observation(Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=True), 1.0),
            make_observation(Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=False), 1.05),
        ]
        specs = [
            PedalRequiresInstrument(),
            Debounce(Signal.PEDALS, field="pressed", min_duration_s=0.15, neutral=False),
        ]
        patches = correct(log, specs, catalog)
        assert [patch.by for patch in patches] == ["pedal_requires_instrument"]
        corrected = apply_patches(log, patches)
        assert not any(obs.value.pressed for obs in corrected)

    def test_the_pass_is_deterministic(self, make_observation, catalog):
        log = [
            make_observation(Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=True), 1.0),
            make_observation(Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=False), 1.2),
        ]
        specs = [PedalRequiresInstrument(), ArmDigitUnique()]
        assert correct(log, specs, catalog) == correct(log, specs, catalog)


class TestNames:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            (
                Debounce(Signal.PEDALS, field="pressed", min_duration_s=0.15, neutral=False),
                "debounce:pedals",
            ),
            (FillGap(Signal.POPUPS, field="texts", max_duration_s=0.5, neutral=()), "fill:popups"),
            (PedalRequiresInstrument(), "pedal_requires_instrument"),
            (PedalRequiresAction(), "pedal_requires_action"),
            (ArmDigitUnique(), "arm_digit_unique"),
            (CompleteLayout(max_gap_s=1.0), "complete_layout"),
        ],
    )
    def test_names_derive_from_the_declarations(self, spec, expected):
        assert spec.name == expected

    def test_settings_expose_the_name_and_the_declared_fields(self):
        debounce = Debounce(Signal.PEDALS, field="pressed", min_duration_s=0.15, neutral=False)
        assert debounce.settings() == {
            "name": "debounce:pedals",
            "signal": Signal.PEDALS,
            "field": "pressed",
            "min_duration_s": 0.15,
            "neutral": False,
        }
        assert ArmDigitUnique().settings() == {"name": "arm_digit_unique"}


class TestRulesFromSettings:
    def test_settings_round_trip_to_equal_rules(self):
        rules = (
            Debounce(Signal.STATUS, field="status", min_duration_s=0.5, neutral=Status.INACTIVE),
            FillGap(Signal.POPUPS, field="texts", max_duration_s=0.5, neutral=()),
            PedalRequiresInstrument(),
            PedalRequiresAction(),
            ArmDigitUnique(),
            CompleteLayout(max_gap_s=1.0),
        )
        settings = [rule.settings() for rule in rules]
        assert rules_from_settings(settings) == rules
        decoded = json.loads(json.dumps(settings))
        assert rules_from_settings(decoded) == rules

    def test_an_unknown_rule_name_raises(self):
        with pytest.raises(ValueError, match="debounce"):
            rules_from_settings([{"name": "smooth:status"}])
