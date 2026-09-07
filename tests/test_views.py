"""Unit tests for the pure views in hudini.views.

encode_runs is tested with hypothesis over generated snapshot sequences —
the interval laws hold for arbitrary inputs, not just domain fixtures.
"""

from collections.abc import Callable
from itertools import pairwise
from typing import ClassVar

import msgspec.structs
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hudini.catalog import CatalogEntry, ResolvedAlias
from hudini.schema import (
    ArmDigit,
    Banner,
    Box,
    Footer,
    FrameGeometry,
    Instrument,
    Laser,
    Observation,
    OCRWindow,
    OffscreenBar,
    OffscreenBars,
    OffscreenBarStatus,
    PedalColor,
    PedalLabel,
    PodGeometry,
    PodStatus,
    PopupMessage,
    PopupStack,
    Press,
    Role,
    Signal,
    Status,
    ToolBadge,
    ToolBadges,
    decode_footer,
)
from hudini.schema import (
    encode as schema_encode,
)
from hudini.views import (
    LANE_PROJECTORS,
    SUMMARY_VERSION,
    UNKNOWN_INSTRUMENT_TYPE,
    FrameState,
    Interval,
    Lane,
    LaneName,
    LogView,
    OffscreenRuns,
    Patch,
    PatchAction,
    RunTotals,
    Summary,
    ToolAssociationRuns,
    apply_patches,
    encode_runs,
    frame_view,
    interval_view,
    iter_frame_states,
    lanes,
    snapshot,
    summarize,
    summary_view,
)

GEOMETRY = FrameGeometry(
    region=Box(x=0, y=0, w=1920, h=1080),
    pods=(
        PodGeometry(column=1, role=Role.CAMERA, box=Box(x=0, y=900, w=200, h=48)),
        PodGeometry(column=2, role=Role.INSTRUMENT, box=Box(x=480, y=900, w=200, h=48)),
    ),
)


class NamingCatalog:
    """Catalog double: names every pressed action deterministically."""

    def pedal_action(self, *, instrument: str, color: PedalColor, label: str | None) -> str | None:
        if label is not None:
            return label
        return f"{color.value}-action"


class SilentCatalog:
    """Catalog double that never resolves an action."""

    def pedal_action(self, *, instrument: str, color: PedalColor, label: str | None) -> str | None:
        return None


@pytest.fixture
def make_observation() -> Callable[..., Observation]:
    """Build an observation with overridable fields."""

    def _make(
        signal: Signal,
        value: object,
        key: tuple = (),
        time_s: float = 0.0,
        frame: int = 0,
        score: float = 1.0,
    ) -> Observation:
        return Observation(
            time_s=time_s, frame=frame, signal=signal, key=key, value=value, score=score
        )

    return _make


@pytest.fixture
def make_frame(make_observation: Callable[..., Observation]) -> Callable[..., list[Observation]]:
    """One frame's observations: geometry plus whatever the test adds."""

    def _make(time_s: float, frame: int, *extra: tuple) -> list[Observation]:
        observations = [
            make_observation(Signal.LAYOUT, GEOMETRY, key=(), time_s=time_s, frame=frame)
        ]
        for signal, key, value in extra:
            observations.append(
                make_observation(signal, value, key=key, time_s=time_s, frame=frame)
            )
        return observations

    return _make


class TestSnapshot:
    def test_a_value_updates_and_a_null_clears(self, make_observation):
        log = [
            make_observation(Signal.INSTRUMENT, _instrument("A"), key=(2,), time_s=1.0),
            make_observation(Signal.INSTRUMENT, None, key=(2,), time_s=2.0),
        ]
        assert (Signal.INSTRUMENT, 2) in snapshot(log, at_s=1.5)
        assert (Signal.INSTRUMENT, 2) not in snapshot(log, at_s=2.5)

    def test_no_observation_holds_the_previous_value(self, make_observation):
        log = [make_observation(Signal.INSTRUMENT, _instrument("A"), key=(2,), time_s=1.0)]
        held = snapshot(log, at_s=100.0)[(Signal.INSTRUMENT, 2)]
        assert held.value.name == "A"
        assert held.time_s == 1.0

    def test_observations_after_at_s_are_invisible(self, make_observation):
        log = [make_observation(Signal.INSTRUMENT, _instrument("A"), key=(2,), time_s=5.0)]
        assert snapshot(log, at_s=4.9) == {}


class TestIterFrameStates:
    def test_groups_by_frame_and_carries_state_forward(self, make_frame):
        log = make_frame(1.0, 10, (Signal.INSTRUMENT, (2,), _instrument("A"))) + make_frame(2.0, 20)
        first, second = list(iter_frame_states(log))
        assert first.frame == 10 and len(first.fresh) == 2
        assert second.frame == 20 and len(second.fresh) == 1
        assert (Signal.INSTRUMENT, 2) in second.state

    def test_state_is_a_copy_per_frame(self, make_frame):
        log = make_frame(1.0, 10) + make_frame(2.0, 20)
        first, second = list(iter_frame_states(log))
        first.state.clear()
        assert (Signal.LAYOUT,) in second.state

    def test_state_includes_the_current_frames_reads(self, make_frame):
        log = make_frame(1.0, 10, (Signal.INSTRUMENT, (2,), _instrument("A")))
        (only,) = list(iter_frame_states(log))
        assert (Signal.INSTRUMENT, 2) in only.state


class TestApplyPatches:
    def test_delete_skips_covered_observations(self, make_observation):
        log = [make_observation(Signal.PEDALS, Press(pressed=True), key=(2, PedalColor.BLUE))]
        patch = Patch(
            signal=Signal.PEDALS,
            key=(2, PedalColor.BLUE),
            start_s=0.0,
            end_s=1.0,
            action=PatchAction.DELETE,
            by="test",
        )
        assert apply_patches(log, [patch]) == []

    def test_set_replaces_the_value_and_keeps_time_and_score(self, make_observation):
        log = [
            make_observation(
                Signal.PEDALS, Press(pressed=True), key=(2, PedalColor.BLUE), score=0.7
            )
        ]
        patch = Patch(
            signal=Signal.PEDALS,
            key=(2, PedalColor.BLUE),
            start_s=0.0,
            end_s=1.0,
            action=PatchAction.SET,
            by="test",
            value=Press(pressed=False),
        )
        (corrected,) = apply_patches(log, [patch])
        assert corrected.value == Press(pressed=False)
        assert corrected.score == 0.7
        assert corrected.time_s == 0.0

    def test_a_span_is_half_open(self, make_observation):
        log = [make_observation(Signal.PEDALS, Press(pressed=True), key=(), time_s=1.0)]
        outside = Patch(
            signal=Signal.PEDALS,
            key=(),
            start_s=0.0,
            end_s=1.0,
            action=PatchAction.DELETE,
            by="test",
        )
        assert apply_patches(log, [outside]) == log

    def test_a_point_patch_covers_exactly_its_time(self, make_observation):
        log = [
            make_observation(Signal.ARM, ArmDigit(digit=3), key=(2,), time_s=1.0),
            make_observation(Signal.ARM, ArmDigit(digit=3), key=(2,), time_s=2.0),
        ]
        point = Patch(
            signal=Signal.ARM,
            key=(2,),
            start_s=1.0,
            end_s=1.0,
            action=PatchAction.SET,
            by="test",
            value=None,
        )
        corrected = apply_patches(log, [point])
        assert corrected[0].value is None
        assert corrected[1].value == ArmDigit(digit=3)

    def test_other_signals_and_keys_are_untouched(self, make_observation):
        log = [make_observation(Signal.INSTRUMENT, _instrument("A"), key=(2,))]
        patch = Patch(
            signal=Signal.PEDALS,
            key=(2,),
            start_s=0.0,
            end_s=10.0,
            action=PatchAction.DELETE,
            by="test",
        )
        assert apply_patches(log, [patch]) == log

    def test_a_delete_patch_with_a_value_raises(self):
        with pytest.raises(ValueError):
            Patch(
                signal=Signal.PEDALS,
                key=(),
                start_s=0.0,
                end_s=1.0,
                action=PatchAction.DELETE,
                by="test",
                value=Press(pressed=False),
            )


def _reference_apply_patches(log, patches):
    """The straightforward per-observation scan apply_patches must equal."""
    corrected = []
    for obs in log:
        keep = True
        for patch in patches:
            if patch.signal is not obs.signal or patch.key != obs.key:
                continue
            if not patch.covers(obs.time_s):
                continue
            if patch.action is PatchAction.DELETE:
                keep = False
                break
            obs = msgspec.structs.replace(obs, value=patch.value)
        if keep:
            corrected.append(obs)
    return corrected


_TIMES = st.floats(min_value=0.0, max_value=8.0, allow_nan=False, width=16)
_KEYS = st.sampled_from([(1,), (2,), (2, PedalColor.BLUE)])


@st.composite
def _patch_strategy(draw) -> Patch:
    start = draw(_TIMES)
    span = draw(st.floats(min_value=0.0, max_value=4.0, allow_nan=False, width=16))
    action = draw(st.sampled_from(list(PatchAction)))
    value = (
        None
        if action is PatchAction.DELETE
        else draw(st.sampled_from([None, Press(pressed=True), Press(pressed=False)]))
    )
    return Patch(
        signal=Signal.PEDALS,
        key=draw(_KEYS),
        start_s=start,
        end_s=start + span,
        action=action,
        by="hypothesis",
        value=value,
    )


@given(
    times=st.lists(_TIMES, max_size=30),
    keys=st.lists(_KEYS, max_size=30),
    patches=st.lists(_patch_strategy(), max_size=15),
)
def test_apply_patches_equals_the_reference_scan(times, keys, patches):
    log = [
        Observation(
            time_s=time_s,
            frame=index,
            signal=Signal.PEDALS,
            key=key,
            value=Press(pressed=True),
            score=1.0,
        )
        for index, (time_s, key) in enumerate(zip(sorted(times), keys, strict=False))
    ]
    assert apply_patches(log, patches) == _reference_apply_patches(log, patches)


class TestLogView:
    def test_series_groups_one_signals_observations_by_key(self, make_observation):
        log = LogView(
            observations=(
                make_observation(Signal.ARM, ArmDigit(digit=1), key=(1,), time_s=1.0),
                make_observation(Signal.ARM, ArmDigit(digit=2), key=(2,), time_s=1.0),
                make_observation(Signal.ARM, ArmDigit(digit=1), key=(1,), time_s=2.0),
                make_observation(Signal.INSTRUMENT, _instrument("A"), key=(1,), time_s=1.0),
            )
        )
        series = log.series(Signal.ARM)
        assert set(series) == {(1,), (2,)}
        assert [obs.time_s for obs in series[(1,)]] == [1.0, 2.0]


class TestLanes:
    def test_no_geometry_paints_only_the_no_ui_lane(self):
        fs = FrameState(time_s=0.0, frame=0, fresh=(), state={})
        assert lanes(fs, NamingCatalog()) == {Lane(name=LaneName.NO_UI): None}

    def test_camera_column_paints_the_role_lane(self, make_frame):
        (fs,) = list(iter_frame_states(make_frame(1.0, 0)))
        painted = lanes(fs, NamingCatalog())
        assert painted[Lane(name=LaneName.ROLE, arm=1, column=1)] == Role.CAMERA

    def test_the_digit_names_the_arm_and_the_column_is_the_fallback(self, make_frame):
        log = make_frame(
            1.0,
            0,
            (Signal.ARM, (2,), ArmDigit(digit=4)),
            (Signal.STATUS, (1,), PodStatus(status=Status.ACTIVE)),
            (Signal.STATUS, (2,), PodStatus(status=Status.INACTIVE)),
        )
        (fs,) = list(iter_frame_states(log))
        painted = lanes(fs, NamingCatalog())
        assert painted[Lane(name=LaneName.STATUS, arm=4, column=2)] == Status.INACTIVE
        assert painted[Lane(name=LaneName.STATUS, arm=1, column=1)] == Status.ACTIVE

    def test_popups_paint_one_lane_per_message(self, make_frame):
        stack = PopupStack(
            messages=(
                PopupMessage(text="check arm 2", box=Box(x=1, y=2, w=3, h=4)),
                PopupMessage(text="move grip", box=Box(x=1, y=6, w=3, h=4)),
            )
        )
        (fs,) = list(iter_frame_states(make_frame(1.0, 0, (Signal.POPUPS, (2,), stack))))
        painted = lanes(fs, NamingCatalog())
        assert painted[Lane(name=LaneName.POPUP, arm=2, column=2, key=("check arm 2",))]
        assert painted[Lane(name=LaneName.POPUP, arm=2, column=2, key=("move grip",))]

    def test_a_pressed_pedal_paints_its_action_from_the_join(self, make_frame):
        log = make_frame(
            1.0,
            0,
            (Signal.INSTRUMENT, (2,), _instrument("Vessel Sealer Extend")),
            (Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=True)),
            (Signal.PEDAL_LABEL, (2, PedalColor.BLUE), PedalLabel(text="SEAL")),
        )
        (fs,) = list(iter_frame_states(log))
        painted = lanes(fs, NamingCatalog())
        assert painted[Lane(name=LaneName.PEDAL_BLUE, arm=2, column=2)] == "SEAL"

    def test_a_press_without_an_instrument_is_pending(self, make_frame):
        log = make_frame(1.0, 0, (Signal.PEDALS, (2, PedalColor.YELLOW), Press(pressed=True)))
        (fs,) = list(iter_frame_states(log))
        assert lanes(fs, NamingCatalog())[Lane(name=LaneName.PEDAL_YELLOW, arm=2, column=2)] is None

    def test_an_unpressed_pedal_paints_nothing(self, make_frame):
        log = make_frame(1.0, 0, (Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=False)))
        (fs,) = list(iter_frame_states(log))
        painted = lanes(fs, NamingCatalog())
        assert Lane(name=LaneName.PEDAL_BLUE, arm=2, column=2) not in painted

    def test_only_identified_bars_and_badges_paint_lanes(self, make_frame):
        bars = OffscreenBars(
            bars=(
                OffscreenBar(
                    score=0.9,
                    box=Box(x=0, y=0, w=1, h=1),
                    status=OffscreenBarStatus.ACTIVE,
                    arm=4,
                ),
                OffscreenBar(score=0.8, box=Box(x=0, y=9, w=1, h=1)),
            )
        )
        badges = ToolBadges(badges=(ToolBadge(score=0.9, box=Box(x=0, y=0, w=1, h=1), arm=3),))
        log = make_frame(
            1.0,
            0,
            (Signal.OFFSCREEN, (), bars),
            (Signal.TOOL_ASSOCIATION, (), badges),
        )
        (fs,) = list(iter_frame_states(log))
        painted = lanes(fs, NamingCatalog())
        offscreen_lanes = [lane for lane in painted if lane.name is LaneName.OFFSCREEN]
        assert offscreen_lanes == [Lane(name=LaneName.OFFSCREEN, arm=4)]
        assert painted[Lane(name=LaneName.TOOL_ASSOCIATION, arm=3)] is None

    def test_the_projector_table_covers_every_signal(self):
        assert set(LANE_PROJECTORS) == set(Signal)


class TestEncodeRuns:
    def test_open_close_and_split(self):
        lane = Lane(name=LaneName.INSTRUMENT, arm=2, column=2)
        frames = [
            (1.0, {lane: "A"}),
            (2.0, {lane: "A"}),
            (3.0, {lane: "B"}),
            (4.0, {}),
        ]
        assert encode_runs(frames) == [
            Interval(lane=LaneName.INSTRUMENT, arm=2, column=2, start_s=1.0, end_s=3.0, value="A"),
            Interval(lane=LaneName.INSTRUMENT, arm=2, column=2, start_s=3.0, end_s=4.0, value="B"),
        ]

    def test_a_pending_value_fills_the_run_backwards(self):
        lane = Lane(name=LaneName.PEDAL_BLUE, arm=1, column=1)
        frames = [(1.0, {lane: None}), (1.1, {lane: "SEAL"}), (1.2, {lane: "SEAL"})]
        (interval,) = encode_runs(frames)
        assert interval.start_s == 1.0
        assert interval.value == "SEAL"

    def test_a_none_after_a_real_value_never_splits(self):
        lane = Lane(name=LaneName.PEDAL_BLUE, arm=1, column=1)
        frames = [(1.0, {lane: "SEAL"}), (2.0, {lane: None}), (3.0, {lane: "SEAL"})]
        assert len(encode_runs(frames)) == 1

    def test_the_last_runs_close_one_spacing_past_the_final_frame(self):
        lane = Lane(name=LaneName.BANNER)
        frames = [(1.0, {lane: "x"}), (2.0, {lane: "x"})]
        (interval,) = encode_runs(frames)
        assert interval.end_s == 3.0

    def test_an_empty_snapshot_closes_everything(self):
        lane = Lane(name=LaneName.STATUS, arm=1, column=1)
        no_ui = Lane(name=LaneName.NO_UI)
        frames = [(1.0, {lane: "active"}), (2.0, {no_ui: None}), (3.0, {lane: "active"})]
        first, second = [entry for entry in encode_runs(frames) if entry.lane is LaneName.STATUS]
        assert (first.start_s, first.end_s) == (1.0, 2.0)
        assert (second.start_s, second.end_s) == (3.0, 4.0)


def _lane_strategy() -> st.SearchStrategy[Lane]:
    return st.builds(
        Lane,
        name=st.sampled_from([LaneName.STATUS, LaneName.BANNER, LaneName.POPUP]),
        arm=st.one_of(st.none(), st.integers(min_value=1, max_value=4)),
    )


def _frames_strategy() -> st.SearchStrategy[list[tuple[float, dict[Lane, str | None]]]]:
    times = st.lists(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, width=16),
        min_size=1,
        max_size=8,
        unique=True,
    ).map(sorted)
    snapshots = st.lists(
        st.dictionaries(
            _lane_strategy(), st.one_of(st.none(), st.sampled_from(["a", "b"])), max_size=3
        ),
        min_size=8,
        max_size=8,
    )
    return st.tuples(times, snapshots).map(
        lambda pair: [(t, snap) for t, snap in zip(pair[0], pair[1], strict=False)]
    )


class TestEncodeRunsLaws:
    @given(frames=_frames_strategy())
    def test_runs_of_one_lane_never_overlap_and_tile_half_open(self, frames):
        intervals = encode_runs(frames)
        by_lane: dict[tuple, list[Interval]] = {}
        for entry in intervals:
            by_lane.setdefault((entry.lane, entry.arm, entry.column), []).append(entry)
        for runs in by_lane.values():
            runs.sort(key=lambda entry: entry.start_s)
            for first, second in pairwise(runs):
                assert first.end_s <= second.start_s

    @given(frames=_frames_strategy())
    def test_every_interval_spans_forward(self, frames):
        for entry in encode_runs(frames):
            assert entry.end_s >= entry.start_s

    @given(frames=_frames_strategy())
    def test_a_lane_present_at_a_frame_is_covered_by_exactly_one_interval(self, frames):
        intervals = encode_runs(frames)
        last_s = frames[-1][0]
        for time_s, snap in frames:
            for lane in snap:
                covering = [
                    entry
                    for entry in intervals
                    if (entry.lane, entry.arm, entry.column) == (lane.name, lane.arm, lane.column)
                    and (entry.start_s <= time_s < entry.end_s or time_s == last_s == entry.end_s)
                ]
                assert len(covering) >= 1

    @given(frames=_frames_strategy())
    def test_encoding_is_deterministic(self, frames):
        assert encode_runs(frames) == encode_runs(list(frames))


class TestIntervalView:
    def test_a_press_with_a_late_action_becomes_one_named_interval(self, make_frame):
        press = (Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=True))
        instrument = (Signal.INSTRUMENT, (2,), _instrument("Vessel Sealer Extend"))
        label = (Signal.PEDAL_LABEL, (2, PedalColor.BLUE), PedalLabel(text="SEAL"))
        released = (Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=False))
        log = (
            make_frame(1.0, 0, press)
            + make_frame(1.1, 3, press, instrument, label)
            + make_frame(1.2, 6, press)
            + make_frame(1.3, 9, released)
        )
        intervals = interval_view(log, [], NamingCatalog())
        (pedal,) = [entry for entry in intervals if entry.lane is LaneName.PEDAL_BLUE]
        assert (pedal.start_s, pedal.end_s, pedal.value) == (1.0, 1.3, "SEAL")

    def test_a_delete_patch_changes_the_intervals(self, make_frame):
        press = (Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=True))
        log = make_frame(1.0, 0, press) + make_frame(2.0, 3)
        patch = Patch(
            signal=Signal.PEDALS,
            key=(2, PedalColor.BLUE),
            start_s=1.0,
            end_s=1.0,
            action=PatchAction.SET,
            by="test",
            value=Press(pressed=False),
        )
        with_press = interval_view(log, [], SilentCatalog())
        without_press = interval_view(log, [patch], SilentCatalog())
        assert any(entry.lane is LaneName.PEDAL_BLUE for entry in with_press)
        assert not any(entry.lane is LaneName.PEDAL_BLUE for entry in without_press)


def _instrument(name: str) -> Instrument:
    return Instrument(name=name, match=95.0, raw=name.lower(), window=OCRWindow.NARROW)


class TestFrameView:
    def test_a_fresh_frame_shapes_the_full_pod_record(self, make_frame):
        log = make_frame(
            0.0,
            0,
            (Signal.STATUS, (2,), PodStatus(status=Status.ACTIVE)),
            (Signal.ARM, (2,), ArmDigit(digit=3)),
            (Signal.INSTRUMENT, (2,), _instrument("force bipolar")),
            (Signal.PEDALS, (2, PedalColor.YELLOW), Press(pressed=True)),
            (Signal.PEDAL_LABEL, (2, PedalColor.YELLOW), PedalLabel(text="strong")),
        )
        record = frame_view(log, (), NamingCatalog())[0]
        assert record["frame_index"] == 0
        assert record["pods"][0] == {
            "column": 1,
            "role": Role.CAMERA,
            "box": {"x": 0, "y": 900, "w": 200, "h": 48},
        }
        assert record["pods"][1] == {
            "column": 2,
            "role": Role.INSTRUMENT,
            "box": {"x": 480, "y": 900, "w": 200, "h": 48},
            "status": {"value": Status.ACTIVE, "score": 1.0},
            "arm": {"value": 3, "score": 1.0},
            "instrument": {
                "value": "force bipolar",
                "score": 1.0,
                "ocr": {"raw": "force bipolar", "window": OCRWindow.NARROW},
            },
            "pedals": {PedalColor.YELLOW: {"value": True, "score": 1.0, "action": "strong"}},
        }

    def test_carried_values_are_stamped_with_observed_at(self, make_frame):
        log = make_frame(0.0, 0, (Signal.INSTRUMENT, (2,), _instrument("needle driver")))
        log += make_frame(1.0, 30)
        first, second = frame_view(log, (), SilentCatalog())
        assert "observed_at_s" not in first["pods"][1]["instrument"]
        assert second["pods"][1]["instrument"]["observed_at_s"] == 0.0

    def test_a_frame_without_ui_keeps_the_belief_but_no_boxes(self, make_frame, make_observation):
        log = make_frame(0.0, 0, (Signal.BANNER, (), Banner(text="table motion")))
        log.append(make_observation(Signal.LAYOUT, None, time_s=5.0, frame=150))
        record = frame_view(log, (), SilentCatalog())[1]
        assert record["pods"] == []
        assert record["banner"] == {"value": "table motion", "score": 1.0, "observed_at_s": 0.0}

    def test_the_banner_box_comes_from_this_frames_geometry(self, make_observation):
        geometry = FrameGeometry(
            region=GEOMETRY.region, pods=GEOMETRY.pods, banner=Box(x=0, y=0, w=1920, h=40)
        )
        log = [
            make_observation(Signal.LAYOUT, geometry),
            make_observation(Signal.BANNER, Banner(text="table motion")),
        ]
        record = frame_view(log, (), SilentCatalog())[0]
        assert record["banner"] == {
            "box": {"x": 0, "y": 0, "w": 1920, "h": 40},
            "value": "table motion",
            "score": 1.0,
        }

    def test_popups_and_detectors_shape_their_lists(self, make_frame):
        stack = PopupStack(
            messages=(PopupMessage(text="check arm", box=Box(x=480, y=850, w=200, h=40)),)
        )
        bars = OffscreenBars(
            bars=(
                OffscreenBar(
                    score=0.95678,
                    box=Box(x=0, y=60, w=220, h=14),
                    status=OffscreenBarStatus.ACTIVE,
                    arm=2,
                ),
            )
        )
        badges = ToolBadges(
            badges=(ToolBadge(score=0.9, box=Box(x=822, y=321, w=72, h=73), arm=4),)
        )
        log = make_frame(
            0.0,
            0,
            (Signal.POPUPS, (2,), stack),
            (Signal.OFFSCREEN, (), bars),
            (Signal.TOOL_ASSOCIATION, (), badges),
        )
        record = frame_view(log, (), SilentCatalog())[0]
        assert record["pods"][1]["popups"] == [
            {"value": "check arm", "score": 1.0, "box": {"x": 480, "y": 850, "w": 200, "h": 40}}
        ]
        assert record["offscreen"] == {
            "detections": [
                {
                    "status": OffscreenBarStatus.ACTIVE,
                    "arm": 2,
                    "score": 0.9568,
                    "box": {"x": 0, "y": 60, "w": 220, "h": 14},
                }
            ]
        }
        assert record["tool_association"]["detections"][0]["arm"] == 4

    def test_patches_apply_before_shaping(self, make_frame):
        log = make_frame(0.0, 0, (Signal.STATUS, (2,), PodStatus(status=Status.ACTIVE)))
        patch = Patch(
            signal=Signal.STATUS,
            key=(2,),
            start_s=0.0,
            end_s=0.0,
            action=PatchAction.SET,
            by="test",
            value=PodStatus(status=Status.INACTIVE),
        )
        record = frame_view(log, [patch], SilentCatalog())[0]
        assert record["pods"][1]["status"]["value"] == Status.INACTIVE


class TestLaserLane:
    def test_laser_on_paints_the_camera_lane(self, make_frame):
        log = make_frame(0.0, 0, (Signal.LASER, (1,), Laser(on=True)))
        (fs,) = list(iter_frame_states(log))
        painted = lanes(fs, SilentCatalog())
        assert painted[Lane(name=LaneName.LASER, arm=1, column=1)] is None

    def test_laser_off_paints_no_lane(self, make_frame):
        log = make_frame(0.0, 0, (Signal.LASER, (1,), Laser(on=False)))
        (fs,) = list(iter_frame_states(log))
        painted = lanes(fs, SilentCatalog())
        assert all(lane.name is not LaneName.LASER for lane in painted)

    def test_the_camera_pod_record_carries_the_laser_value(self, make_frame):
        log = make_frame(0.0, 0, (Signal.LASER, (1,), Laser(on=True)))
        (record,) = frame_view(log, (), SilentCatalog())
        assert record["pods"][0]["laser"]["value"] is True


class ResolvingCatalog(NamingCatalog):
    """Catalog double: resolves a fixed alias-to-class table."""

    TYPES: ClassVar[dict[str, str]] = {
        "Cadiere Forceps": "grasper",
        "Vessel Sealer Extend": "vessel_sealer",
    }

    def resolve(self, alias: str) -> ResolvedAlias | None:
        entry_type = self.TYPES.get(alias)
        if entry_type is None:
            return None
        entry = CatalogEntry(name=alias, display_name=alias, type=entry_type, pedals=None)
        return ResolvedAlias(entry=entry, alias=alias, reload_color=None)


@pytest.fixture
def make_interval() -> Callable[..., Interval]:
    """Build an interval with overridable fields."""

    def _make(
        lane: LaneName,
        start_s: float,
        end_s: float,
        value: str | None = None,
        arm: int | None = None,
        column: int | None = None,
    ) -> Interval:
        return Interval(
            lane=lane, arm=arm, column=column, start_s=start_s, end_s=end_s, value=value
        )

    return _make


class TestSummarize:
    def test_instruments_fold_by_first_appearance_with_type_and_arms(self, make_interval):
        summary = summarize(
            [
                make_interval(LaneName.INSTRUMENT, 0.0, 10.0, "Vessel Sealer Extend", arm=2),
                make_interval(LaneName.INSTRUMENT, 0.0, 4.0, "Cadiere Forceps", arm=1),
                make_interval(LaneName.INSTRUMENT, 12.0, 15.0, "Cadiere Forceps", arm=4),
                make_interval(LaneName.INSTRUMENT, 20.0, 21.0, "Mystery Tool", arm=1),
            ],
            ResolvingCatalog(),
        )
        assert [use.name for use in summary.instruments] == [
            "Vessel Sealer Extend",
            "Cadiere Forceps",
            "Mystery Tool",
        ]
        cadiere = summary.instruments[1]
        assert cadiere.type == "grasper"
        assert cadiere.duration_s == 7.0
        assert cadiere.arms == (1, 4)
        assert summary.instruments[2].type == UNKNOWN_INSTRUMENT_TYPE

    def test_instrument_changes_count_swaps_per_arm(self, make_interval):
        summary = summarize(
            [
                make_interval(LaneName.INSTRUMENT, 0.0, 5.0, "Cadiere Forceps", arm=1),
                make_interval(LaneName.INSTRUMENT, 5.0, 9.0, "Vessel Sealer Extend", arm=1),
                make_interval(LaneName.INSTRUMENT, 9.0, 12.0, "Vessel Sealer Extend", arm=1),
                make_interval(LaneName.INSTRUMENT, 0.0, 12.0, "Cadiere Forceps", arm=2),
                make_interval(LaneName.INSTRUMENT, 13.0, 14.0, None, arm=2),
            ],
            ResolvingCatalog(),
        )
        assert summary.instrument_changes == 1

    def test_press_runs_fold_per_color_and_action(self, make_interval):
        summary = summarize(
            [
                make_interval(LaneName.PEDAL_YELLOW, 0.0, 0.5, "coag", arm=1),
                make_interval(LaneName.PEDAL_YELLOW, 2.0, 2.25, "coag", arm=4),
                make_interval(LaneName.PEDAL_BLUE, 3.0, 3.5, None, arm=1),
            ],
            ResolvingCatalog(),
        )
        assert summary.presses[PedalColor.YELLOW] == RunTotals(count=2, duration_s=0.75)
        assert summary.presses[PedalColor.BLUE] == RunTotals(count=1, duration_s=0.5)
        assert summary.presses_by_action == {"coag": 2}
        assert summary.actions_by_pedal == {PedalColor.YELLOW: {"coag": 2}, PedalColor.BLUE: {}}

    def test_actions_split_by_the_pedal_that_pressed_them(self, make_interval):
        summary = summarize(
            [
                make_interval(LaneName.PEDAL_YELLOW, 0.0, 0.5, "grip", arm=1),
                make_interval(LaneName.PEDAL_BLUE, 1.0, 1.5, "seal", arm=1),
                make_interval(LaneName.PEDAL_BLUE, 2.0, 2.5, "seal", arm=1),
            ],
            ResolvingCatalog(),
        )
        assert summary.actions_by_pedal == {
            PedalColor.YELLOW: {"grip": 1},
            PedalColor.BLUE: {"seal": 2},
        }

    def test_laser_runs_fold_to_totals(self, make_interval):
        summary = summarize(
            [
                make_interval(LaneName.LASER, 10.0, 16.0, None, arm=1),
                make_interval(LaneName.LASER, 20.0, 21.5, None, arm=1),
            ],
            ResolvingCatalog(),
        )
        assert summary.laser == RunTotals(count=2, duration_s=7.5)

    def test_offscreen_rows_sort_by_arm_then_status(self, make_interval):
        summary = summarize(
            [
                make_interval(LaneName.OFFSCREEN, 0.0, 2.0, "inactive", arm=4),
                make_interval(LaneName.OFFSCREEN, 3.0, 4.0, "active", arm=1),
                make_interval(LaneName.OFFSCREEN, 5.0, 6.0, "active", arm=1),
                make_interval(LaneName.OFFSCREEN, 7.0, 8.0, None, arm=1),
            ],
            ResolvingCatalog(),
        )
        assert summary.offscreen == (
            OffscreenRuns(arm=1, status=OffscreenBarStatus.ACTIVE, count=2, duration_s=2.0),
            OffscreenRuns(arm=1, status=None, count=1, duration_s=1.0),
            OffscreenRuns(arm=4, status=OffscreenBarStatus.INACTIVE, count=1, duration_s=2.0),
        )

    def test_tool_association_rows_fold_per_arm(self, make_interval):
        summary = summarize(
            [
                make_interval(LaneName.TOOL_ASSOCIATION, 0.0, 1.0, None, arm=2),
                make_interval(LaneName.TOOL_ASSOCIATION, 5.0, 5.5, None, arm=2),
                make_interval(LaneName.TOOL_ASSOCIATION, 6.0, 6.5, None, arm=4),
            ],
            ResolvingCatalog(),
        )
        assert summary.tool_association == (
            ToolAssociationRuns(arm=2, count=2, duration_s=1.5),
            ToolAssociationRuns(arm=4, count=1, duration_s=0.5),
        )

    def test_popup_banner_warning_and_no_ui_fold(self, make_interval):
        summary = summarize(
            [
                make_interval(LaneName.POPUP, 0.0, 2.0, "Move grip to match instrument", arm=2),
                make_interval(LaneName.POPUP, 5.0, 6.0, "Move grip to match instrument", arm=2),
                make_interval(LaneName.POPUP, 7.0, 8.0, "Instrument fully inserted", arm=1),
                make_interval(LaneName.BANNER, 10.0, 14.0, "Table motion in progress"),
                make_interval(LaneName.STATUS, 0.0, 3.5, Status.WARNING, arm=4),
                make_interval(LaneName.STATUS, 3.5, 9.0, Status.ACTIVE, arm=4),
                make_interval(LaneName.NO_UI, 20.0, 26.0),
            ],
            ResolvingCatalog(),
        )
        assert summary.popup_count == 3
        assert summary.popup_texts == (
            "Instrument fully inserted",
            "Move grip to match instrument",
        )
        assert summary.banner_duration_s == 4.0
        assert summary.banner_texts == ("Table motion in progress",)
        assert summary.warning_duration_s == 3.5
        assert summary.no_ui_duration_s == 6.0

    def test_durations_round_to_milliseconds(self, make_interval):
        summary = summarize(
            [make_interval(LaneName.LASER, 0.0, 0.1 + 0.2, None, arm=1)],
            ResolvingCatalog(),
        )
        assert summary.laser.duration_s == 0.3

    def test_the_same_intervals_give_the_same_bytes(self, make_interval):
        intervals = [
            make_interval(LaneName.INSTRUMENT, 0.0, 4.0, "Cadiere Forceps", arm=1),
            make_interval(LaneName.PEDAL_BLUE, 1.0, 1.5, "grip", arm=1),
            make_interval(LaneName.OFFSCREEN, 2.0, 3.0, "active", arm=1),
        ]
        forward = msgspec.json.encode(summarize(intervals, ResolvingCatalog()).to_wire())
        backward = msgspec.json.encode(
            summarize(list(reversed(intervals)), ResolvingCatalog()).to_wire()
        )
        assert forward == backward


class TestSummaryView:
    def test_a_small_log_folds_end_to_end(self, make_frame):
        log = [
            *make_frame(
                0.0,
                0,
                (Signal.INSTRUMENT, (2,), _instrument("Cadiere Forceps")),
                (Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=True)),
            ),
            *make_frame(
                1.0,
                1,
                (Signal.INSTRUMENT, (2,), _instrument("Cadiere Forceps")),
                (Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=False)),
            ),
        ]
        summary = summary_view(log, (), ResolvingCatalog())
        assert summary.instruments[0].name == "Cadiere Forceps"
        assert summary.presses[PedalColor.BLUE].count == 1
        assert summary.presses[PedalColor.YELLOW].count == 0


class TestSummaryFooterCodec:
    def test_a_footer_summary_round_trips(self, make_interval):
        summary = summarize(
            [make_interval(LaneName.LASER, 0.0, 2.0, None, arm=1)], ResolvingCatalog()
        )
        footer = Footer(
            observations=1,
            frames_sampled=1,
            achieved_fps=1.0,
            runtime_s=1.0,
            summary_version=SUMMARY_VERSION,
            summary=summary.to_wire(),
        )
        line = schema_encode(footer)
        assert Summary.from_footer(decode_footer(line)) == summary

    def test_a_summary_written_without_actions_by_pedal_still_converts(self, make_interval):
        summary = summarize(
            [make_interval(LaneName.LASER, 0.0, 2.0, None, arm=1)], ResolvingCatalog()
        )
        wire = summary.to_wire()
        del wire["actions_by_pedal"]
        footer = Footer(
            observations=1,
            frames_sampled=1,
            achieved_fps=1.0,
            runtime_s=1.0,
            summary_version=SUMMARY_VERSION,
            summary=wire,
        )
        converted = Summary.from_footer(decode_footer(schema_encode(footer)))
        assert converted is not None
        assert converted.actions_by_pedal == {}

    def test_a_footer_without_a_summary_reads_as_none(self):
        footer = Footer(observations=1, frames_sampled=1, achieved_fps=1.0, runtime_s=1.0)
        assert Summary.from_footer(decode_footer(schema_encode(footer))) is None

    def test_an_unknown_summary_version_is_not_converted(self):
        footer = Footer(
            observations=1,
            frames_sampled=1,
            achieved_fps=1.0,
            runtime_s=1.0,
            summary_version=SUMMARY_VERSION + 1,
            summary={"not": "a summary shape"},
        )
        assert Summary.from_footer(decode_footer(schema_encode(footer))) is None
