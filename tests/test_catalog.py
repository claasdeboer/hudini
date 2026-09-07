"""Unit tests for hudini.catalog.

Behavior is tested against a synthetic catalog file; two smoke tests load
the real bundled pods.json.
"""

from collections.abc import Callable

import pytest

from hudini.catalog import Catalog, _CatalogFile, _RawEntry, _RawPedals
from hudini.schema import (
    Box,
    FrameGeometry,
    Instrument,
    Observation,
    OCRWindow,
    PedalColor,
    PodGeometry,
    Press,
    Role,
    Signal,
)
from hudini.views import Lane, LaneName, iter_frame_states, lanes

COLOR_LOCALES = {
    "en": {"blue": "blue", "white": "white"},
    "de": {"blue": "blau", "white": "weiss"},
}


@pytest.fixture
def make_catalog() -> Callable[..., Catalog]:
    """Build a catalog from synthetic entries, defaulting to a realistic set."""

    def _make(entries: tuple[_RawEntry, ...] | None = None, locale: str = "en") -> Catalog:
        if entries is None:
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
                _RawEntry(name="mystery instrument", type="grasper"),
                _RawEntry(
                    name="stapler 45",
                    type="stapler",
                    localized={"de": "klammergerät 45"},
                    reload_colors=("blue", "white"),
                ),
                _RawEntry(name="menu is open", type="system_message"),
            )
        return Catalog(_CatalogFile(color_locales=COLOR_LOCALES, entries=entries), locale=locale)

    return _make


class TestResolve:
    def test_canonical_name_resolves_to_its_entry(self, make_catalog):
        resolution = make_catalog().resolve("force instrument")
        assert resolution.entry.name == "force instrument"
        assert resolution.reload_color is None

    def test_resolution_ignores_letter_case(self, make_catalog):
        assert make_catalog().resolve("Force Instrument").entry.name == "force instrument"

    def test_a_localized_alias_resolves_in_its_locale(self, make_catalog):
        catalog = make_catalog(locale="de")
        resolution = catalog.resolve("klammergerät 45")
        assert resolution.entry.name == "stapler 45"
        assert resolution.entry.display_name == "klammergerät 45"

    def test_a_reload_variant_carries_the_canonical_color_key(self, make_catalog):
        resolution = make_catalog(locale="de").resolve("stapler 45 [blau]")
        assert resolution.reload_color == "blue"

    def test_an_unknown_alias_resolves_to_none(self, make_catalog):
        assert make_catalog().resolve("no such thing") is None


class TestValidation:
    def test_an_unknown_locale_raises(self, make_catalog):
        with pytest.raises(ValueError):
            make_catalog(locale="fr")

    def test_two_entries_claiming_one_alias_raise(self, make_catalog):
        entries = (
            _RawEntry(name="same name", type="grasper"),
            _RawEntry(name="Same Name", type="stapler"),
        )
        with pytest.raises(ValueError):
            make_catalog(entries=entries)

    def test_an_untranslated_reload_color_raises(self, make_catalog):
        entries = (_RawEntry(name="stapler 60", type="stapler", reload_colors=("green",)),)
        with pytest.raises(ValueError):
            make_catalog(entries=entries)


class TestMatch:
    def test_exact_text_wins_with_a_full_score(self, make_catalog):
        match = make_catalog().match("force instrument")
        assert match.best.resolution.entry.name == "force instrument"
        assert match.best.score == 100

    def test_empty_text_matches_nothing(self, make_catalog):
        assert make_catalog().match("   ") is None

    def test_text_below_the_threshold_matches_nothing(self, make_catalog):
        assert make_catalog().match("zzzzqqqq") is None

    def test_aliases_of_one_entry_are_not_opponents(self, make_catalog):
        # "stapler 45" and its reload variants must collapse to one candidate.
        match = make_catalog().match("stapler 45")
        names = [candidate.resolution.entry.name for candidate in match.candidates]
        assert names.count("stapler 45") == 1

    def test_margin_is_none_without_competition(self, make_catalog):
        match = make_catalog().match("menu is open")
        assert match.margin is None

    def test_margin_measures_the_distance_to_a_different_entry(self, make_catalog):
        entries = (
            _RawEntry(name="sureform 45 curved-tip", type="stapler"),
            _RawEntry(name="sureform 30 curved-tip", type="stapler"),
        )
        match = make_catalog(entries=entries).match("sureform 45 curved-tip")
        assert match.best.resolution.entry.name == "sureform 45 curved-tip"
        assert match.margin is not None and match.margin > 0


class TestPedalAction:
    def test_an_unknown_instrument_gives_none(self, make_catalog):
        action = make_catalog().pedal_action(
            instrument="no such thing", color=PedalColor.BLUE, label=None
        )
        assert action is None

    def test_missing_pedal_data_gives_none(self, make_catalog):
        action = make_catalog().pedal_action(
            instrument="mystery instrument", color=PedalColor.BLUE, label=None
        )
        assert action is None

    def test_a_pedal_with_no_action_gives_none(self, make_catalog):
        action = make_catalog().pedal_action(
            instrument="plain instrument", color=PedalColor.YELLOW, label="COAG"
        )
        assert action is None

    def test_a_single_label_needs_no_ocr(self, make_catalog):
        action = make_catalog().pedal_action(
            instrument="plain instrument", color=PedalColor.BLUE, label=None
        )
        assert action == "coag"

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("STRONG", "strong"),
            ("strono", "strong"),
            ("GRIP", "grip"),
            (None, None),
            ("zzzz", None),
        ],
    )
    def test_a_press_dependent_pedal_resolves_its_label(self, make_catalog, label, expected):
        action = make_catalog().pedal_action(
            instrument="force instrument", color=PedalColor.YELLOW, label=label
        )
        assert action == expected


class TestBundledFile:
    def test_the_bundled_catalog_loads_for_both_locales(self):
        for locale in ("en", "de"):
            catalog = Catalog.load(locale=locale)
            assert catalog.resolve("maryland bipolar forceps") is not None

    def test_maryland_blue_names_its_action_without_a_label(self):
        catalog = Catalog.load()
        action = catalog.pedal_action(
            instrument="maryland bipolar forceps", color=PedalColor.BLUE, label=None
        )
        assert action == "bipolar"

    def test_the_catalog_satisfies_the_lane_projections_protocol(self):
        # The seam: views.lanes takes the real Catalog as its ActionCatalog.
        geometry = FrameGeometry(
            region=Box(x=0, y=0, w=100, h=100),
            pods=(PodGeometry(column=2, role=Role.INSTRUMENT, box=Box(x=0, y=0, w=10, h=5)),),
        )
        instrument = Instrument(
            name="maryland bipolar forceps",
            match=95.0,
            raw="maryland bipolar",
            window=OCRWindow.NARROW,
        )
        log = [
            _observation(Signal.LAYOUT, (), geometry),
            _observation(Signal.INSTRUMENT, (2,), instrument),
            _observation(Signal.PEDALS, (2, PedalColor.BLUE), Press(pressed=True)),
        ]
        (fs,) = list(iter_frame_states(log))
        painted = lanes(fs, Catalog.load())
        assert painted[Lane(name=LaneName.PEDAL_BLUE, arm=2, column=2)] == "bipolar"


def _observation(signal: Signal, key: tuple, value: object) -> Observation:
    return Observation(time_s=1.0, frame=0, signal=signal, key=key, value=value, score=1.0)
