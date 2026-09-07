"""Unit tests for the privacy screen in hudini.privacy.

Loader behavior is tested against synthetic template files; one smoke
test loads the real bundled sensitive_popups.json. Screen and mask
behavior is tested over hand-built observation logs.
"""

import json
from collections.abc import Callable
from pathlib import Path

import msgspec
import pytest

from hudini.privacy import (
    MASK_PREFIX,
    MASK_RULE,
    MIN_TAIL_LENGTH,
    PopupTemplate,
    load_templates,
    mask_patches,
    matched_texts,
    screen,
)
from hudini.schema import (
    Box,
    FrameGeometry,
    Observation,
    PopupMessage,
    PopupStack,
    Signal,
)
from hudini.views import PatchAction, apply_patches

GEOMETRY = FrameGeometry(region=Box(x=0, y=0, w=1920, h=1080))
POPUP_BOX = Box(x=480, y=820, w=200, h=40)
OTHER_BOX = Box(x=480, y=770, w=200, h=40)

DE_TEMPLATE = PopupTemplate(
    id="energy_preset_applied",
    locale="de",
    text="{surgeon}s 'preset {x}' energievoreinstellung angewendet",
    tail="energievoreinstellung angewendet",
)
EN_TEMPLATE = PopupTemplate(
    id="energy_preset_applied",
    locale="en",
    text="{surgeon}'s '{x}' energy preset applied",
    tail="energy preset applied",
)
TEMPLATES = (DE_TEMPLATE, EN_TEMPLATE)

PII_TEXT = "claas' 'preset lap' energievoreinstellung angewendet"
HARMLESS_TEXT = "instrument wird ausserhalb des sichtfelds bewegt"


@pytest.fixture
def write_templates(tmp_path: Path) -> Callable[..., Path]:
    """Write a template file from plain entry dicts."""

    def _write(entries: list[dict]) -> Path:
        file = tmp_path / "templates.json"
        file.write_text(json.dumps(entries))
        return file

    return _write


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
def make_log(make_observation: Callable[..., Observation]) -> Callable[..., list[Observation]]:
    """A log from per-frame popup texts: layout each second, one popup
    stack for column 2 where texts are given (None means no read)."""

    def _make(texts_by_second: list[tuple[str, ...] | None]) -> list[Observation]:
        log: list[Observation] = []
        for second, texts in enumerate(texts_by_second):
            time_s = float(second)
            log.append(make_observation(Signal.LAYOUT, GEOMETRY, time_s=time_s, frame=second))
            if texts is None:
                continue
            messages = tuple(PopupMessage(text=text, box=POPUP_BOX) for text in texts)
            log.append(
                make_observation(
                    Signal.POPUPS,
                    PopupStack(messages=messages),
                    key=(2,),
                    time_s=time_s,
                    frame=second,
                )
            )
        return log

    return _make


class TestLoadTemplates:
    def test_flattens_entries_per_locale_and_derives_tails(self, write_templates):
        file = write_templates(
            [
                {
                    "id": "energy_preset_applied",
                    "templates": {
                        "de": "{surgeon}s 'preset {x}' energievoreinstellung angewendet",
                        "en": "{surgeon}'s '{x}' energy preset applied",
                    },
                }
            ]
        )
        templates = load_templates(file)
        assert [(template.id, template.locale, template.tail) for template in templates] == [
            ("energy_preset_applied", "de", "energievoreinstellung angewendet"),
            ("energy_preset_applied", "en", "energy preset applied"),
        ]

    @pytest.mark.parametrize(
        ("entries", "message"),
        [
            (
                [
                    {"id": "a", "templates": {"de": "{x} energievoreinstellung angewendet"}},
                    {"id": "a", "templates": {"en": "{x} energy preset applied"}},
                ],
                "duplicate template id",
            ),
            (
                [
                    {"id": "a", "templates": {"de": "{x} energievoreinstellung angewendet"}},
                    {"id": "b", "templates": {"de": "{y} energievoreinstellung angewendet"}},
                ],
                "same tail",
            ),
            (
                [{"id": "a", "templates": {"de": "{x} Energievoreinstellung angewendet"}}],
                "lowercase",
            ),
            ([{"id": "a", "templates": {"de": "energievoreinstellung angewendet"}}], "placeholder"),
            ([{"id": "a", "templates": {"de": "{x} kurz"}}], "over-matches"),
        ],
    )
    def test_rejects_an_invalid_file(self, write_templates, entries, message):
        with pytest.raises(ValueError, match=message):
            load_templates(write_templates(entries))

    def test_rejects_a_template_ending_in_a_placeholder(self, write_templates):
        file = write_templates([{"id": "a", "templates": {"de": "energiemodus fuer {surgeon}"}}])
        with pytest.raises(ValueError, match="over-matches"):
            load_templates(file)

    def test_a_malformed_placeholder_raises(self, write_templates):
        file = write_templates([{"id": "a", "templates": {"de": "abc { def angewendet xyz"}}])
        with pytest.raises(ValueError):
            load_templates(file)

    def test_an_unknown_field_raises(self, write_templates, tmp_path):
        file = tmp_path / "templates.json"
        file.write_text(json.dumps([{"id": "a", "templates": {}, "notes": "no"}]))
        with pytest.raises(msgspec.ValidationError):
            load_templates(file)

    def test_the_bundled_file_is_valid(self):
        templates = load_templates()
        assert any(template.id == "energy_preset_applied" for template in templates)
        assert all(len(template.tail) >= MIN_TAIL_LENGTH for template in templates)


class TestScreen:
    def test_an_episode_rounds_outward_to_the_sample_grid(self, make_log):
        log = make_log([(), (PII_TEXT,), (PII_TEXT,), ()])
        (finding,) = screen(log, TEMPLATES)
        assert finding.template == DE_TEMPLATE
        assert finding.column == 2
        assert finding.start_s == 0.0
        assert finding.end_s == 3.0
        assert [timed.time_s for timed in finding.boxes] == [1.0, 2.0]

    def test_ocr_noise_still_matches(self, make_log):
        noisy = "claas' 'preset lap' energievoreinstel1ung angewendet"
        log = make_log([(noisy,)])
        (finding,) = screen(log, TEMPLATES)
        assert finding.template == DE_TEMPLATE

    def test_a_non_matching_read_splits_episodes(self, make_log):
        log = make_log([(PII_TEXT,), (), (PII_TEXT,), ()])
        first, second = screen(log, TEMPLATES)
        assert first.end_s <= second.start_s

    def test_a_confirmed_absence_closes_the_episode(self, make_observation):
        stack = PopupStack(messages=(PopupMessage(text=PII_TEXT, box=POPUP_BOX),))
        log = [
            make_observation(Signal.LAYOUT, GEOMETRY, time_s=0.0, frame=0),
            make_observation(Signal.POPUPS, stack, key=(2,), time_s=0.0, frame=0),
            make_observation(Signal.LAYOUT, GEOMETRY, time_s=1.0, frame=1),
            make_observation(Signal.POPUPS, None, key=(2,), time_s=1.0, frame=1),
            make_observation(Signal.LAYOUT, GEOMETRY, time_s=2.0, frame=2),
        ]
        (finding,) = screen(log, TEMPLATES)
        assert finding.end_s == 1.0

    def test_an_episode_at_the_log_end_closes_one_spacing_past(self, make_log):
        log = make_log([(), (PII_TEXT,)])
        (finding,) = screen(log, TEMPLATES)
        assert finding.end_s == 2.0

    def test_all_locales_match_at_once(self, make_log):
        log = make_log([("dr smith's 'cut' energy preset applied",)])
        (finding,) = screen(log, TEMPLATES)
        assert finding.template == EN_TEMPLATE

    def test_a_clean_log_gives_no_findings(self, make_log):
        log = make_log([(HARMLESS_TEXT,), (HARMLESS_TEXT,), ()])
        assert screen(log, TEMPLATES) == []

    def test_the_union_box_covers_every_matched_box(self, make_observation):
        low = PopupMessage(text=PII_TEXT, box=Box(x=480, y=820, w=200, h=40))
        high = PopupMessage(text=PII_TEXT, box=Box(x=470, y=770, w=200, h=40))
        log = [
            make_observation(Signal.LAYOUT, GEOMETRY, time_s=0.0, frame=0),
            make_observation(
                Signal.POPUPS, PopupStack(messages=(low, high)), key=(2,), time_s=0.0, frame=0
            ),
        ]
        (finding,) = screen(log, TEMPLATES)
        assert finding.union_box == Box(x=470, y=770, w=210, h=90)

    def test_matched_texts_returns_the_raw_text_of_one_finding(self, make_log):
        log = make_log([(PII_TEXT, HARMLESS_TEXT), (PII_TEXT,), ()])
        (finding,) = screen(log, TEMPLATES)
        assert matched_texts(log, finding) == (PII_TEXT,)


class TestMaskPatches:
    def test_a_matched_message_is_masked_and_keeps_its_box(self, make_observation):
        stack = PopupStack(
            messages=(
                PopupMessage(text=PII_TEXT, box=POPUP_BOX),
                PopupMessage(text=HARMLESS_TEXT, box=OTHER_BOX),
            )
        )
        log = [make_observation(Signal.POPUPS, stack, key=(2,), time_s=1.0, frame=1)]
        (patch,) = mask_patches(log, TEMPLATES)
        assert patch.action is PatchAction.SET
        assert patch.by == MASK_RULE
        assert (patch.start_s, patch.end_s) == (1.0, 1.0)
        assert patch.value.messages == (
            PopupMessage(text=f"{MASK_PREFIX}energy_preset_applied", box=POPUP_BOX),
            PopupMessage(text=HARMLESS_TEXT, box=OTHER_BOX),
        )

    def test_applying_the_patches_removes_the_raw_text(self, make_log):
        log = make_log([(PII_TEXT,), (PII_TEXT,)])
        corrected = apply_patches(log, mask_patches(log, TEMPLATES))
        texts = [
            message.text
            for obs in corrected
            if isinstance(obs.value, PopupStack)
            for message in obs.value.messages
        ]
        assert PII_TEXT not in texts
        assert texts == [f"{MASK_PREFIX}energy_preset_applied"] * 2

    def test_a_clean_log_gives_no_patches(self, make_log):
        log = make_log([(HARMLESS_TEXT,), ()])
        assert mask_patches(log, TEMPLATES) == []
