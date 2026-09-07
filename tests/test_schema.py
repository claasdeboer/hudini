"""Unit tests for the log contract in hudini.schema.

msgspec is the substrate under test here on purpose: these tests pin the
wire contract (field names, tags, the three-way value semantics), which is
exactly what the struct declarations promise.
"""

import json
from collections.abc import Callable

import pytest

from hudini.schema import (
    FOOTER_MARK,
    LOG_VERSION,
    ArmDigit,
    Banner,
    Box,
    Engine,
    EngineInfo,
    Footer,
    FrameGeometry,
    Header,
    Instrument,
    Log,
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
    VideoInfo,
    decode_footer,
    decode_header,
    decode_observation,
    encode,
    json_schema,
)


@pytest.fixture
def make_observation() -> Callable[..., Observation]:
    """Build an observation with overridable fields."""

    def _make(
        signal: Signal = Signal.PEDALS,
        key: tuple = (2, PedalColor.BLUE),
        value: object = Press(pressed=True),
        score: float = 0.99,
        time_s: float = 98.433,
        frame: int = 2953,
    ) -> Observation:
        return Observation(
            time_s=time_s, frame=frame, signal=signal, key=key, value=value, score=score
        )

    return _make


@pytest.fixture
def make_header() -> Callable[..., Header]:
    """Build a header with overridable fields."""

    def _make(log_version: int = LOG_VERSION) -> Header:
        return Header(
            log_version=log_version,
            version="1.0.0",
            created_at="2026-08-07T14:12:03+02:00",
            video=VideoInfo(
                filename="case_002.mp4",
                width=1920,
                height=1080,
                codec="h264",
                sha256="9f2c",
                frame_rate=60.0,
                duration_s=10798.5,
            ),
            requested=("pedals",),
            signals=(Signal.ARM, Signal.INSTRUMENT, Signal.PEDAL_LABEL, Signal.PEDALS),
            rates={
                Signal.ARM: 1.0,
                Signal.INSTRUMENT: 1.0,
                Signal.PEDAL_LABEL: 10.0,
                Signal.PEDALS: 10.0,
            },
            batch_size=1,
            device="cuda:0",
            engines={
                Engine.OCR: EngineInfo(
                    used_by=(Signal.INSTRUMENT, Signal.PEDAL_LABEL), model="pp-ocrv6-tiny"
                ),
                Engine.CATALOG: EngineInfo(
                    used_by=(Signal.INSTRUMENT, Signal.PEDAL_LABEL), locale="en"
                ),
            },
            corrections=(
                {
                    "name": "debounce:pedals",
                    "field": "pressed",
                    "min_duration_s": 0.15,
                    "neutral": False,
                },
            ),
        )

    return _make


class TestObservationWire:
    def test_wire_uses_the_short_field_names(self, make_observation: Callable[..., Observation]):
        line = json.loads(encode(make_observation()))
        assert set(line) == {"t", "f", "s", "k", "v", "c"}

    def test_roundtrip_preserves_every_field(self, make_observation: Callable[..., Observation]):
        original = make_observation()
        assert decode_observation(encode(original)) == original

    def test_a_null_value_is_encoded_not_omitted(
        self, make_observation: Callable[..., Observation]
    ):
        # Confirmed absence must stay visible on the wire; omission means
        # "could not tell" and is expressed by writing no line at all.
        line = json.loads(encode(make_observation(value=None)))
        assert line["v"] is None
        assert decode_observation(encode(make_observation(value=None))).value is None

    def test_pedal_color_key_elements_decode_to_the_enum(
        self, make_observation: Callable[..., Observation]
    ):
        decoded = decode_observation(encode(make_observation(key=(2, PedalColor.BLUE))))
        assert decoded.key == (2, PedalColor.BLUE)

    def test_an_unknown_signal_name_is_a_loud_error(self):
        line = b'{"t":0.0,"f":0,"s":"nonsense","k":[],"v":null,"c":0.0}'
        with pytest.raises(ValueError):
            decode_observation(line)


class TestPayloadUnion:
    @pytest.mark.parametrize(
        "value",
        [
            FrameGeometry(
                region=Box(x=320, y=0, w=1280, h=1024),
                pods=(
                    PodGeometry(
                        column=1,
                        role=Role.CAMERA,
                        box=Box(x=340, y=900, w=200, h=48),
                        popups=(Box(x=340, y=850, w=200, h=40),),
                    ),
                ),
                banner=Box(x=600, y=0, w=700, h=40),
            ),
            PodStatus(status=Status.ACTIVE),
            ArmDigit(digit=3),
            Instrument(
                name="Maryland Bipolar Forceps",
                match=94.0,
                raw="maryland bipolar",
                window=OCRWindow.NARROW,
            ),
            Press(pressed=True),
            PedalLabel(text="STRONG"),
            PopupStack(messages=(PopupMessage(text="check arm 2", box=Box(x=1, y=2, w=3, h=4)),)),
            Banner(text="table motion in progress"),
            OffscreenBars(
                bars=(
                    OffscreenBar(
                        score=0.91,
                        box=Box(x=0, y=100, w=20, h=300),
                        status=OffscreenBarStatus.ACTIVE,
                        arm=4,
                    ),
                )
            ),
            ToolBadges(badges=(ToolBadge(score=0.88, box=Box(x=5, y=6, w=7, h=8), arm=2),)),
        ],
        ids=lambda value: type(value).__name__,
    )
    def test_every_payload_roundtrips_through_an_observation(
        self, make_observation: Callable[..., Observation], value: object
    ):
        original = make_observation(value=value)
        assert decode_observation(encode(original)) == original

    def test_struct_payloads_carry_a_type_tag_on_the_wire(
        self, make_observation: Callable[..., Observation]
    ):
        line = json.loads(encode(make_observation(value=Press(pressed=False))))
        assert line["v"]["type"] == "press"

    def test_omitted_defaults_decode_back_to_none(
        self, make_observation: Callable[..., Observation]
    ):
        instrument = Instrument(
            name="Permanent Cautery Hook", match=88.0, raw="cautery hook", window=OCRWindow.WIDE
        )
        line = json.loads(encode(make_observation(value=instrument)))
        assert "reload_color" not in line["v"]
        assert decode_observation(encode(make_observation(value=instrument))).value == instrument


class TestHeader:
    def test_roundtrip_preserves_enum_keyed_rates(self, make_header: Callable[..., Header]):
        original = make_header()
        assert decode_header(encode(original)) == original

    def test_the_first_key_identifies_the_format(self, make_header: Callable[..., Header]):
        line = json.loads(encode(make_header()))
        assert line["hudini"] == "header"
        assert line["log_version"] == LOG_VERSION

    def test_an_unsupported_format_version_raises(self, make_header: Callable[..., Header]):
        with pytest.raises(ValueError):
            decode_header(encode(make_header(log_version=LOG_VERSION + 1)))


class TestFooter:
    def test_roundtrip(self):
        original = Footer(
            observations=1727312,
            frames_sampled=107985,
            achieved_fps=10.0,
            runtime_s=5981.2,
        )
        assert json.loads(encode(original))["hudini"] == FOOTER_MARK
        assert decode_footer(encode(original)) == original

    def test_a_non_footer_line_raises(self, make_header: Callable[..., Header]):
        with pytest.raises(ValueError):
            decode_footer(encode(make_header()))


class TestLog:
    def test_iterates_as_its_observations(
        self, make_header: Callable[..., Header], make_observation: Callable[..., Observation]
    ):
        observations = (make_observation(time_s=1.0), make_observation(time_s=2.0))
        log = Log(header=make_header(), observations=observations)
        assert tuple(log) == observations
        assert len(log) == 2

    def test_footer_defaults_to_none_for_a_partial_file(self, make_header: Callable[..., Header]):
        assert Log(header=make_header(), observations=()).footer is None


class TestStatusVocabulary:
    def test_bar_status_and_pod_status_agree_at_the_value_level(self):
        # Both are StrEnums, so cross-enum comparison by value works.
        assert OffscreenBarStatus.ACTIVE == Status.ACTIVE
        assert OffscreenBarStatus.INACTIVE == Status.INACTIVE

    def test_warning_is_not_a_bar_status(self):
        assert {member.value for member in OffscreenBarStatus} == {"active", "inactive"}


def test_json_schema_covers_all_four_line_shapes():
    schema = json_schema()
    definitions = schema.get("$defs", {})
    assert {"Header", "Observation", "Footer"} <= set(definitions)


def test_signal_members_are_their_wire_strings():
    assert Signal.PEDAL_LABEL == "pedal_label"
    assert Signal("status") is Signal.STATUS
