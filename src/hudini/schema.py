"""Defines the line types of hudini log format 1 and their encoder and decoders.

A log file has one JSON line per entry: a :class:`Header`,
:class:`Observation` lines, an optional :class:`Appendix`, and a
:class:`Footer`. This module owns every wire type, the encoder, the
decoders, and the generated JSON Schema. ``python -m hudini.schema``
prints the schema. All boxes are in full-frame pixels. This module does
not import torch or paddleocr.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

import msgspec

LOG_VERSION = 1
HEADER_MARK = "header"
APPENDIX_MARK = "appendix"
FOOTER_MARK = "footer"


class Signal(StrEnum):
    """The signal names: registry keys, spec names, and wire values."""

    LAYOUT = "layout"
    STATUS = "status"
    ARM = "arm"
    INSTRUMENT = "instrument"
    PEDALS = "pedals"
    PEDAL_LABEL = "pedal_label"
    LASER = "laser"
    POPUPS = "popups"
    BANNER = "banner"
    OFFSCREEN = "offscreen"
    TOOL_ASSOCIATION = "tool_association"


class Engine(StrEnum):
    """The shared engines a sensor can require."""

    OCR = "ocr"
    CATALOG = "catalog"


class PedalColor(StrEnum):
    """The two pedal cells of an instrument pod."""

    YELLOW = "yellow"
    BLUE = "blue"


class Role(StrEnum):
    """What a pod column controls."""

    CAMERA = "camera"
    INSTRUMENT = "instrument"


class Status(StrEnum):
    """What a status pod shows."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    WARNING = "warning"


class OffscreenBarStatus(StrEnum):
    """What an off-screen indicator bar shows. Bars cannot warn."""

    ACTIVE = "active"
    INACTIVE = "inactive"


class OCRWindow(StrEnum):
    """Which status-pod crop won an instrument read."""

    WIDE = "wide"
    NARROW = "narrow"


class Box(msgspec.Struct, frozen=True):
    """A rectangle in full-frame pixels."""

    x: int
    y: int
    w: int
    h: int


class PodGeometry(msgspec.Struct, frozen=True, omit_defaults=True):
    """One pod column's geometry.

    Attributes:
        column: 1-4, left to right.
        role: what the column controls.
        box: the status pod, or None when its segmentation failed.
        popups: the popup segment boxes, bottom-up from the pod.
    """

    column: int
    role: Role
    box: Box | None = None
    popups: tuple[Box, ...] = ()


class FrameGeometry(msgspec.Struct, frozen=True, tag="geometry", omit_defaults=True):
    """Everything one frame's pixels say about UI geometry. Holds no pixels.

    Attributes:
        region: the active (non-letterboxed) area.
        pods: one entry per column.
        banner: the system-status strip, when present.
    """

    region: Box
    pods: tuple[PodGeometry, ...] = ()
    banner: Box | None = None

    @property
    def popup_counts(self) -> tuple[int, ...]:
        """Each column's popup count, in pod order."""
        return tuple(len(pod.popups) for pod in self.pods)


class PodStatus(msgspec.Struct, frozen=True, tag="status"):
    """The status a pod shows."""

    status: Status


class ArmDigit(msgspec.Struct, frozen=True, tag="digit"):
    """The arm number 1-4 read from a pod's digit circle."""

    digit: int


class Instrument(msgspec.Struct, frozen=True, tag="instrument", omit_defaults=True):
    """A catalog-matched instrument read.

    Attributes:
        name: the catalog display name in the run's locale.
        match: the catalog match ratio, 0-100.
        raw: the OCR text the match was made from.
        window: which pod crop won the read.
        reload_color: canonical stapler reload color, when that alias matched.
    """

    name: str
    match: float
    raw: str
    window: OCRWindow
    reload_color: str | None = None


class Press(msgspec.Struct, frozen=True, tag="press"):
    """Whether a pedal cell is pressed."""

    pressed: bool


class PedalLabel(msgspec.Struct, frozen=True, tag="label"):
    """The action label text read from a pressed pedal cell."""

    text: str


class Laser(msgspec.Struct, frozen=True, tag="laser"):
    """Whether the camera pod's laser readout shows ON."""

    on: bool


class PopupMessage(msgspec.Struct, frozen=True):
    """One popup message and the segment box it was read from."""

    text: str
    box: Box


class PopupStack(msgspec.Struct, frozen=True, tag="popups", omit_defaults=True):
    """A column's popup messages, bottom-up. An empty stack is a valid reading."""

    messages: tuple[PopupMessage, ...] = ()

    @property
    def texts(self) -> tuple[str, ...]:
        """The message texts in stack order: the stack's identity without boxes."""
        return tuple(message.text for message in self.messages)


class Banner(msgspec.Struct, frozen=True, tag="banner"):
    """The system-status banner text."""

    text: str


class OffscreenBar(msgspec.Struct, frozen=True, omit_defaults=True):
    """One detected off-screen indicator bar.

    Attributes:
        score: detector confidence, 0-1.
        box: the bar.
        status: what the bar shows, or None when the detector class is unknown.
        arm: the arm number read off the bar's ends, or None on abstain.
    """

    score: float
    box: Box
    status: OffscreenBarStatus | None = None
    arm: int | None = None


class OffscreenBars(msgspec.Struct, frozen=True, tag="offscreen", omit_defaults=True):
    """All off-screen indicator bars found on one frame."""

    bars: tuple[OffscreenBar, ...] = ()


class ToolBadge(msgspec.Struct, frozen=True, omit_defaults=True):
    """One detected tool-association badge.

    Attributes:
        score: detector confidence, 0-1.
        box: the badge.
        arm: the arm number of the badge class, or None when unreadable.
    """

    score: float
    box: Box
    arm: int | None = None


class ToolBadges(msgspec.Struct, frozen=True, tag="badges", omit_defaults=True):
    """All tool-association badges found on one frame."""

    badges: tuple[ToolBadge, ...] = ()


Value = (
    FrameGeometry
    | PodStatus
    | ArmDigit
    | Instrument
    | Press
    | PedalLabel
    | Laser
    | PopupStack
    | Banner
    | OffscreenBars
    | ToolBadges
)


class Observation(
    msgspec.Struct,
    frozen=True,
    rename={"time_s": "t", "frame": "f", "signal": "s", "key": "k", "value": "v", "score": "c"},
):
    """One fact one sensor read at one time.

    ``value`` carries the three-way contract: a payload updates the key, None
    is confirmed absence and clears the key, and no observation at all means
    the sensor could not tell.

    Attributes:
        time_s: presentation time in seconds.
        frame: source frame index.
        key: sensor-private identity, e.g. ``(column,)`` or ``(column, color)``;
            empty for a frame-level signal.
        score: confidence in the value, 0-1.
    """

    time_s: float
    frame: int
    signal: Signal
    key: tuple[int | PedalColor, ...]
    value: Value | None
    score: float


class VideoInfo(msgspec.Struct, frozen=True, omit_defaults=True):
    """Container facts of the input video. Names follow ffprobe where common.

    Attributes:
        sha256: input file hash, or None when hashing was skipped.
        frame_rate: average rate in frames per second, when the container
            states one.
        duration_s: duration in seconds, when the container states one.
    """

    filename: str
    width: int
    height: int
    codec: str
    sha256: str | None = None
    frame_rate: float | None = None
    duration_s: float | None = None


class ModelInfo(msgspec.Struct, frozen=True, omit_defaults=True):
    """Provenance of one checkpoint.

    Attributes:
        threshold: the run's confidence floor, for detector checkpoints.
    """

    file: str
    sha256: str
    threshold: float | None = None


class EngineInfo(msgspec.Struct, frozen=True, omit_defaults=True):
    """One shared engine and the signals that read through it."""

    used_by: tuple[Signal, ...]
    model: str | None = None
    locale: str | None = None


CorrectionSetting = str | float | int | bool | tuple[str, ...] | None


class Header(msgspec.Struct, frozen=True, tag_field="hudini", tag=HEADER_MARK, omit_defaults=True):
    """First line of a log.

    The line starts with ``"hudini": "header"``: the key is the file's
    magic, the value names the line kind.

    Attributes:
        log_version: the log format version.
        version: the hudini package version that wrote the log.
        created_at: ISO timestamp of the parse start.
        requested: the signal tokens the user asked for.
        signals: the signals that ran, after dependency expansion.
        rates: the configured rate per signal, in frames per second.
        corrections: each declared correction's fields, name included.
            Informative defaults of ``version``; settings given at read
            time win.
    """

    log_version: int
    version: str
    created_at: str
    video: VideoInfo
    requested: tuple[str, ...]
    signals: tuple[Signal, ...]
    rates: dict[Signal, float]
    batch_size: int
    device: str
    models: dict[str, ModelInfo] = {}
    engines: dict[Engine, EngineInfo] = {}
    corrections: tuple[dict[str, CorrectionSetting], ...] = ()


class Appendix(
    msgspec.Struct, frozen=True, tag_field="hudini", tag=APPENDIX_MARK, omit_defaults=True
):
    """Second-to-last line of a finished log: cached derived views.

    The line starts with ``"hudini": "appendix"``. Supplementary
    material, like a document appendix: derived from the body,
    skippable without losing the record. A reader uses the intervals
    only when ``intervals_version`` matches the version it knows, and
    derives from the body otherwise. The content must equal a
    recomputation from the body under the header's correction rules.

    Attributes:
        intervals_version: version of the interval shape and the lane
            vocabulary.
        intervals: the corrected intervals as plain JSON objects.
    """

    intervals_version: int
    intervals: list[dict]


class Footer(msgspec.Struct, frozen=True, tag_field="hudini", tag=FOOTER_MARK, omit_defaults=True):
    """Last line of a log: the completeness seal, plus a digest of the body.

    The line starts with ``"hudini": "footer"``. A file without a footer
    is a partial parse. Every field except ``runtime_s`` is derivable
    from the body, and the digest must equal a recomputation from it.

    Attributes:
        observations: body line count.
        frames_sampled: count of ``layout`` observations.
        achieved_fps: the sampling rate the source could deliver.
        runtime_s: wall-clock duration of the parse in seconds.
        summary_version: version of the embedded summary shape.
        summary: the case summary as a plain JSON object. Convert it to
            the typed shape only when ``summary_version`` matches the
            version you know, so an old shape cannot break the read.
    """

    observations: int
    frames_sampled: int
    achieved_fps: float
    runtime_s: float
    summary_version: int | None = None
    summary: dict | None = None


@dataclass(frozen=True, slots=True)
class Log:
    """The in-memory form of one log file.

    Iterates as its observations, so views accept a Log or a plain sequence
    of observations through one signature.

    Attributes:
        appendix: None for a partial file or a log written without one.
        footer: None for a partial file.
    """

    header: Header
    observations: tuple[Observation, ...]
    appendix: "Appendix | None" = None
    footer: Footer | None = None

    def __iter__(self) -> Iterator[Observation]:
        return iter(self.observations)

    def __len__(self) -> int:
        return len(self.observations)


_encoder = msgspec.json.Encoder()
_header_decoder = msgspec.json.Decoder(Header)
_observation_decoder = msgspec.json.Decoder(Observation)
_appendix_decoder = msgspec.json.Decoder(Appendix)
_footer_decoder = msgspec.json.Decoder(Footer)


def encode(entry: Header | Observation | Appendix | Footer) -> bytes:
    """One log entry as its JSON line, without a trailing newline."""
    return _encoder.encode(entry)


def decode_header(line: bytes | str) -> Header:
    """Decode line 1 of a log.

    Raises:
        ValueError: the line is not a header, or its log version is not
            ``LOG_VERSION``.
    """
    header = _header_decoder.decode(line)
    if header.log_version != LOG_VERSION:
        raise ValueError(f"unsupported log version {header.log_version}; supported: {LOG_VERSION}")
    return header


def decode_appendix(line: bytes | str) -> Appendix:
    """Decode the appendix line of a finished log.

    Raises:
        ValueError: the line is not an appendix.
    """
    return _appendix_decoder.decode(line)


def decode_observation(line: bytes | str) -> Observation:
    """Decode one body line.

    Raises:
        ValueError: the line is not a valid observation.
    """
    return _observation_decoder.decode(line)


def decode_footer(line: bytes | str) -> Footer:
    """Decode the last line of a finished log.

    Raises:
        ValueError: the line is not a footer.
    """
    return _footer_decoder.decode(line)


def json_schema() -> dict:
    """The JSON Schema of the three line shapes, generated from the structs.

    The shapes are listed under ``anyOf`` rather than a tagged union.
    The ``hudini`` key names the line kind: ``header``, ``appendix``,
    or ``footer``. A line without it is an observation.
    """
    shapes, definitions = msgspec.json.schema_components((Header, Observation, Appendix, Footer))
    return {"$defs": definitions, "anyOf": list(shapes)}


if __name__ == "__main__":
    print(json.dumps(json_schema(), indent=2))
