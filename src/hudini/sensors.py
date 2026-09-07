"""Declares, loads, and reads the sensors.

A sensor is four things: facts (:class:`SensorSpec`), loading
(``build``), where to look (``plan``), and what is there (``read``).
``plan`` cuts its own crops from the frame, and ``read`` returns one
:class:`Result` per task, by position. The ``after`` edges of the specs
form the scheduling graph. :func:`topological_levels`,
:func:`execution_levels`, and :func:`dependency_closure` answer its
questions. The registry rejects an invalid declaration at import. This
module imports torch.
"""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from pathlib import Path
from typing import ClassVar, NamedTuple, Protocol, Self, get_args

import msgspec
import numpy as np
from huggingface_hub import hf_hub_download

from hudini.catalog import Catalog
from hudini.cnn import ArmDigitClassifier, OffscreenDigitClassifier, StatusClassifier
from hudini.corrections import (
    ArmDigitUnique,
    CompleteLayout,
    CorrectionSpec,
    Debounce,
    FillGap,
    PedalRequiresAction,
    PedalRequiresInstrument,
)
from hudini.detection import Detector, load_detector
from hudini.hsv import (
    classify_instrument_status,
    classify_laser,
    classify_pedals,
    crop_pedal_action_label,
    warning_score,
)
from hudini.layout import detect_region, frame_geometry
from hudini.ocr import OCREngine
from hudini.schema import (
    ArmDigit,
    Banner,
    Box,
    Engine,
    FrameGeometry,
    Instrument,
    Laser,
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
    Value,
)
from hudini.views import LANE_PROJECTORS, FrameState, payload

WakeCondition = Callable[[FrameState, FrameState | None], bool]

# Instrument OCR windows, as fractions of pod width.
#
#   | arm | instrument name | pedals |
#   0     0.16              0.74     1
#         |<---- narrow --->|
#         |<--------- wide --------->|
ARM_CIRCLE_END_FRAC = 0.16
PEDAL_SECTION_START_FRAC = 0.74


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Weights that exist because one sensor exists.

    Attributes:
        name: the ``--model NAME=PATH`` token and the header key.
        file: the file name in the checkpoint repository.
        reports_threshold: whether the run's detector threshold shapes this
            checkpoint's behavior, so the header records it.
    """

    name: str
    file: str
    reports_threshold: bool = False


@dataclass(frozen=True, slots=True)
class SensorSpec:
    """One sensor's declared facts.

    Attributes:
        payload: the value struct this sensor emits.
        rate: frames per second past which reading more buys nothing.
        thrift_rate: the same sensor's ``--fast`` rate.
        greedy: read on every decoded frame; ``rate`` only sets the decode
            floor.
        after: the signals this sensor's ``plan`` needs from the same frame.
        wake: an off-cadence read condition on this frame and the last one.
        requires: the shared engines ``build`` takes from :class:`Engines`.
        corrections: this sensor's declared correction rules, in order.

    Raises:
        ValueError: a rate that is not positive, or a thrift rate above the
            rate.
    """

    name: Signal
    payload: type[msgspec.Struct]
    rate: float
    thrift_rate: float
    greedy: bool = False
    after: tuple[Signal, ...] = ()
    wake: WakeCondition | None = None
    requires: tuple[Engine, ...] = ()
    checkpoints: tuple[Checkpoint, ...] = ()
    corrections: tuple[CorrectionSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.rate <= 0 or self.thrift_rate <= 0:
            raise ValueError(f"{self.name}: rates must be positive")
        if self.thrift_rate > self.rate:
            raise ValueError(f"{self.name}: thrift_rate must not exceed rate")


class Crop(NamedTuple):
    """Located pixels: an RGB copy and the full-frame box it came from."""

    rgb: np.ndarray
    box: Box

    @classmethod
    def cut(cls, rgb: np.ndarray, box: Box) -> Self:
        """Copy ``box`` out of a full frame."""
        return cls(rgb=rgb[box.y : box.y + box.h, box.x : box.x + box.w].copy(), box=box)


class Task(NamedTuple):
    """One reading to make.

    Attributes:
        key: exactly the observation key.
        crops: the located crops the reading needs, zero for a reading a
            sensor can answer from geometry alone.
        context: sensor-private facts for ``read``.
    """

    key: tuple
    crops: tuple[Crop, ...]
    context: tuple = ()


@dataclass(frozen=True, slots=True)
class Result:
    """One task's reading: a value with a score.

    ``Result.absent()`` is a confirmed absence and clears the key; a read
    that returns None instead of a Result could not tell, and the key
    holds.
    """

    value: Value | None
    score: float

    @classmethod
    def absent(cls) -> "Result":
        """The reading that found nothing there."""
        return cls(value=None, score=0.0)


@dataclass(frozen=True, slots=True)
class Engines:
    """The shared engine instances sensors build against."""

    catalog: Catalog
    ocr: OCREngine | None = None


@dataclass(frozen=True, slots=True)
class LoadConfig:
    """Load-time facts each sensor reads while building."""

    device: str = "cpu"
    detector_threshold: float = 0.5
    batch_size: int = 1
    overrides: Mapping[str, Path] = field(default_factory=dict)


class Sensor(Protocol):
    """One sensor. The class declares, the instance reads."""

    spec: ClassVar[SensorSpec]

    @classmethod
    def build(cls, engines: Engines, config: LoadConfig) -> "Sensor | None":
        """A loaded instance, or None when something it needs is missing."""
        ...

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        """The readings this sensor wants from this frame."""
        ...

    def read(self, tasks: list[Task]) -> list[Result | None]:
        """One result per task, by position."""
        ...


def pods_with_box(fs: FrameState) -> Iterator[tuple[PodGeometry, Box]]:
    """The pods whose status bar segmented, each with its box: the ones a
    sensor can crop."""
    geometry = fs.geometry
    if geometry is None:
        return
    for pod in geometry.pods:
        if pod.box is not None:
            yield pod, pod.box


def wake_on_popup_change(current: FrameState, previous: FrameState | None) -> bool:
    """The popup sensor's wake: a column's popup count moved between two
    parsed frames."""
    if previous is None or current.geometry is None or previous.geometry is None:
        return False
    return current.geometry.popup_counts != previous.geometry.popup_counts


def wake_on_new_banner(current: FrameState, previous: FrameState | None) -> bool:
    """The banner sensor's wake: a banner is on screen that was not there
    a frame ago."""
    geometry = current.geometry
    if geometry is None or geometry.banner is None:
        return False
    before = previous.geometry if previous is not None else None
    return before is None or before.banner is None


class LayoutSensor:
    """Pixels to geometry, at level 0.

    The session feeds it directly: one task per frame whose crop is the
    full frame and whose context holds the cached region, or None to
    re-detect. An absent result means the frame parsed no UI.
    """

    spec = SensorSpec(
        name=Signal.LAYOUT,
        payload=FrameGeometry,
        rate=1.0,
        thrift_rate=1.0,
        greedy=True,
        corrections=(CompleteLayout(max_gap_s=1.0),),
    )

    @classmethod
    def build(cls, engines: Engines, config: LoadConfig) -> Self:
        return cls()

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        return []

    def read(self, tasks: list[Task]) -> list[Result | None]:
        results: list[Result | None] = []
        for task in tasks:
            (frame,) = task.crops
            (region,) = task.context
            if region is None:
                region = detect_region(frame.rgb)
            if region is None:
                results.append(Result.absent())
                continue
            geometry = frame_geometry(frame.rgb, region)
            if not any(pod.box is not None for pod in geometry.pods):
                results.append(Result.absent())
                continue
            results.append(Result(value=geometry, score=1.0))
        return results


class StatusSensor:
    """Active / inactive / warning per pod."""

    spec = SensorSpec(
        name=Signal.STATUS,
        payload=PodStatus,
        rate=1.0,
        thrift_rate=1.0,
        greedy=True,
        after=(Signal.LAYOUT,),
        checkpoints=(Checkpoint(name="status", file="camera_state_cnn.pt"),),
        corrections=(
            Debounce(Signal.STATUS, field="status", min_duration_s=0.15, neutral=Status.INACTIVE),
        ),
    )

    def __init__(self, classifier: StatusClassifier) -> None:
        self._classifier = classifier

    @classmethod
    def build(cls, engines: Engines, config: LoadConfig) -> Self:
        checkpoint = config.overrides.get("status") or resolve_checkpoint(
            checkpoint_file(cls.spec, "status")
        )
        return cls(classifier=StatusClassifier.load(checkpoint, device=config.device))

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        return [
            Task(key=(pod.column,), crops=(Crop.cut(rgb, box),), context=(pod.role,))
            for pod, box in pods_with_box(fs)
        ]

    def read(self, tasks: list[Task]) -> list[Result | None]:
        results: list[Result | None] = [None] * len(tasks)
        camera_indices = []
        for index, task in enumerate(tasks):
            (role,) = task.context
            crop = task.crops[0].rgb
            if role is not Role.CAMERA:
                status, score = classify_instrument_status(crop)
                results[index] = Result(value=PodStatus(status=status), score=score)
                continue
            # The camera pod renders warning like instrument pods, and the
            # binary CNN cannot say "warning", so the rule goes first.
            warning = warning_score(crop)
            if warning is not None:
                results[index] = Result(value=PodStatus(status=Status.WARNING), score=warning)
            else:
                camera_indices.append(index)
        verdicts = self._classifier.classify(
            [tasks[index].crops[0].rgb for index in camera_indices]
        )
        for index, (status, score) in zip(camera_indices, verdicts, strict=True):
            results[index] = Result(value=PodStatus(status=status), score=score)
        return results


class ArmSensor:
    """The arm digit 1-4 in a pod's left circle, camera pods included."""

    spec = SensorSpec(
        name=Signal.ARM,
        payload=ArmDigit,
        rate=1.0,
        thrift_rate=1.0,
        greedy=True,
        after=(Signal.LAYOUT,),
        checkpoints=(Checkpoint(name="arm", file="arm_digit_cnn.pt"),),
        corrections=(
            Debounce(Signal.ARM, field="digit", min_duration_s=0.15, neutral=None),
            ArmDigitUnique(),
        ),
    )

    def __init__(self, classifier: ArmDigitClassifier) -> None:
        self._classifier = classifier

    @classmethod
    def build(cls, engines: Engines, config: LoadConfig) -> Self:
        checkpoint = config.overrides.get("arm") or resolve_checkpoint(
            checkpoint_file(cls.spec, "arm")
        )
        return cls(classifier=ArmDigitClassifier.load(checkpoint, device=config.device))

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        return [
            Task(key=(pod.column,), crops=(Crop.cut(rgb, box),)) for pod, box in pods_with_box(fs)
        ]

    def read(self, tasks: list[Task]) -> list[Result | None]:
        verdicts = self._classifier.classify([task.crops[0].rgb for task in tasks])
        return [Result(value=ArmDigit(digit=digit), score=score) for digit, score in verdicts]


class InstrumentSensor:
    """The mounted instrument, OCR'd off the pod and matched to the catalog.

    Both crops drop the arm circle; the wide one keeps the pedal section,
    the narrow one stops before it. The pass with the better catalog match
    wins. A matched read updates, an empty pod clears, and unmatched text
    holds.
    """

    spec = SensorSpec(
        name=Signal.INSTRUMENT,
        payload=Instrument,
        rate=1.0,
        thrift_rate=0.2,
        after=(Signal.LAYOUT,),
        requires=(Engine.OCR, Engine.CATALOG),
    )

    def __init__(self, ocr: OCREngine, catalog: Catalog) -> None:
        self._ocr = ocr
        self._catalog = catalog

    @classmethod
    def build(cls, engines: Engines, config: LoadConfig) -> "Self | None":
        if engines.ocr is None:
            return None
        return cls(ocr=engines.ocr, catalog=engines.catalog)

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        return [
            Task(key=(pod.column,), crops=(Crop.cut(rgb, box),))
            for pod, box in pods_with_box(fs)
            if pod.role is not Role.CAMERA
        ]

    def read(self, tasks: list[Task]) -> list[Result | None]:
        results: list[Result | None] = []
        for task in tasks:
            pod = task.crops[0].rgb
            width = pod.shape[1]
            wide = pod[:, int(width * ARM_CIRCLE_END_FRAC) :]
            narrow = pod[
                :, int(width * ARM_CIRCLE_END_FRAC) : int(width * PEDAL_SECTION_START_FRAC)
            ]
            wide_read, narrow_read = self._ocr.read([wide, narrow])
            wide_match = self._catalog.match(wide_read.text)
            narrow_match = self._catalog.match(narrow_read.text)
            narrow_wins = (narrow_match.best.score if narrow_match else 0) >= (
                wide_match.best.score if wide_match else 0
            )
            match = narrow_match if narrow_wins else wide_match
            if match is not None:
                best = match.best
                results.append(
                    Result(
                        value=Instrument(
                            name=best.resolution.entry.display_name,
                            match=best.score,
                            raw=narrow_read.text if narrow_wins else wide_read.text,
                            window=OCRWindow.NARROW if narrow_wins else OCRWindow.WIDE,
                            reload_color=best.resolution.reload_color,
                        ),
                        score=best.score / 100,
                    )
                )
            elif wide_read.text or narrow_read.text:
                # Text neither window could match says nothing; hold.
                results.append(None)
            else:
                results.append(Result.absent())
        return results


class PedalsSensor:
    """Whether each pedal cell is pressed, from the HSV rule."""

    spec = SensorSpec(
        name=Signal.PEDALS,
        payload=Press,
        rate=10.0,
        thrift_rate=1.0,
        greedy=True,
        after=(Signal.LAYOUT,),
        corrections=(
            PedalRequiresInstrument(),
            PedalRequiresAction(),
            Debounce(Signal.PEDALS, field="pressed", min_duration_s=0.05, neutral=False),
        ),
    )

    @classmethod
    def build(cls, engines: Engines, config: LoadConfig) -> Self:
        return cls()

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        tasks = []
        for pod, box in pods_with_box(fs):
            if pod.role is Role.CAMERA:
                continue
            crop = Crop.cut(rgb, box)
            for color in (PedalColor.YELLOW, PedalColor.BLUE):
                tasks.append(Task(key=(pod.column, color), crops=(crop,)))
        return tasks

    def read(self, tasks: list[Task]) -> list[Result | None]:
        results: list[Result | None] = []
        for task in tasks:
            _column, color = task.key
            press = classify_pedals(task.crops[0].rgb)[color]
            results.append(Result(value=Press(pressed=press.pressed), score=press.score))
        return results


class PedalLabelSensor:
    """The action label of a pressed pedal, read only when the catalog
    cannot name the action itself."""

    spec = SensorSpec(
        name=Signal.PEDAL_LABEL,
        payload=PedalLabel,
        rate=10.0,
        thrift_rate=1.0,
        greedy=True,
        after=(Signal.LAYOUT, Signal.PEDALS, Signal.INSTRUMENT),
        requires=(Engine.OCR, Engine.CATALOG),
    )

    def __init__(self, ocr: OCREngine, catalog: Catalog) -> None:
        self._ocr = ocr
        self._catalog = catalog

    @classmethod
    def build(cls, engines: Engines, config: LoadConfig) -> "Self | None":
        if engines.ocr is None:
            return None
        return cls(ocr=engines.ocr, catalog=engines.catalog)

    def _catalog_cannot_name(self, instrument_name: str | None, color: PedalColor) -> bool:
        if instrument_name is None:
            return True
        resolution = self._catalog.resolve(instrument_name)
        if resolution is None or resolution.entry.pedals is None:
            return True
        return resolution.entry.pedals.of(color).press_dependent

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        boxes = {pod.column: box for pod, box in pods_with_box(fs)}
        tasks = []
        for obs in fs.fresh:
            if obs.signal is not Signal.PEDALS or not isinstance(obs.value, Press):
                continue
            if not obs.value.pressed:
                continue
            column, color = obs.key
            box = boxes.get(column)
            if box is None:
                continue
            instrument = fs.state.get((Signal.INSTRUMENT, column))
            name = payload(instrument, Instrument).name if instrument is not None else None
            if not self._catalog_cannot_name(name, PedalColor(color)):
                continue
            tasks.append(Task(key=(column, color), crops=(Crop.cut(rgb, box),)))
        return tasks

    def read(self, tasks: list[Task]) -> list[Result | None]:
        cells = []
        for task in tasks:
            _column, color = task.key
            cells.append(crop_pedal_action_label(task.crops[0].rgb, color))
        readable = [index for index, cell in enumerate(cells) if cell.size]
        lines = self._ocr.read_line([cells[index] for index in readable]) if readable else []
        results: list[Result | None] = [None] * len(tasks)
        for index, line in zip(readable, lines, strict=True):
            if line.text:
                results[index] = Result(value=PedalLabel(text=line.text), score=line.score)
        return results


class LaserSensor:
    """Whether the camera pod's laser readout shows ON, from the HSV rule."""

    spec = SensorSpec(
        name=Signal.LASER,
        payload=Laser,
        rate=10.0,
        thrift_rate=1.0,
        greedy=True,
        after=(Signal.LAYOUT,),
        corrections=(Debounce(Signal.LASER, field="on", min_duration_s=1.0, neutral=False),),
    )

    @classmethod
    def build(cls, engines: Engines, config: LoadConfig) -> Self:
        return cls()

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        return [
            Task(key=(pod.column,), crops=(Crop.cut(rgb, box),))
            for pod, box in pods_with_box(fs)
            if pod.role is Role.CAMERA
        ]

    def read(self, tasks: list[Task]) -> list[Result | None]:
        results: list[Result | None] = []
        for task in tasks:
            on, score = classify_laser(task.crops[0].rgb)
            results.append(Result(value=Laser(on=on), score=score))
        return results


class PopupsSensor:
    """Message text per popup segment, reported as one stack per column.

    One task per column carries every segment crop; an empty stack is a
    valid reading, so a column without segments still reports.
    """

    spec = SensorSpec(
        name=Signal.POPUPS,
        payload=PopupStack,
        rate=1.0,
        thrift_rate=0.2,
        after=(Signal.LAYOUT,),
        wake=wake_on_popup_change,
        requires=(Engine.OCR,),
    )

    def __init__(self, ocr: OCREngine) -> None:
        self._ocr = ocr

    @classmethod
    def build(cls, engines: Engines, config: LoadConfig) -> "Self | None":
        if engines.ocr is None:
            return None
        return cls(ocr=engines.ocr)

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        geometry = fs.geometry
        if geometry is None:
            return []
        return [
            Task(key=(pod.column,), crops=tuple(Crop.cut(rgb, box) for box in pod.popups))
            for pod in geometry.pods
        ]

    def read(self, tasks: list[Task]) -> list[Result | None]:
        flat = [crop.rgb for task in tasks for crop in task.crops]
        found = iter(self._ocr.read(flat)) if flat else iter(())
        results: list[Result | None] = []
        for task in tasks:
            messages = []
            for crop in task.crops:
                read = next(found)
                if read.text:
                    messages.append(PopupMessage(text=read.text, box=crop.box))
            results.append(Result(value=PopupStack(messages=tuple(messages)), score=1.0))
        return results


class BannerSensor:
    """Text of the system-status banner across the top of the region.

    A frame whose geometry shows no banner plans a cropless task, and the
    read confirms the absence.
    """

    spec = SensorSpec(
        name=Signal.BANNER,
        payload=Banner,
        rate=1.0,
        thrift_rate=0.2,
        after=(Signal.LAYOUT,),
        wake=wake_on_new_banner,
        requires=(Engine.OCR,),
    )

    def __init__(self, ocr: OCREngine) -> None:
        self._ocr = ocr

    @classmethod
    def build(cls, engines: Engines, config: LoadConfig) -> "Self | None":
        if engines.ocr is None:
            return None
        return cls(ocr=engines.ocr)

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        geometry = fs.geometry
        if geometry is None:
            return []
        if geometry.banner is None:
            return [Task(key=(), crops=())]
        return [Task(key=(), crops=(Crop.cut(rgb, geometry.banner),))]

    def read(self, tasks: list[Task]) -> list[Result | None]:
        results: list[Result | None] = []
        for task in tasks:
            if not task.crops:
                results.append(Result.absent())
                continue
            (read,) = self._ocr.read([task.crops[0].rgb])
            results.append(
                Result(value=Banner(text=read.text), score=read.score) if read.text else None
            )
        return results


def region_task(fs: FrameState, rgb: np.ndarray) -> list[Task]:
    """The whole active region as one task: what a frame-level detector
    reads."""
    geometry = fs.geometry
    if geometry is None:
        return []
    return [Task(key=(), crops=(Crop.cut(rgb, geometry.region),))]


def _offset_box(box: Box, origin: Box) -> Box:
    return Box(x=origin.x + box.x, y=origin.y + box.y, w=box.w, h=box.h)


HF_REPO = "nct-tso/hudini"
HF_REVISION = "d24a5fe60a4db1dcd45bea511716e6ab78d02563"


def resolve_checkpoint(file: str) -> Path:
    """The local path of a model checkpoint, downloaded on first use.

    Downloads the file from the HuggingFace Hub at the pinned revision
    and caches it. The revision is a full commit hash, so a cached file
    resolves without a network call.

    Raises:
        FileNotFoundError: the file is not cached and cannot be
            downloaded, for example offline.
    """
    return Path(hf_hub_download(repo_id=HF_REPO, filename=file, revision=HF_REVISION))


def checkpoint_file(spec: SensorSpec, name: str) -> str:
    """The declared file of one of the spec's checkpoints.

    Raises:
        ValueError: the spec declares no checkpoint of that name.
    """
    for checkpoint in spec.checkpoints:
        if checkpoint.name == name:
            return checkpoint.file
    raise ValueError(f"{spec.name}: no checkpoint named {name!r}")


class OffscreenSensor:
    """Hazard-stripe bars at the frame edge, with the arm digit per bar.

    Both stages stay inside ``read``: the detector locates the bars, then
    the digit model reads each bar's ends. Boxes come back in full-frame
    pixels through the crop's own box.
    """

    spec = SensorSpec(
        name=Signal.OFFSCREEN,
        payload=OffscreenBars,
        rate=5.0,
        thrift_rate=1.0,
        after=(Signal.LAYOUT,),
        checkpoints=(
            Checkpoint(name="offscreen", file="offscreen_state_rfdetr.pt", reports_threshold=True),
            Checkpoint(name="offscreen_digit", file="offscreen_digit_cnn.pt"),
        ),
    )

    REQUIRED_CLASSES: ClassVar[frozenset[str]] = frozenset({"active", "inactive"})

    def __init__(self, detector: Detector, digit_classifier: OffscreenDigitClassifier) -> None:
        self._detector = detector
        self._digit_classifier = digit_classifier

    @classmethod
    def build(cls, engines: Engines, config: LoadConfig) -> "Self | None":
        checkpoint = config.overrides.get("offscreen") or resolve_checkpoint(
            checkpoint_file(cls.spec, "offscreen")
        )
        detector = load_detector(
            checkpoint, threshold=config.detector_threshold, batched=config.batch_size > 1
        )
        if detector is None:
            return None
        missing = cls.REQUIRED_CLASSES - set(detector.labels)
        if missing:
            raise ValueError(f"{checkpoint}: detector lacks classes {sorted(missing)}")
        digit_checkpoint = config.overrides.get("offscreen_digit") or resolve_checkpoint(
            checkpoint_file(cls.spec, "offscreen_digit")
        )
        return cls(
            detector=detector,
            digit_classifier=OffscreenDigitClassifier.load(digit_checkpoint, device=config.device),
        )

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        return region_task(fs, rgb)

    def read(self, tasks: list[Task]) -> list[Result | None]:
        if not tasks:
            return []
        found = self._detector.detect([task.crops[0].rgb for task in tasks])
        results: list[Result | None] = []
        for task, detections in zip(tasks, found, strict=True):
            region = task.crops[0]
            digits = self._digit_classifier.classify(
                region.rgb, [detection.box for detection in detections]
            )
            bars = []
            for detection, (arm, _digit_score) in zip(detections, digits, strict=True):
                status = None
                if detection.label in ("active", "inactive"):
                    status = OffscreenBarStatus(detection.label)
                bars.append(
                    OffscreenBar(
                        score=detection.score,
                        box=_offset_box(detection.box, region.box),
                        status=status,
                        arm=arm,
                    )
                )
            results.append(Result(value=OffscreenBars(bars=tuple(bars)), score=1.0))
        return results


class ToolAssociationSensor:
    """Per-arm association badges, found on the whole active region."""

    spec = SensorSpec(
        name=Signal.TOOL_ASSOCIATION,
        payload=ToolBadges,
        rate=5.0,
        thrift_rate=1.0,
        after=(Signal.LAYOUT,),
        checkpoints=(
            Checkpoint(
                name="tool_association", file="tool_association_rfdetr.pt", reports_threshold=True
            ),
        ),
    )

    REQUIRED_CLASSES: ClassVar[frozenset[str]] = frozenset({"1", "2", "3", "4"})

    def __init__(self, detector: Detector) -> None:
        self._detector = detector

    @classmethod
    def build(cls, engines: Engines, config: LoadConfig) -> "Self | None":
        checkpoint = config.overrides.get("tool_association") or resolve_checkpoint(
            checkpoint_file(cls.spec, "tool_association")
        )
        detector = load_detector(
            checkpoint, threshold=config.detector_threshold, batched=config.batch_size > 1
        )
        if detector is None:
            return None
        missing = cls.REQUIRED_CLASSES - set(detector.labels)
        if missing:
            raise ValueError(f"{checkpoint}: detector lacks classes {sorted(missing)}")
        return cls(detector=detector)

    def plan(self, fs: FrameState, rgb: np.ndarray) -> list[Task]:
        return region_task(fs, rgb)

    def read(self, tasks: list[Task]) -> list[Result | None]:
        if not tasks:
            return []
        found = self._detector.detect([task.crops[0].rgb for task in tasks])
        results: list[Result | None] = []
        for task, detections in zip(tasks, found, strict=True):
            badges = tuple(
                ToolBadge(
                    score=detection.score,
                    box=_offset_box(detection.box, task.crops[0].box),
                    arm=int(detection.label) if detection.label.isdigit() else None,
                )
                for detection in detections
            )
            results.append(Result(value=ToolBadges(badges=badges), score=1.0))
        return results


REGISTRY: dict[Signal, type[Sensor]] = {
    sensor.spec.name: sensor
    for sensor in (
        LayoutSensor,
        StatusSensor,
        ArmSensor,
        InstrumentSensor,
        PedalsSensor,
        PedalLabelSensor,
        LaserSensor,
        PopupsSensor,
        BannerSensor,
        OffscreenSensor,
        ToolAssociationSensor,
    )
}


def topological_levels(after: Mapping[Signal, tuple[Signal, ...]]) -> list[list[Signal]]:
    """Topological levels of the ``after`` edges: execution order, each
    level sorted. Edges to names outside the mapping are ignored.

    Raises:
        ValueError: the edges contain a cycle.
    """
    sorter = TopologicalSorter(
        {name: [edge for edge in edges if edge in after] for name, edges in after.items()}
    )
    sorter.prepare()
    ordered: list[list[Signal]] = []
    while sorter.is_active():
        ready = sorted(sorter.get_ready())
        ordered.append(list(ready))
        sorter.done(*ready)
    return ordered


def execution_levels(selected: set[Signal]) -> list[list[Signal]]:
    """The selected signals' DAG levels, from the registry's specs."""
    return topological_levels({name: REGISTRY[name].spec.after for name in selected})


def dependency_closure(selected: set[Signal]) -> set[Signal]:
    """The selection closed over ``after`` edges, layout included."""
    closed = set(selected) | {Signal.LAYOUT}
    frontier = list(closed)
    while frontier:
        name = frontier.pop()
        for dependency in REGISTRY[name].spec.after:
            if dependency not in closed:
                closed.add(dependency)
                frontier.append(dependency)
    return closed


def _validate_registry() -> None:
    if set(REGISTRY) != set(Signal):
        missing = set(Signal) - set(REGISTRY)
        raise ValueError(f"signals without a sensor: {sorted(missing)}")
    if set(LANE_PROJECTORS) != set(Signal):
        raise ValueError("LANE_PROJECTORS does not cover every signal")
    payloads = set(get_args(Value))
    for sensor in REGISTRY.values():
        spec = sensor.spec
        if spec.payload not in payloads:
            raise ValueError(f"{spec.name}: payload {spec.payload.__name__} is not in Value")
        for correction in spec.corrections:
            if not isinstance(correction, Debounce | FillGap):
                continue
            target = REGISTRY[correction.signal].spec.payload
            addressable = correction.field in target.__struct_fields__ or isinstance(
                getattr(target, correction.field, None), property
            )
            if not addressable:
                raise ValueError(
                    f"{spec.name}: correction {correction.name} names unknown field "
                    f"{correction.field!r}"
                )
    execution_levels(set(Signal))  # raises on a cycle


_validate_registry()
