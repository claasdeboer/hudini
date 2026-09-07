"""Loads the models once and parses many videos.

:class:`Parser` loads the engines and sensors of a signal selection at
construction. It then serves any number of videos. ``parse_video``
writes the log file and returns the :class:`Log` that ``storage.load``
gives. ``iter_observations`` streams without a file. ``session`` gives a
fresh :class:`Session` for any frame source. This module imports torch.
"""

import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from hudini.corrections import CorrectionSpec, correct
from hudini.ocr import ModelSize, OCREngine
from hudini.schema import (
    LOG_VERSION,
    CorrectionSetting,
    Engine,
    EngineInfo,
    Header,
    Log,
    ModelInfo,
    Observation,
    Signal,
    VideoInfo,
)
from hudini.sensors import (
    REGISTRY,
    Catalog,
    Engines,
    LoadConfig,
    Sensor,
    dependency_closure,
    resolve_checkpoint,
)
from hudini.session import Session
from hudini.storage import LogWriter, sha256_of
from hudini.video import Frame, iter_frames, probe_video
from hudini.views import (
    INTERVALS_VERSION,
    SUMMARY_VERSION,
    interval_view,
    intervals_to_wire,
    summarize,
)

logger = logging.getLogger(__name__)

Progress = Callable[[float], None]


def _as_signal(token: Signal | str) -> Signal:
    if isinstance(token, Signal):
        return token
    members = {member.value: member for member in Signal}
    if token not in members:
        raise ValueError(f"unknown signal {token!r}; valid: {sorted(members)}")
    return members[token]


def _package_version() -> str:
    return metadata.version("hudini")


def resolve_device() -> str:
    """The one place the runtime device is decided: cuda when available,
    else cpu. Everything below the parser takes the answer as an
    argument."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def log_path(video: Path, output: Path | str | None) -> Path:
    """The log file a parse of ``video`` writes: ``output`` itself when it
    is a file path, ``<output>/<stem>.hudini.jsonl.gz`` for a directory,
    and the working directory when ``output`` is None."""
    log_name = f"{video.stem}.hudini.jsonl.gz"
    if output is None:
        return Path.cwd() / log_name
    output = Path(output)
    if output.is_dir():
        return output / log_name
    return output


def _with_progress(frames: Iterator[Frame], progress: Progress) -> Iterator[Frame]:
    for frame in frames:
        progress(frame.time_s)
        yield frame


class Parser:
    """An immutable bundle of loaded models and the selection they serve.

    Construction loads every engine and checkpoint the selection needs,
    so one instance serves many videos. A signal whose models are not
    available is disabled with a logged warning, along with anything
    that depends on it.

    Args:
        signals: the signals to read, as ``Signal`` members or their
            string values; None selects every signal. Dependencies are
            added automatically.
        ocr_model_size: PP-OCRv6 size for the OCR engine, when one loads.
        locale: catalog locale of the recordings.
        detector_threshold: confidence floor for the RF-DETR detectors.
        models: checkpoint overrides by checkpoint name.
        batch_size: frames handed to the model levels at once.

    Raises:
        ValueError: an unknown signal name, an unknown checkpoint name,
            or an unknown locale.
    """

    def __init__(
        self,
        signals: Iterable[Signal | str] | None = None,
        *,
        ocr_model_size: ModelSize | str = ModelSize.TINY,
        locale: str = "en",
        detector_threshold: float = 0.5,
        models: Mapping[str, Path | str] | None = None,
        batch_size: int = 1,
    ) -> None:
        self._requested = ("all",) if signals is None else tuple(str(token) for token in signals)
        requested = set(Signal) if signals is None else {_as_signal(token) for token in signals}
        selected = dependency_closure(requested)
        self._overrides = self._checkpoint_overrides(selected, models)
        self._batch_size = batch_size
        self._detector_threshold = detector_threshold
        self._device = resolve_device()
        self._catalog = Catalog.load(locale=locale)
        self._ocr = self._load_ocr(selected, ocr_model_size)
        config = LoadConfig(
            device=self._device,
            detector_threshold=detector_threshold,
            batch_size=batch_size,
            overrides=self._overrides,
        )
        self._sensors = self._build_sensors(selected, config)
        self.signals: tuple[Signal, ...] = tuple(
            name for name in Signal if name in self._sensors and name is not Signal.LAYOUT
        )

    def _checkpoint_overrides(
        self, selected: set[Signal], models: Mapping[str, Path | str] | None
    ) -> dict[str, Path]:
        known = {
            checkpoint.name for name in selected for checkpoint in REGISTRY[name].spec.checkpoints
        }
        overrides = {name: Path(path) for name, path in (models or {}).items()}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(f"unknown checkpoint names {unknown}; valid: {sorted(known)}")
        return overrides

    def _load_ocr(self, selected: set[Signal], model_size: ModelSize | str) -> OCREngine | None:
        readers = [name for name in selected if Engine.OCR in REGISTRY[name].spec.requires]
        if not readers:
            return None
        logger.info("loading OCR for %s", ", ".join(sorted(readers)))
        paddle_device = "gpu" if self._device == "cuda" else self._device
        return OCREngine(model_size=model_size, device=paddle_device)

    def _build_sensors(self, selected: set[Signal], config: LoadConfig) -> dict[Signal, Sensor]:
        engines = Engines(catalog=self._catalog, ocr=self._ocr)
        built: dict[Signal, Sensor] = {}
        for name in Signal:
            if name not in selected:
                continue
            sensor = REGISTRY[name].build(engines=engines, config=config)
            if sensor is None:
                logger.warning("signal %s disabled: its models are not available", name)
                continue
            built[name] = sensor
        dropped = True
        while dropped:
            dropped = False
            for name in list(built):
                if any(dep not in built for dep in REGISTRY[name].spec.after):
                    logger.warning("signal %s disabled: a signal it needs is disabled", name)
                    del built[name]
                    dropped = True
        return built

    def _resolve_rates(
        self, rates: Mapping[Signal | str, float] | None, fast: bool
    ) -> dict[Signal, float]:
        overrides = {_as_signal(name): fps for name, fps in (rates or {}).items()}
        unknown = sorted(name for name in overrides if name not in self._sensors)
        if unknown:
            raise ValueError(f"rates given for unselected signals: {unknown}")
        invalid = sorted(name for name, fps in overrides.items() if fps <= 0)
        if invalid:
            raise ValueError(f"rates must be positive: {invalid}")
        resolved = {}
        for name in self._sensors:
            spec = REGISTRY[name].spec
            resolved[name] = overrides.get(name, spec.thrift_rate if fast else spec.rate)
        return resolved

    def session(
        self, rates: Mapping[Signal | str, float] | None = None, fast: bool = False
    ) -> Session:
        """A fresh session over the loaded sensors, for any frame source.

        Raises:
            ValueError: a rate for an unselected signal, or a rate that
                is not positive.
        """
        return Session(
            sensors=self._sensors,
            rates=self._resolve_rates(rates, fast),
            batch_size=self._batch_size,
        )

    def parse_video(
        self,
        video: Path | str,
        output: Path | str | None = None,
        *,
        rates: Mapping[Signal | str, float] | None = None,
        fast: bool = False,
        progress: Progress | None = None,
    ) -> Log:
        """Parse one video into its log file.

        Decodes at the highest selected rate, writes
        ``<output>/<stem>.hudini.jsonl.gz`` (``output`` None means the
        working directory; a file path is used as given), and returns the
        same :class:`Log` that ``storage.load`` gives for the file.

        Args:
            rates: per-signal rate overrides in frames per second.
            fast: use each signal's thrift rate where no override is given.
            progress: called with each decoded frame's time in seconds.

        Raises:
            ValueError: a rate for an unselected signal, or a rate that
                is not positive.
        """
        video_path = Path(video)
        resolved = self._resolve_rates(rates, fast)
        header = self._header(video_info=probe_video(video_path), rates=resolved)
        frames = iter_frames(video_path, fps=max(resolved.values()))
        if progress is not None:
            frames = _with_progress(frames, progress)
        session = Session(sensors=self._sensors, rates=resolved, batch_size=self._batch_size)
        observations: list[Observation] = []
        with LogWriter(log_path(video_path, output)) as writer:
            writer.write_header(header)
            for obs in session.iter_observations(frames):
                writer.write(obs)
                observations.append(obs)
            self._seal(writer, observations)
        return Log(
            header=header,
            observations=tuple(observations),
            appendix=writer.appendix,
            footer=writer.footer,
        )

    def _seal(self, writer: LogWriter, observations: list[Observation]) -> None:
        """Attach the derived views the finished file carries: the
        corrected intervals for the appendix, and the summary folded
        from them for the footer."""
        patches = correct(observations, self._correction_rules(), self._catalog)
        intervals = interval_view(observations, patches, self._catalog)
        writer.set_intervals(INTERVALS_VERSION, intervals_to_wire(intervals))
        writer.set_summary(SUMMARY_VERSION, summarize(intervals, self._catalog).to_wire())

    def iter_observations(
        self,
        video: Path | str,
        *,
        rates: Mapping[Signal | str, float] | None = None,
        fast: bool = False,
    ) -> Iterator[Observation]:
        """Stream one video's observations without writing a file.

        Raises:
            ValueError: a rate for an unselected signal, or a rate that
                is not positive.
        """
        resolved = self._resolve_rates(rates, fast)
        session = Session(sensors=self._sensors, rates=resolved, batch_size=self._batch_size)
        yield from session.iter_observations(iter_frames(Path(video), fps=max(resolved.values())))

    def parse_frame(self, rgb: np.ndarray) -> list[Observation]:
        """Every selected signal's observations for one RGB image."""
        session = Session(
            sensors=self._sensors, rates=self._resolve_rates(None, fast=False), batch_size=1
        )
        return list(session.iter_observations([Frame(idx=0, time_s=0.0, rgb=rgb)]))

    def _header(self, video_info: VideoInfo, rates: dict[Signal, float]) -> Header:
        return Header(
            log_version=LOG_VERSION,
            version=_package_version(),
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            video=video_info,
            requested=self._requested,
            signals=self.signals,
            rates={name: fps for name, fps in rates.items() if name is not Signal.LAYOUT},
            batch_size=self._batch_size,
            device=self._device,
            models=self._model_infos(),
            engines=self._engine_infos(),
            corrections=self._correction_settings(),
        )

    def _model_infos(self) -> dict[str, ModelInfo]:
        infos = {}
        for name in self._sensors:
            for checkpoint in REGISTRY[name].spec.checkpoints:
                path = self._overrides.get(checkpoint.name) or resolve_checkpoint(checkpoint.file)
                if not path.exists():
                    continue
                threshold = self._detector_threshold if checkpoint.reports_threshold else None
                infos[checkpoint.name] = ModelInfo(
                    file=path.name, sha256=sha256_of(path), threshold=threshold
                )
        return infos

    def _engine_infos(self) -> dict[Engine, EngineInfo]:
        def readers(engine: Engine) -> tuple[Signal, ...]:
            return tuple(name for name in self.signals if engine in REGISTRY[name].spec.requires)

        infos = {}
        if self._ocr is not None:
            infos[Engine.OCR] = EngineInfo(
                model=f"pp-ocrv6-{self._ocr.model_size}", used_by=readers(Engine.OCR)
            )
        if readers(Engine.CATALOG):
            infos[Engine.CATALOG] = EngineInfo(
                locale=self._catalog.locale, used_by=readers(Engine.CATALOG)
            )
        return infos

    def _correction_rules(self) -> tuple[CorrectionSpec, ...]:
        """The declared rules of the running sensors, layout first."""
        return tuple(
            correction
            for name in (Signal.LAYOUT, *self.signals)
            for correction in REGISTRY[name].spec.corrections
        )

    def _correction_settings(self) -> tuple[dict[str, CorrectionSetting], ...]:
        return tuple(correction.settings() for correction in self._correction_rules())
