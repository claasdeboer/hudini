"""Detects boxes with RF-DETR.

:class:`Detector` wraps one trained checkpoint. The checkpoint states its
model class, its resolution, and its class names, so :func:`load_detector`
takes only the file. ``rfdetr`` is an optional dependency, the
``[rfdetr]`` extra. :func:`load_detector` returns None when it is missing.
This module imports torch only inside its functions.
"""

import logging
import math
from collections.abc import Mapping
from importlib.util import find_spec
from pathlib import Path
from typing import NamedTuple

import numpy as np

from hudini.schema import Box

logger = logging.getLogger(__name__)

RFDETR_MODEL_NAMES = frozenset(
    {"RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRBase", "RFDETRLarge"}
)
RFDETR_PATCH_SIZE_PX = 16  # DINOv2 patch size in all RFDETR backbones
RFDETR_POSITION_EMBEDDINGS_KEY = "backbone.0.encoder.encoder.embeddings.position_embeddings"


class Detection(NamedTuple):
    """One detected box, in the input image's pixel space.

    Attributes:
        label: the class name, or the class id as a string when the
            detector has no name for it.
        score: detector confidence, rounded to four decimals.
    """

    label: str
    score: float
    box: Box


class Detector:
    """A trained RF-DETR checkpoint at its training resolution.

    ``detect`` reads any number of images in one call and never pads
    the batch. With ``batched`` False, the model is called once per
    image, which matches the compiled model's fixed input shape.
    """

    def __init__(
        self,
        model,
        class_names: Mapping[int, str],
        threshold: float,
        batched: bool,
    ) -> None:
        self._model = model
        self._class_names = dict(class_names)
        self._threshold = threshold
        self._batched = batched

    @property
    def labels(self) -> tuple[str, ...]:
        """The class names, in class-id order."""
        return tuple(self._class_names[class_id] for class_id in sorted(self._class_names))

    def detect(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """One detection list per RGB image, boxes in image pixels."""
        if not images:
            return []
        contiguous = [np.ascontiguousarray(image) for image in images]
        if not self._batched or len(contiguous) == 1:
            outputs = [
                self._model.predict(image, threshold=self._threshold) for image in contiguous
            ]
        else:
            outputs = self._model.predict(contiguous, threshold=self._threshold)
        return [self._decode(output) for output in outputs]

    def _decode(self, detections) -> list[Detection]:
        decoded = []
        for xyxy, class_id, confidence in zip(
            detections.xyxy, detections.class_id, detections.confidence, strict=True
        ):
            x0, y0, x1, y1 = (float(value) for value in xyxy)
            decoded.append(
                Detection(
                    label=self._class_names.get(int(class_id), str(int(class_id))),
                    score=round(float(confidence), 4),
                    box=Box(
                        x=round(x0),
                        y=round(y0),
                        w=round(x1 - x0),
                        h=round(y1 - y0),
                    ),
                )
            )
        return decoded


def _checkpoint_metadata(checkpoint: Path) -> tuple[str, int, dict[int, str]]:
    """Model class name, training resolution, and class names of a checkpoint.

    The load uses ``mmap=True``, so the weight bytes stay on disk. The
    resolution comes from the position-embedding token count.
    ``tokens - 1`` is ``(resolution / patch size) ** 2``.

    Raises:
        ValueError: the model name is missing or unknown, or the token
            count is not a square.
        KeyError: the checkpoint does not have a required rfdetr key.
    """
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model_name = payload.get("model_name")
    if model_name not in RFDETR_MODEL_NAMES:
        raise ValueError(f"{checkpoint}: model {model_name!r} is not a known rfdetr class")
    tokens = payload["model"][RFDETR_POSITION_EMBEDDINGS_KEY].shape[1] - 1
    side = math.isqrt(tokens)
    if side * side != tokens:
        raise ValueError(f"{checkpoint}: {tokens} position tokens is not a square grid")
    class_names = payload["args"]["class_names"]
    return model_name, side * RFDETR_PATCH_SIZE_PX, dict(enumerate(class_names))


def _adopt_library_loggers() -> None:
    """Route the rf-detr loggers through the root logging configuration.

    The library gives its loggers their own stdout and stderr handlers,
    pins their level, and stops propagation, so the hudini logging
    setup cannot see or silence them. This strips those handlers and
    restores propagation, so the records flow to the root logger. The
    NullHandler stays behind, because the library only installs its
    handlers on a logger that has none.
    """
    names = [
        name
        for name in logging.Logger.manager.loggerDict
        if name in ("rf-detr", "rfdetr") or name.startswith("rfdetr.")
    ]
    for name in names:
        library_logger = logging.getLogger(name)
        library_logger.handlers = [logging.NullHandler()]
        library_logger.setLevel(logging.NOTSET)
        library_logger.propagate = True


def load_detector(checkpoint: Path, *, threshold: float, batched: bool = False) -> Detector | None:
    """A loaded detector, or None when it cannot load.

    The checkpoint states its own model class, training resolution, and
    class names, so loading needs only the file. The result is None
    when the ``[rfdetr]`` extra is not installed or the checkpoint file
    is missing. Both causes are logged. With ``batched`` False, the
    model compiles for single-image calls. The compiled model accepts
    only that one input shape. With ``batched`` True, the model stays
    uncompiled and accepts any batch size, because read batches vary
    and are never padded.

    Raises:
        ValueError: the checkpoint names an unknown model class or has
            a non-square token grid.
        KeyError: the checkpoint does not have a required rfdetr key.
    """
    if find_spec("rfdetr") is None:
        logger.warning("detector unavailable: install the [rfdetr] extra. Continuing without.")
        return None
    if not checkpoint.exists():
        logger.warning("detector checkpoint missing: %s. Continuing without.", checkpoint)
        return None

    import rfdetr

    model_name, resolution, class_names = _checkpoint_metadata(checkpoint)
    model = getattr(rfdetr, model_name)(pretrain_weights=str(checkpoint), resolution=resolution)
    if not batched and hasattr(model, "inference"):
        try:
            model.inference()
        except Exception as error:  # noqa: BLE001
            logger.warning("compilation for inference skipped: %s", error)
    _adopt_library_loggers()
    return Detector(model=model, class_names=class_names, threshold=threshold, batched=batched)
