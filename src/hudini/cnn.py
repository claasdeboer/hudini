"""Classifies pod crops with the checkpoint-backed networks, one class per model.

Each class holds a loaded model and its device. It classifies a batch of
crops in one call. The preprocessing functions define the crop geometry.
Training and inference must use the same functions.
"""

from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from torch import nn

from hudini.schema import Box, Status

STATUS_INPUT_HEIGHT = 32
STATUS_INPUT_WIDTH = 192
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DIGIT_INPUT_SIZE = 24
# Reference geometry (px) used as a ratio: the arm square spans ~53 px
# of a 314 px-wide pod. Pod width is the anchor because it is identical
# for camera and instrument pods.
POD_W_REF = 314
ARM_CROP_SIZE = 53

ARMS = (1, 2, 3, 4)
# Square margin around the bar thickness, matching the digit-crop
# extractors the model was trained on.
END_MARGIN_FRAC = 0.25


def pad_pod_to_aspect(rgb: np.ndarray) -> np.ndarray:
    """Black-pad a status-pod crop to the model's W:H aspect, so the
    resize that follows cannot distort it."""
    height, width = rgb.shape[:2]
    target_ratio = STATUS_INPUT_WIDTH / STATUS_INPUT_HEIGHT
    ratio = width / height
    if ratio < target_ratio:
        target_width = int(height * target_ratio)
        left = (target_width - width) // 2
        return cv2.copyMakeBorder(
            rgb, 0, 0, left, target_width - width - left, cv2.BORDER_CONSTANT, value=0
        )
    if ratio > target_ratio:
        target_height = int(width / target_ratio)
        top = (target_height - height) // 2
        return cv2.copyMakeBorder(
            rgb, top, target_height - height - top, 0, 0, cv2.BORDER_CONSTANT, value=0
        )
    return rgb


def status_input_tensor(pod_rgb: np.ndarray) -> torch.Tensor:
    """A status pod as the CNN's (3, H, W) ImageNet-normalized tensor."""
    padded = pad_pod_to_aspect(pod_rgb)
    resized = cv2.resize(
        padded, (STATUS_INPUT_WIDTH, STATUS_INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR
    )
    scaled = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(scaled.transpose(2, 0, 1))


class StatusClassifier:
    """The camera pod's binary active/inactive CNN."""

    def __init__(self, model: nn.Module, device: str) -> None:
        self._model = model
        self._device = device

    @classmethod
    def load(cls, checkpoint: Path, device: str) -> "StatusClassifier":
        """Load the checkpoint. Its ``model_name``/``arch`` key names
        the timm architecture.

        Raises:
            FileNotFoundError: the checkpoint file is missing.
            ValueError: the checkpoint has no architecture key.
        """
        stored = torch.load(checkpoint, weights_only=False, map_location="cpu")
        arch = stored.get("model_name") or stored.get("arch")
        if not arch:
            raise ValueError("status checkpoint has no architecture key (model_name/arch)")

        state = stored.get("model_state_dict", stored)
        model = timm.create_model(
            arch, pretrained=False, num_classes=state["classifier.weight"].shape[0]
        )
        model.load_state_dict(state)
        model.eval().to(device)

        return cls(model=model, device=device)

    def classify(self, pods: list[np.ndarray]) -> list[tuple[Status, float]]:
        """One (status, probability) per RGB status-pod crop."""
        if not pods:
            return []

        tensor = torch.stack([status_input_tensor(pod) for pod in pods]).to(self._device)
        with torch.no_grad():
            probabilities = torch.softmax(self._model(tensor), dim=1).cpu().numpy()

        verdicts = []
        for row in probabilities:
            winner = int(np.argmax(row))
            status = Status.ACTIVE if winner == 1 else Status.INACTIVE
            verdicts.append((status, float(row[winner])))
        return verdicts


class ArmDigitNet(nn.Module):
    """VGG-style CNN for 24x24 grayscale digit classification: three 3x3
    convolution stages with BatchNorm and ReLU, max-pool downsampling,
    and a global-average-pooling head. ~6k parameters."""

    def __init__(self, input_size: int = DIGIT_INPUT_SIZE, num_classes: int = 4) -> None:
        super().__init__()
        self.input_size = input_size
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def crop_arm_region(pod_rgb: np.ndarray) -> np.ndarray:
    """The crop of the arm square at the pod's left edge.

    The side scales with pod width (``ARM_CROP_SIZE`` at ``POD_W_REF``)
    and is clamped to the pod size, so taller camera pods get the same
    crop as instrument pods at the same resolution.
    """
    height, width = pod_rgb.shape[:2]
    size = min(max(1, round(width * ARM_CROP_SIZE / POD_W_REF)), height, width)
    top = max(0, (height - size) // 2)
    return pod_rgb[top : top + size, :size]


def arm_input_array(pod_rgb: np.ndarray, input_size: int) -> np.ndarray:
    """The arm crop as the CNN's grayscale float array in [0, 1]."""
    gray = cv2.cvtColor(crop_arm_region(pod_rgb), cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (input_size, input_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class ArmDigitClassifier:
    """The arm-digit CNN: which digit (1-4) a pod's left circle shows."""

    def __init__(self, model: ArmDigitNet, device: str) -> None:
        self._model = model
        self._device = device

    @classmethod
    def load(cls, checkpoint: Path, device: str) -> "ArmDigitClassifier":
        """Load the checkpoint.

        Raises:
            FileNotFoundError: the checkpoint file is missing.
        """
        stored = torch.load(checkpoint, map_location=device, weights_only=True)
        model = ArmDigitNet(input_size=stored.get("input_size", DIGIT_INPUT_SIZE))
        model.load_state_dict(stored.get("model_state_dict", stored))
        model.to(device).eval()

        return cls(model=model, device=device)

    def classify(self, pods: list[np.ndarray]) -> list[tuple[int, float]]:
        """One (digit, probability) per RGB status-pod crop."""
        if not pods:
            return []

        crops = [arm_input_array(pod, self._model.input_size) for pod in pods]
        tensor = torch.from_numpy(np.stack(crops)).unsqueeze(1).to(self._device)
        with torch.no_grad():
            probabilities = torch.softmax(self._model(tensor), dim=-1).cpu().numpy()

        return [(int(np.argmax(row)) + 1, float(row[np.argmax(row)])) for row in probabilities]


def end_centers(box: Box) -> tuple[int, list[tuple[int, int]]]:
    """The end-square side and the two end centers of a detected bar.

    The ends sit inward by half the bar thickness. The digit glyph
    lives at one of them.
    """
    if box.w >= box.h:
        thickness, center_y = box.h, box.y + box.h // 2
        ends = [(box.x + thickness // 2, center_y), (box.x + box.w - thickness // 2, center_y)]
    else:
        thickness, center_x = box.w, box.x + box.w // 2
        ends = [(center_x, box.y + thickness // 2), (center_x, box.y + box.h - thickness // 2)]
    side = thickness + 2 * round(END_MARGIN_FRAC * thickness)
    return side, ends


def crop_square(
    image: np.ndarray, center_x: int, center_y: int, side: int, out_size: int
) -> np.ndarray | None:
    """A square crop around a center, resized to ``out_size``. None
    when it falls (almost) entirely off the image."""
    height, width = image.shape[:2]
    radius = side // 2
    crop = image[
        max(0, center_y - radius) : min(height, center_y + radius),
        max(0, center_x - radius) : min(width, center_x + radius),
    ]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


class OffscreenDigitClassifier:
    """The arm digit (1-4) of a detected off-screen bar, from its end crops.

    Four independent sigmoid heads, one per arm. The strongest head
    across both end crops names the arm. Below the checkpoint threshold
    the classifier abstains.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        input_size: int,
        mean: np.ndarray,
        std: np.ndarray,
        threshold: float,
    ) -> None:
        self._model = model
        self._device = device
        self._input_size = input_size
        self._mean = mean
        self._std = std
        self._threshold = threshold

    @classmethod
    def load(cls, checkpoint: Path, device: str) -> "OffscreenDigitClassifier":
        """Load the checkpoint. Its ``backbone`` key names the timm
        architecture.

        Raises:
            FileNotFoundError: the checkpoint file is missing.
        """
        stored = torch.load(checkpoint, weights_only=False, map_location="cpu")
        model = timm.create_model(stored["backbone"], pretrained=False, num_classes=len(ARMS))
        model.load_state_dict(stored["model_state_dict"])
        model.eval().to(device)

        return cls(
            model=model,
            device=device,
            input_size=stored["input_size"],
            mean=np.array(stored["mean"], np.float32),
            std=np.array(stored["std"], np.float32),
            threshold=stored["thr"],
        )

    def classify(self, region_rgb: np.ndarray, boxes: list[Box]) -> list[tuple[int | None, float]]:
        """One (arm, score) per bar box. The arm is None on abstain.

        ``region_rgb`` is the image the boxes live in. The model was
        trained on BGR decodes, so the conversion happens here.
        """
        region_bgr = region_rgb[:, :, ::-1]
        return [self._read_bar(region_bgr, box) for box in boxes]

    def _read_bar(self, region_bgr: np.ndarray, box: Box) -> tuple[int | None, float]:
        side, centers = end_centers(box)
        crops = []
        for center_x, center_y in centers:
            crop = crop_square(region_bgr, round(center_x), round(center_y), side, self._input_size)
            if crop is not None:
                crops.append(crop)
        if not crops:
            return None, 0.0

        rgb = np.stack([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops])
        scaled = (rgb.astype(np.float32) / 255.0 - self._mean) / self._std
        tensor = (
            torch.from_numpy(np.ascontiguousarray(scaled))
            .permute(0, 3, 1, 2)
            .float()
            .to(self._device)
        )
        with torch.no_grad():
            head_scores = torch.sigmoid(self._model(tensor)).max(0).values

        score = float(head_scores.max())
        if score < self._threshold:
            return None, round(score, 4)
        return ARMS[int(head_scores.argmax())], round(score, 4)
