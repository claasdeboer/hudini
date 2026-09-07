"""Classifies pod areas by color fractions in HSV space, with no learned weights.

Pedal presses, instrument-pod status, and the camera laser readout come
from color fractions over fixed areas of the status pod. The geometry
constants are ratios of the reference pod (314x50 px), so any resolution
works. This module does not import torch or paddleocr.
"""

from typing import NamedTuple

import cv2
import numpy as np

from hudini.schema import PedalColor, Status

POD_W_REF = 314
POD_H_REF = 50
PEDAL_W = 84
PEDAL_LR_W = 17
BORDER = 1
PEDAL_UPPER_H = 19
PEDAL_LOWER_H = 19
PEDAL_GAP = 11

PEDAL_YELLOW_H_RANGE = (20, 40)
PEDAL_YELLOW_S_MIN = 120
PEDAL_YELLOW_V_MIN = 120
PEDAL_BLUE_H_RANGE = (95, 115)
PEDAL_BLUE_S_MIN = 120
# The active-pod overlay bleeds a dim blue (V around 70) into the cell;
# a real press is far brighter (V around 154).
PEDAL_BLUE_V_MIN = 80
# A pressed yellow cell fills above 0.2; compression bleed on the label
# text stays below 0.11. Blue needs a majority fill.
PEDAL_YELLOW_THRESHOLD = 0.15
PEDAL_BLUE_THRESHOLD = 0.50

# The arm square: the square area at the pod's left edge. It shows the
# arm digit in its circle and the L/R hand-controller icons.
ARM_SQUARE_WIDTH_FRAC = 0.166
TEAL_H_RANGE = (85, 115)
TEAL_S_MIN = 100
# Active pods fill the square with teal well above this; overlay bleed on
# some recordings pulls a few down to just above it.
TEAL_THRESHOLD = 0.35
WARNING_H_RANGE = (15, 35)
WARNING_S_MIN = 100
WARNING_V_MIN = 80
# Midpoint of the gap between inactive pods (below 0.13) and warning
# pods (above 0.21).
WARNING_THRESHOLD = 0.17

# The laser readout: the top-right corner of the camera pod. ON renders
# saturated green; high S excludes firefly scene-bleed behind the text.
LASER_RIGHT_FRAC = 0.25
LASER_TOP_FRAC = 0.25
LASER_H_RANGE = (50, 75)
LASER_S_MIN = 200
LASER_V_MIN = 150
# The readout is a few saturated green pixels, so a tiny fraction is ON.
# The rule can still fire on the one frame of a firefly transition, where
# the scene bleeds green behind the text.
LASER_ON_THRESHOLD = 0.001


class PressRead(NamedTuple):
    """One pedal's verdict: pressed or not, with the color fraction."""

    pressed: bool
    score: float


def _action_boxes(height: int, width: int) -> tuple[tuple[int, int, int, int], ...]:
    """The (upper, lower) action areas as (y0, y1, x0, x1)."""
    pedal_x = round(width * (POD_W_REF - PEDAL_W) / POD_W_REF)
    divider_x = pedal_x + round(width * PEDAL_LR_W / POD_W_REF)

    upper_top = round(height * BORDER / POD_H_REF)
    upper_bottom = round(height * (BORDER + PEDAL_UPPER_H) / POD_H_REF)
    lower_top = round(height * (BORDER + PEDAL_UPPER_H + PEDAL_GAP) / POD_H_REF)
    lower_bottom = round(height * (BORDER + PEDAL_UPPER_H + PEDAL_GAP + PEDAL_LOWER_H) / POD_H_REF)

    return (
        (upper_top, upper_bottom, divider_x, width),
        (lower_top, lower_bottom, divider_x, width),
    )


def _color_fraction(
    hsv_patch: np.ndarray, h_range: tuple[int, int], s_min: int, v_min: int
) -> float:
    if hsv_patch.size == 0:
        return 0.0
    hue, sat, val = cv2.split(hsv_patch)
    mask = (hue >= h_range[0]) & (hue <= h_range[1]) & (sat >= s_min) & (val >= v_min)
    return float(mask.mean())


def classify_pedals(pod_rgb: np.ndarray) -> dict[PedalColor, PressRead]:
    """Both pedals' press verdicts for one status-pod crop."""
    height, width = pod_rgb.shape[:2]
    hsv = cv2.cvtColor(pod_rgb, cv2.COLOR_RGB2HSV)
    upper, lower = _action_boxes(height, width)

    yellow = _color_fraction(
        hsv[upper[0] : upper[1], upper[2] : upper[3]],
        PEDAL_YELLOW_H_RANGE,
        PEDAL_YELLOW_S_MIN,
        PEDAL_YELLOW_V_MIN,
    )
    blue = _color_fraction(
        hsv[lower[0] : lower[1], lower[2] : lower[3]],
        PEDAL_BLUE_H_RANGE,
        PEDAL_BLUE_S_MIN,
        PEDAL_BLUE_V_MIN,
    )

    return {
        PedalColor.YELLOW: PressRead(yellow >= PEDAL_YELLOW_THRESHOLD, round(yellow, 4)),
        PedalColor.BLUE: PressRead(blue >= PEDAL_BLUE_THRESHOLD, round(blue, 4)),
    }


def crop_pedal_action_label(pod_rgb: np.ndarray, color: PedalColor) -> np.ndarray:
    """The action-label cell of one pedal. Empty when the pod is too small.

    The label tracks the press (force bipolar reads GRIP idle and STRONG
    held). Cropping is split from reading so a whole batch of cells goes
    to the recognizer in one call.
    """
    height, width = pod_rgb.shape[:2]
    upper, lower = _action_boxes(height, width)
    y0, y1, x0, x1 = upper if color is PedalColor.YELLOW else lower
    return pod_rgb[max(0, y0 - 1) : min(height, y1 + 1), x0:x1]


def _arm_square_fractions(pod_rgb: np.ndarray) -> tuple[float, float]:
    """(teal, warning-yellow) fractions over the arm square."""
    width = pod_rgb.shape[1]
    square = pod_rgb[:, : int(width * ARM_SQUARE_WIDTH_FRAC)]
    hsv = cv2.cvtColor(square, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    teal = (hue >= TEAL_H_RANGE[0]) & (hue <= TEAL_H_RANGE[1]) & (sat > TEAL_S_MIN)
    warning = (
        (hue >= WARNING_H_RANGE[0])
        & (hue <= WARNING_H_RANGE[1])
        & (sat > WARNING_S_MIN)
        & (val > WARNING_V_MIN)
    )
    return float(teal.mean()), float(warning.mean())


def classify_instrument_status(pod_rgb: np.ndarray) -> tuple[Status, float]:
    """The status of an instrument pod, read from its arm square."""
    teal, warning = _arm_square_fractions(pod_rgb)
    if teal >= TEAL_THRESHOLD:
        return Status.ACTIVE, round(teal, 4)
    if warning >= WARNING_THRESHOLD:
        return Status.WARNING, round(warning, 4)
    return Status.INACTIVE, round(max(teal, warning), 4)


def classify_laser(pod_rgb: np.ndarray) -> tuple[bool, float]:
    """Whether the laser readout shows ON, with the green fraction.

    Reads the top-right corner of the crop. Meaningful only on the
    camera pod.
    """
    height, width = pod_rgb.shape[:2]
    corner = pod_rgb[: round(height * LASER_TOP_FRAC), round(width * (1 - LASER_RIGHT_FRAC)) :]
    if corner.size == 0:
        return False, 0.0
    fraction = _color_fraction(
        cv2.cvtColor(corner, cv2.COLOR_RGB2HSV), LASER_H_RANGE, LASER_S_MIN, LASER_V_MIN
    )
    return fraction >= LASER_ON_THRESHOLD, round(fraction, 4)


def warning_score(pod_rgb: np.ndarray) -> float | None:
    """The warning fraction when the warning rule fires, else None.

    The camera pod renders warning the same yellow way as instrument
    pods, and its binary CNN cannot say "warning", so the caller applies
    this rule first.
    """
    _, warning = _arm_square_fractions(pod_rgb)
    if warning >= WARNING_THRESHOLD:
        return round(warning, 4)
    return None
