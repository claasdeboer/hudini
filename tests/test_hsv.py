"""Unit tests for hudini.hsv: the rules on synthetic pixels."""

import numpy as np
import pytest

from hudini.hsv import (
    PressRead,
    classify_instrument_status,
    classify_laser,
    classify_pedals,
    crop_pedal_action_label,
    warning_score,
)
from hudini.schema import PedalColor, Status

POD_HEIGHT, POD_WIDTH = 50, 314

# RGB fills that land inside the tuned HSV windows.
PRESSED_YELLOW = (230, 190, 40)
PRESSED_BLUE = (40, 140, 230)
TEAL = (40, 200, 210)
LASER_GREEN = (40, 230, 60)
PALE_GREEN = (150, 210, 150)
DARK = (30, 30, 30)


def pod(fill=DARK) -> np.ndarray:
    image = np.zeros((POD_HEIGHT, POD_WIDTH, 3), dtype=np.uint8)
    image[:] = fill
    return image


def fill_action_area(image: np.ndarray, color: PedalColor, rgb) -> None:
    top, bottom = (1, 20) if color is PedalColor.YELLOW else (31, 50)
    image[top:bottom, 247:] = rgb


class TestClassifyPedals:
    def test_a_filled_action_area_reads_as_pressed(self):
        image = pod()
        fill_action_area(image, PedalColor.YELLOW, PRESSED_YELLOW)
        reads = classify_pedals(image)
        assert reads[PedalColor.YELLOW].pressed is True
        assert reads[PedalColor.YELLOW].score > 0.9
        assert reads[PedalColor.BLUE] == PressRead(pressed=False, score=0.0)

    def test_a_blue_fill_presses_only_the_blue_pedal(self):
        image = pod()
        fill_action_area(image, PedalColor.BLUE, PRESSED_BLUE)
        reads = classify_pedals(image)
        assert reads[PedalColor.BLUE].pressed is True
        assert reads[PedalColor.YELLOW].pressed is False

    def test_an_empty_pod_presses_nothing(self):
        reads = classify_pedals(pod())
        assert all(read.pressed is False for read in reads.values())


class TestClassifyLaser:
    def test_a_saturated_green_corner_reads_on(self):
        image = pod()
        image[:12, 236:] = LASER_GREEN
        on, score = classify_laser(image)
        assert on is True
        assert score > 0.9

    def test_a_dark_pod_reads_off(self):
        assert classify_laser(pod()) == (False, 0.0)

    def test_unsaturated_green_reads_off(self):
        image = pod()
        image[:12, 236:] = PALE_GREEN
        on, _score = classify_laser(image)
        assert on is False

    def test_a_degenerate_crop_reads_off(self):
        assert classify_laser(np.zeros((2, 2, 3), dtype=np.uint8)) == (False, 0.0)


class TestCropPedalActionLabel:
    @pytest.mark.parametrize("color", list(PedalColor))
    def test_the_cell_sits_right_of_the_side_indicator(self, color):
        cell = crop_pedal_action_label(pod(), color)
        assert cell.shape[1] == POD_WIDTH - 247
        assert cell.shape[0] > 0

    def test_a_degenerate_pod_gives_an_empty_cell(self):
        cell = crop_pedal_action_label(np.zeros((2, 1, 3), dtype=np.uint8), PedalColor.YELLOW)
        assert cell.size == 0


class TestInstrumentStatus:
    def test_a_teal_arm_square_reads_active(self):
        image = pod()
        image[:, :52] = TEAL
        status, score = classify_instrument_status(image)
        assert status is Status.ACTIVE
        assert score > 0.9

    def test_a_yellow_arm_square_reads_warning(self):
        image = pod()
        image[:, :52] = PRESSED_YELLOW
        status, _score = classify_instrument_status(image)
        assert status is Status.WARNING

    def test_a_dark_arm_square_reads_inactive(self):
        status, score = classify_instrument_status(pod())
        assert (status, score) == (Status.INACTIVE, 0.0)


class TestWarningScore:
    def test_fires_on_the_yellow_arm_square(self):
        image = pod()
        image[:, :52] = PRESSED_YELLOW
        assert warning_score(image) > 0.9

    def test_stays_silent_on_teal_and_dark(self):
        teal = pod()
        teal[:, :52] = TEAL
        assert warning_score(teal) is None
        assert warning_score(pod()) is None
