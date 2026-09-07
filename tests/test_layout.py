"""Unit tests for hudini.layout.

The geometry runs on synthetic frames with drawn border lines: a one-pixel
line on a uniform background fires Sobel at the two adjacent rows, and the
cluster average recovers the exact line row.
"""

from collections.abc import Callable

import numpy as np
import pytest

from hudini.layout import detect_region, frame_geometry, layout_complete
from hudini.schema import Box, FrameGeometry, PodGeometry, Role

REGION = Box(x=50, y=20, w=400, h=400)


@pytest.fixture
def make_frame() -> Callable[..., np.ndarray]:
    """A letterboxed frame with a gray active region and optional lines.

    Lines are ``(row, x_start, x_end)`` in region coordinates.
    """

    def _make(*lines: tuple[int, int, int]) -> np.ndarray:
        frame = np.zeros((440, 500, 3), dtype=np.uint8)
        frame[REGION.y : REGION.y + REGION.h, REGION.x : REGION.x + REGION.w] = 80
        for row, x_start, x_end in lines:
            frame[REGION.y + row, REGION.x + x_start : REGION.x + x_end] = 255
        return frame

    return _make


UI_LINES = (
    (350, 0, 200),  # status border, columns 1 and 2
    (350, 300, 400),  # status border, column 4
    (340, 200, 300),  # status border, column 3 — taller pod, the camera
    (330, 100, 200),  # popup border, column 2
    (300, 100, 200),  # second popup border, column 2
    (30, 0, 400),  # banner bottom, full width
)


@pytest.mark.parametrize(
    ("boxes", "camera", "expected"),
    [
        ((True, True, True, True), 3, True),
        ((True, True, True, True), None, False),
        ((True, False, True, True), 3, False),
    ],
)
def test_layout_complete_needs_four_boxes_and_a_camera(boxes, camera, expected):
    pods = tuple(
        PodGeometry(
            column=column,
            role=Role.CAMERA if column == camera else Role.INSTRUMENT,
            box=Box(x=0, y=0, w=10, h=10) if boxed else None,
            popups=(),
        )
        for column, boxed in enumerate(boxes, start=1)
    )
    assert layout_complete(FrameGeometry(region=REGION, pods=pods, banner=None)) is expected


class TestDetectRegion:
    def test_an_all_black_frame_has_no_region(self):
        assert detect_region(np.zeros((100, 100, 3), dtype=np.uint8)) is None

    def test_the_region_is_the_bounding_box_of_the_bright_pixels(self, make_frame):
        assert detect_region(make_frame()) == REGION


class TestFrameGeometry:
    def test_status_pods_span_border_to_bottom_in_full_frame_pixels(self, make_frame):
        geometry = frame_geometry(make_frame(*UI_LINES), REGION)
        by_column = {pod.column: pod for pod in geometry.pods}
        assert by_column[1].box == Box(x=50, y=370, w=100, h=50)
        assert by_column[2].box == Box(x=150, y=370, w=100, h=50)
        assert by_column[3].box == Box(x=250, y=360, w=100, h=60)
        assert by_column[4].box == Box(x=350, y=370, w=100, h=50)

    def test_the_taller_pod_is_the_camera(self, make_frame):
        geometry = frame_geometry(make_frame(*UI_LINES), REGION)
        roles = {pod.column: pod.role for pod in geometry.pods}
        assert roles[3] is Role.CAMERA
        assert {roles[1], roles[2], roles[4]} == {Role.INSTRUMENT}

    def test_popups_stack_bottom_up_above_the_pod(self, make_frame):
        geometry = frame_geometry(make_frame(*UI_LINES), REGION)
        by_column = {pod.column: pod for pod in geometry.pods}
        assert by_column[2].popups == (
            Box(x=150, y=350, w=100, h=20),
            Box(x=150, y=320, w=100, h=30),
        )
        assert by_column[1].popups == ()

    def test_the_banner_spans_the_top_edge_to_its_border(self, make_frame):
        geometry = frame_geometry(make_frame(*UI_LINES), REGION)
        assert geometry.banner == Box(x=50, y=20, w=400, h=30)

    def test_a_region_without_ui_gives_boxless_pods_and_no_banner(self, make_frame):
        geometry = frame_geometry(make_frame(), REGION)
        assert [pod.box for pod in geometry.pods] == [None, None, None, None]
        assert all(pod.popups == () for pod in geometry.pods)
        assert geometry.banner is None
        assert geometry.region == REGION

    def test_a_speck_region_gives_boxless_pods_and_no_banner(self, make_frame):
        speck = Box(x=REGION.x, y=REGION.y, w=20, h=20)
        geometry = frame_geometry(make_frame((15, 0, 20)), speck)
        assert [pod.box for pod in geometry.pods] == [None, None, None, None]
        assert geometry.banner is None
        assert geometry.region == speck

    def test_a_line_in_the_top_half_makes_no_pod(self, make_frame):
        geometry = frame_geometry(make_frame((100, 0, 400)), REGION)
        assert all(pod.box is None for pod in geometry.pods)

    def test_a_bottom_sliver_is_not_a_border(self, make_frame):
        geometry = frame_geometry(make_frame((395, 0, 400)), REGION)
        assert all(pod.box is None for pod in geometry.pods)

    def test_a_deep_border_is_not_a_banner(self, make_frame):
        geometry = frame_geometry(make_frame((150, 0, 400)), REGION)
        assert geometry.banner is None
