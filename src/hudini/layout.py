"""Finds the UI geometry of one frame from its pixels.

One Sobel-Y pass over the active region marks the horizontal edge rows.
The pod walk reads each column from the bottom up and delimits the
status pod and the popups above it. The banner walk reads the top of the
region for the system-status banner. :func:`detect_region` finds the
active region, and :func:`frame_geometry` needs one. All boxes are in
full-frame pixels.
"""

from itertools import pairwise

import cv2
import numpy as np

from hudini.schema import Box, FrameGeometry, PodGeometry, Role

# A Sobel-Y magnitude at or above this counts a pixel as a horizontal edge.
MIN_EDGE_MAGNITUDE = 30
# Fraction of a column's width that must be edge for a row to be a UI border.
MIN_EDGE_COVERAGE = 0.90
# Qualifying rows within this many pixels merge into one border. UI border
# lines are 1-2 px thick at any resolution, so this stays absolute.
BORDER_MERGE_RADIUS_PX = 5

# The geometric thresholds below are fractions of the region height, so any
# recording resolution works unchanged.

# Tallest single UI segment (a popup, or the pod-to-popup gap) the border
# walk steps across before halting.
MAX_SEGMENT_HEIGHT_FRAC = 0.25
# Borders closer than this to the region's top or bottom edge are slivers.
MIN_DISTANCE_FROM_EDGE_FRAC = 15 / 720
# Segments shorter than this are dropped as border slivers.
MIN_SEGMENT_HEIGHT_FRAC = 20 / 720
# A region narrower than this fraction of the frame width is a bright speck
# on a dark frame, not the endoscope image the UI spans. Only the width is
# gated: on a black image the region shrinks to the UI bar itself.
MIN_REGION_WIDTH_FRAC = 0.5
# A camera pod is at least this many times taller than the mean of the rest.
CAMERA_HEIGHT_RATIO = 1.15


def detect_region(rgb: np.ndarray) -> Box | None:
    """The non-black region of a frame, or None for an all-black frame."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coordinates = cv2.findNonZero(thresholded)
    if coordinates is None:
        return None
    x, y, w, h = cv2.boundingRect(coordinates)
    return Box(x=x, y=y, w=w, h=h)


def _column_bounds(width: int, column_index: int) -> tuple[int, int]:
    column_width = width // 4
    x_start = column_index * column_width
    x_end = (column_index + 1) * column_width if column_index < 3 else width
    return x_start, x_end


def _edge_magnitude(region_rgb: np.ndarray) -> np.ndarray:
    """Absolute Sobel-Y magnitude of the full region."""
    gray = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2GRAY)
    return np.abs(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3))


def _merge_rows(rows: list[int], radius: int) -> list[int]:
    """Merge runs of near-adjacent rows into their mean positions.

    A row joins the current run when it is within ``radius`` rows of the
    run's last member. ``rows`` must be sorted and must not be empty.
    """
    merged: list[int] = []
    run = [rows[0]]
    for row in rows[1:]:
        if abs(run[-1] - row) <= radius:
            run.append(row)
        else:
            merged.append(sum(run) // len(run))
            run = [row]
    merged.append(sum(run) // len(run))
    return merged


def _edge_rows(magnitude: np.ndarray, first_row: int, last_row: int) -> list[int]:
    """Rows of an edge map that are almost fully edge, merged, ascending.

    A row qualifies when more than ``MIN_EDGE_COVERAGE`` of its width is
    edge. Only rows from ``first_row`` through ``last_row`` count.
    Near-adjacent qualifying rows merge into one position.
    """
    coverage = (magnitude > MIN_EDGE_MAGNITUDE).sum(axis=1) / magnitude.shape[1]
    qualifying = np.where(coverage > MIN_EDGE_COVERAGE)[0]
    qualifying = qualifying[(qualifying >= first_row) & (qualifying <= last_row)]
    if len(qualifying) == 0:
        return []
    return _merge_rows(qualifying.tolist(), radius=BORDER_MERGE_RADIUS_PX)


def _contiguous_prefix(values: list[int], max_gap: int) -> list[int]:
    """The longest prefix whose consecutive values are at most ``max_gap`` apart.

    The direction of the values does not matter, only the size of each
    gap. ``values`` must not be empty.
    """
    prefix = [values[0]]
    for value in values[1:]:
        if abs(value - prefix[-1]) > max_gap:
            break
        prefix.append(value)
    return prefix


def _pod_borders(magnitude_column: np.ndarray) -> list[int]:
    """UI borders of one column's edge map, ordered bottom to top.

    The search band is the bottom half of the region, without the sliver
    margin at the bottom edge. The walk starts at the bottommost border.
    It stops before the first gap that is more than the maximum segment
    height.
    """
    height = magnitude_column.shape[0]
    max_segment = max(1, round(height * MAX_SEGMENT_HEIGHT_FRAC))
    min_from_edge = max(1, round(height * MIN_DISTANCE_FROM_EDGE_FRAC))
    borders = _edge_rows(
        magnitude_column, first_row=height // 2 + 1, last_row=height - min_from_edge
    )
    if not borders:
        return []
    return _contiguous_prefix(borders[::-1], max_gap=max_segment)


def _banner_bottom(magnitude: np.ndarray) -> int | None:
    """The banner's bottom edge row, or None when there is no banner.

    The search band is the top half of the region, without the sliver
    margin at the top edge. The first border must be no more than one
    maximum segment height below the top edge. The walk steps down and
    returns the deepest border it reaches.
    """
    height = magnitude.shape[0]
    max_segment = max(1, round(height * MAX_SEGMENT_HEIGHT_FRAC))
    min_from_edge = max(1, round(height * MIN_DISTANCE_FROM_EDGE_FRAC))
    borders = _edge_rows(magnitude, first_row=min_from_edge, last_row=height // 2)
    if not borders or borders[0] > max_segment:
        return None
    return _contiguous_prefix(borders, max_gap=max_segment)[-1]


def _camera_column(heights: dict[int, int]) -> int | None:
    """The column whose pod is meaningfully taller, or None without one."""
    if len(heights) < 2:
        return None
    tallest = max(heights, key=lambda column: heights[column])
    others = [height for column, height in heights.items() if column != tallest]
    mean_others = sum(others) / len(others)
    if mean_others > 0 and heights[tallest] / mean_others >= CAMERA_HEIGHT_RATIO:
        return tallest
    return None


def frame_geometry(rgb: np.ndarray, region: Box) -> FrameGeometry:
    """The UI geometry of one frame, inside a known active region.

    Every column is listed. A column whose segmentation failed has a
    None ``box``. All boxes are in full-frame pixels. A frame without
    any parseable UI gives four boxless pods and no banner. So does a
    region narrower than :data:`MIN_REGION_WIDTH_FRAC` of the frame.
    """
    if region.w < rgb.shape[1] * MIN_REGION_WIDTH_FRAC:
        return _no_ui(region)
    region_rgb = rgb[region.y : region.y + region.h, region.x : region.x + region.w]
    height, width = region_rgb.shape[:2]
    min_segment = max(1, round(height * MIN_SEGMENT_HEIGHT_FRAC))
    magnitude = _edge_magnitude(region_rgb)

    pod_boxes: dict[int, Box] = {}
    popup_boxes: dict[int, list[Box]] = {column: [] for column in range(1, 5)}
    for column_index in range(4):
        column = column_index + 1
        x_start, x_end = _column_bounds(width, column_index)
        borders = _pod_borders(magnitude[:, x_start:x_end])
        if not borders:
            continue
        status_top = borders[0]
        if height - status_top >= min_segment:
            pod_boxes[column] = Box(
                x=region.x + x_start,
                y=region.y + status_top,
                w=x_end - x_start,
                h=height - status_top,
            )
        for below, above in pairwise(borders):
            if below - above >= min_segment:
                popup_boxes[column].append(
                    Box(
                        x=region.x + x_start, y=region.y + above, w=x_end - x_start, h=below - above
                    )
                )

    camera = _camera_column({column: box.h for column, box in pod_boxes.items()})
    pods = tuple(
        PodGeometry(
            column=column,
            role=Role.CAMERA if column == camera else Role.INSTRUMENT,
            box=pod_boxes.get(column),
            popups=tuple(popup_boxes[column]),
        )
        for column in range(1, 5)
    )

    banner_bottom = _banner_bottom(magnitude)
    banner = None
    if banner_bottom is not None:
        banner = Box(x=region.x, y=region.y, w=width, h=banner_bottom)
    return FrameGeometry(region=region, pods=pods, banner=banner)


def layout_complete(geometry: FrameGeometry) -> bool:
    """Whether every pod has a box and one of them is the camera pod."""
    return all(pod.box is not None for pod in geometry.pods) and any(
        pod.role is Role.CAMERA for pod in geometry.pods
    )


def _no_ui(region: Box) -> FrameGeometry:
    """Four boxless pods and no banner."""
    pods = tuple(
        PodGeometry(column=column, role=Role.INSTRUMENT, box=None, popups=())
        for column in range(1, 5)
    )
    return FrameGeometry(region=region, pods=pods, banner=None)
