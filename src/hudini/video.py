"""Reads the container facts and the frames of a video file.

PyAV wheels bundle ffmpeg, so there is no system prerequisite.
:func:`probe_video` gives the container facts as the header's
``VideoInfo``. :func:`iter_frames` yields the frames a sampling rate asks
for and skips the rest before pixel conversion.
"""

import logging
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import av
import numpy as np

from hudini.schema import VideoInfo

logger = logging.getLogger(__name__)

_TICK_EPSILON = 1e-6
_FALLBACK_FPS = 60.0
DECODE_THREADS = 4


class Frame(NamedTuple):
    """One decoded frame.

    Attributes:
        idx: the source frame index, as observations record it.
        time_s: presentation time in seconds. The first frame is at zero.
        rgb: the frame as an RGB array with square pixels.
    """

    idx: int
    time_s: float
    rgb: np.ndarray


def probe_video(video_path: Path | str) -> VideoInfo:
    """Container facts of a video. ``sha256`` stays None, because hashing
    is a separate step.

    ``frame_rate`` and ``duration_s`` are None when the container does not
    state them.
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        frame_rate = float(stream.average_rate) if stream.average_rate else None
        if stream.duration is not None and stream.time_base is not None:
            duration_s = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_s = container.duration / av.time_base
        else:
            duration_s = None
        return VideoInfo(
            filename=Path(video_path).name,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            codec=stream.codec_context.name,
            frame_rate=round(frame_rate, 3) if frame_rate else None,
            duration_s=round(duration_s, 3) if duration_s is not None else None,
        )


def iter_frames(video_path: Path | str, fps: float = 1.0) -> Iterator[Frame]:
    """Yield the frames at ``fps``.

    The loop decodes every frame but converts only the kept frames.
    Timing is pts-based, so variable-frame-rate sources keep honest
    timestamps. The first frame at or past each ``1 / fps`` tick is kept,
    and the achieved rate is ``min(fps, source_fps)``. A frame without a
    pts falls back to ``index / average_rate``. The first such frame logs
    a warning. Non-square source pixels are resized so ``rgb`` is square.
    """
    with av.open(str(Path(video_path))) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # PyAV defaults to single-threaded decode
        # AUTO sizes the thread pool to every core of the machine, per
        # process. Cap it: four threads decode 1080p60 far above any
        # sampling rate, and parallel parses stop flooding a shared node.
        stream.thread_count = min(DECODE_THREADS, len(os.sched_getaffinity(0)))
        fallback_fps = float(stream.average_rate) if stream.average_rate else _FALLBACK_FPS
        aspect = stream.sample_aspect_ratio
        pixel_aspect = (
            aspect.numerator / aspect.denominator if aspect and aspect.denominator else 1.0
        )

        origin: float | None = None
        tick = 0  # the next sample point is tick / fps
        missing_pts_warned = False
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and frame.time_base is not None:
                time_s = float(frame.pts * frame.time_base)
            else:
                if not missing_pts_warned:
                    logger.warning(
                        "frame %d has no pts. Timing falls back to index / %.3g fps.",
                        index,
                        fallback_fps,
                    )
                    missing_pts_warned = True
                time_s = index / fallback_fps
            if origin is None:
                origin = time_s
            time_s -= origin
            if time_s < tick / fps - _TICK_EPSILON:
                continue
            # Integer tick arithmetic, so float error cannot accumulate.
            # Skip the ticks that a slow source or a VFR gap left behind.
            tick = max(tick + 1, math.floor(time_s * fps + _TICK_EPSILON) + 1)
            if pixel_aspect != 1.0:
                frame = frame.reformat(width=round(frame.width * pixel_aspect), height=frame.height)
            yield Frame(idx=index, time_s=time_s, rgb=frame.to_ndarray(format="rgb24"))
