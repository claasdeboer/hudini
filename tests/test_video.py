"""Unit tests for hudini.video.

The decode loop is tested against fake PyAV objects, which pins the tick
arithmetic and the convert-only-kept-frames contract. One integration test
encodes a real file with PyAV.
"""

from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Self
from unittest.mock import patch

import av
import numpy as np
import pytest

from hudini.video import Frame, iter_frames, probe_video


class FakeVideoFrame:
    """A decodable frame that records conversion and reformatting."""

    def __init__(self, time: float | None, width: int = 64, height: int = 48) -> None:
        self.time = time
        self.time_base = Fraction(1, 90000)
        self.pts = None if time is None else round(time * 90000)
        self.width = width
        self.height = height
        self.converted = False
        self.reformatted_to: tuple[int, int] | None = None

    def reformat(self, width: int, height: int) -> "FakeVideoFrame":
        self.reformatted_to = (width, height)
        resized = FakeVideoFrame(time=self.time, width=width, height=height)
        resized.converted = self.converted
        resized.reformatted_to = self.reformatted_to
        return resized

    def to_ndarray(self, format: str) -> np.ndarray:
        self.converted = True
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


class FakeContainer:
    def __init__(self, stream: SimpleNamespace, frames: list[FakeVideoFrame]) -> None:
        self.streams = SimpleNamespace(video=[stream])
        self.duration = None
        self._frames = frames

    def decode(self, stream: SimpleNamespace) -> list[FakeVideoFrame]:
        return self._frames

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


@pytest.fixture
def make_source() -> Callable[..., tuple[FakeContainer, list[FakeVideoFrame]]]:
    """Build a fake container from frame times."""

    def _make(
        times: list[float | None],
        average_rate: Fraction | None = Fraction(30, 1),
        sample_aspect_ratio: Fraction | None = None,
    ) -> tuple[FakeContainer, list[FakeVideoFrame]]:
        frames = [FakeVideoFrame(time=t) for t in times]
        stream = SimpleNamespace(
            average_rate=average_rate,
            sample_aspect_ratio=sample_aspect_ratio,
            thread_type=None,
            codec_context=SimpleNamespace(name="h264", width=64, height=48),
            duration=None,
            time_base=None,
            frames=len(frames),
        )
        return FakeContainer(stream, frames), frames

    return _make


def _collect(container: FakeContainer, fps: float) -> list[Frame]:
    with patch("hudini.video.av.open", return_value=container):
        return list(iter_frames("fake.mp4", fps=fps))


class TestIterFrames:
    def test_keeps_the_first_frame_at_or_past_each_tick(self, make_source):
        container, _frames = make_source(times=[i / 30 for i in range(12)])
        kept = _collect(container, fps=10.0)
        assert [frame.idx for frame in kept] == [0, 3, 6, 9]

    def test_dropped_frames_are_never_converted(self, make_source):
        container, frames = make_source(times=[i / 30 for i in range(12)])
        _collect(container, fps=10.0)
        converted = [index for index, frame in enumerate(frames) if frame.converted]
        assert converted == [0, 3, 6, 9]

    def test_a_rate_above_the_source_keeps_every_frame(self, make_source):
        container, _frames = make_source(times=[i / 30 for i in range(6)])
        kept = _collect(container, fps=120.0)
        assert [frame.idx for frame in kept] == [0, 1, 2, 3, 4, 5]

    def test_a_vfr_gap_skips_the_missed_ticks(self, make_source):
        container, _frames = make_source(times=[0.0, 0.1, 5.0, 5.1])
        kept = _collect(container, fps=1.0)
        assert [frame.idx for frame in kept] == [0, 2]

    def test_the_first_frame_defines_time_zero(self, make_source):
        container, _frames = make_source(times=[100.0, 100.5, 101.0])
        kept = _collect(container, fps=2.0)
        assert [frame.time_s for frame in kept] == [0.0, 0.5, 1.0]

    def test_frames_without_a_pts_fall_back_to_the_average_rate(self, make_source):
        container, _frames = make_source(times=[None, None, None], average_rate=Fraction(2, 1))
        kept = _collect(container, fps=2.0)
        assert [frame.time_s for frame in kept] == [0.0, 0.5, 1.0]

    def test_the_pts_fallback_warns_once(self, make_source, caplog):
        container, _frames = make_source(times=[None, None, None], average_rate=Fraction(2, 1))
        with caplog.at_level("WARNING", logger="hudini.video"):
            _collect(container, fps=2.0)
        (record,) = caplog.records
        assert "no pts" in record.message and "2" in record.message

    def test_honest_timestamps_log_no_warning(self, make_source, caplog):
        container, _frames = make_source(times=[0.0, 0.5, 1.0])
        with caplog.at_level("WARNING", logger="hudini.video"):
            _collect(container, fps=2.0)
        assert caplog.records == []

    def test_non_square_pixels_are_resized(self, make_source):
        container, frames = make_source(times=[0.0], sample_aspect_ratio=Fraction(2, 1))
        (frame,) = _collect(container, fps=1.0)
        assert frames[0].reformatted_to == (128, 48)
        assert frame.rgb.shape == (48, 128, 3)

    def test_square_pixels_are_not_reformatted(self, make_source):
        container, frames = make_source(times=[0.0])
        _collect(container, fps=1.0)
        assert frames[0].reformatted_to is None


class TestProbeVideo:
    def test_reports_the_container_facts(self, make_source):
        container, _frames = make_source(times=[0.0])
        stream = container.streams.video[0]
        stream.duration = 3000
        stream.time_base = Fraction(1, 1000)
        with patch("hudini.video.av.open", return_value=container):
            info = probe_video("fake.mp4")
        assert (info.filename, info.codec) == ("fake.mp4", "h264")
        assert (info.width, info.height) == (64, 48)
        assert info.frame_rate == 30.0
        assert info.duration_s == 3.0
        assert info.sha256 is None

    def test_falls_back_to_the_container_duration(self, make_source):
        container, _frames = make_source(times=[0.0], average_rate=None)
        container.duration = 2 * av.time_base
        with patch("hudini.video.av.open", return_value=container):
            info = probe_video("fake.mp4")
        assert info.duration_s == 2.0
        assert info.frame_rate is None


@pytest.mark.integration
def test_a_real_encoded_file_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(25):
            image = np.full((48, 64, 3), fill_value=index * 10 % 255, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    info = probe_video(path)
    assert (info.width, info.height, info.frame_rate) == (64, 48, 25.0)

    kept = list(iter_frames(path, fps=5.0))
    assert [frame.idx for frame in kept] == [0, 5, 10, 15, 20]
    assert kept[0].rgb.shape == (48, 64, 3)
    assert kept[1].time_s == pytest.approx(0.2, abs=1e-3)
