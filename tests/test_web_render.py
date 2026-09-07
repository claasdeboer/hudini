"""Unit tests for hudini.web.render."""

import json

from hudini.views import Interval, LaneName
from hudini.web.render import DATA_TOKEN, SCRIPT_ORDER, STATIC, assemble, write_timeline

VIDEO_ROW = {
    "name": "case",
    "state": "complete",
    "duration_s": 120.0,
    "parsed_at": "2026-08-07T14:12:03+02:00",
    "version": "1.0.0",
    "log_version": 1,
    "achieved_fps": 10.0,
    "progress": None,
    "video_file": "case.mp4",
    "summary": {"instruments": [], "popup_count": 0},
}


def test_the_bake_embeds_the_blob_and_stays_self_contained(tmp_path):
    intervals = [
        Interval(lane=LaneName.STATUS, arm=2, column=2, start_s=0.0, end_s=42.0, value="active")
    ]
    path = tmp_path / "case.html"
    write_timeline(path, video=VIDEO_ROW, intervals=intervals)
    page = path.read_text()
    assert DATA_TOKEN not in page
    assert "/static/" not in page
    fetched_nothing = page.replace("http.server", "").replace("http://www.w3.org/2000/svg", "")
    assert "http" not in fetched_nothing
    assert "data:image/svg+xml;base64," in page
    blob = json.loads(page.split("window.HUDINI_EMBEDDED = ", 1)[1].split(";</script>", 1)[0])
    assert blob["video"]["name"] == "case"
    assert blob["intervals"] == [
        {"lane": "status", "arm": 2, "column": 2, "start_s": 0.0, "end_s": 42.0, "value": "active"}
    ]


def test_a_bake_without_intervals_still_renders(tmp_path):
    path = tmp_path / "case.html"
    write_timeline(path, video=VIDEO_ROW, intervals=[])
    assert '"intervals": []' in path.read_text()


def test_the_served_form_keeps_the_data_token(tmp_path):
    page = assemble(None)
    assert DATA_TOKEN in page
    assert "/static/" not in page.split("window.HUDINI_EMBEDDED")[1].split("</script>")[0]


def test_every_declared_script_exists_and_is_inlined():
    page = assemble(None)
    for name in SCRIPT_ORDER:
        assert (STATIC / name).is_file()
        assert f'src="/static/{name}"' not in page
