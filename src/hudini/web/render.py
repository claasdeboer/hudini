"""Bakes one video's timeline into one self-contained HTML file.

``write_timeline`` assembles the same page the server serves. It inlines
the static files, the favicon, and one video's data. The file opens
offline and loads nothing from the network. This module does not import
torch or paddleocr.
"""

import base64
import json
from dataclasses import asdict
from pathlib import Path

from hudini.views import Interval

STATIC = Path(__file__).parent / "static"
DATA_TOKEN = "/*__HUDINI_DATA__*/null"
SCRIPT_ORDER = ("shared.js", "data.js", "view_index.js", "view_timeline.js", "app.js")


def assemble(blob: dict | None) -> str:
    """The app as one HTML page: styles, favicon, and scripts inlined.

    ``blob`` is embedded as ``window.HUDINI_EMBEDDED``; None assembles
    the served form of the page, which reads the local API instead.
    """
    page = (STATIC / "app.html").read_text(encoding="utf-8")
    favicon = base64.b64encode((STATIC / "favicon.svg").read_bytes()).decode("ascii")
    page = page.replace(
        '<link rel="icon" href="/static/favicon.svg">',
        f'<link rel="icon" href="data:image/svg+xml;base64,{favicon}">',
    )
    style = (STATIC / "app.css").read_text(encoding="utf-8")
    page = page.replace(
        '<link rel="stylesheet" href="/static/app.css">',
        f"<style>\n{style}\n</style>",
    )
    for name in SCRIPT_ORDER:
        script = (STATIC / name).read_text(encoding="utf-8")
        page = page.replace(
            f'<script src="/static/{name}"></script>',
            f"<script>\n{script}\n</script>",
        )
    if blob is not None:
        page = page.replace(DATA_TOKEN, json.dumps(blob))
    return page


def write_timeline(path: Path, video: dict, intervals: list[Interval]) -> None:
    """Write the standalone timeline page for one log.

    ``video`` is one ``/api/videos`` row, the same shape the served
    app reads, so the two modes cannot drift.
    """
    blob = {"video": video, "intervals": [asdict(interval) for interval in intervals]}
    path.write_text(assemble(blob), encoding="utf-8")
