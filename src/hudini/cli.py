"""Parses videos and inspects logs from the command line.

Every command takes at most one positional input. stdout carries data,
and stderr carries diagnostics and progress. Output is readable by
default, and ``--json`` switches to machine output. Fast commands do not
import torch. The model layer loads inside the handlers that need it.

``--signals`` takes signal names and group tokens. ``all`` selects every
signal. ``pedals`` selects what a pedal press needs: pedals, pedal_label,
arm, and instrument. The group token wins over the signal of the same
name. ``detectors`` selects offscreen and tool_association. A leading
``-`` subtracts, as in ``all,-detectors``.
"""

import argparse
import errno
import json
import logging
import signal
import sys
import warnings
import webbrowser
from collections.abc import Sequence
from enum import StrEnum
from importlib import metadata
from pathlib import Path

import msgspec

from hudini.catalog import Catalog, CatalogEntry, PedalAction
from hudini.corrections import correct, rules_from_settings
from hudini.privacy import (
    Finding,
    load_templates,
    mask_patches,
    matched_texts,
    screen,
)
from hudini.schema import Engine, Log, Observation, PedalColor, Signal
from hudini.storage import load, write_frames
from hudini.views import (
    FrameState,
    LaneName,
    apply_patches,
    frame_view,
    interval_view,
    iter_frame_states,
    lanes,
    snapshot,
    summarize,
)
from hudini.web.render import write_timeline
from hudini.web.server import DEFAULT_HOST, DEFAULT_PORT, make_server

PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)

GROUPS: dict[str, set[Signal]] = {
    "all": set(Signal) - {Signal.LAYOUT},
    "pedals": {Signal.PEDALS, Signal.PEDAL_LABEL, Signal.ARM, Signal.INSTRUMENT},
    "detectors": {Signal.OFFSCREEN, Signal.TOOL_ASSOCIATION},
}


def _signal(token: str) -> Signal:
    members = {member.value: member for member in Signal}
    if token not in members:
        valid = sorted(GROUPS) + sorted(members)
        raise argparse.ArgumentTypeError(f"unknown signal {token!r}; valid: {valid}")
    return members[token]


def signal_selection(text: str) -> set[Signal]:
    """The ``--signals`` value as a signal set: groups and names, with
    ``-`` subtraction.

    Raises:
        argparse.ArgumentTypeError: an unknown token, or a selection
            that ends up empty.
    """
    selected: set[Signal] = set()
    for token in text.split(","):
        token = token.strip()
        name = token.removeprefix("-")
        members = GROUPS.get(name) or {_signal(name)}
        if token.startswith("-"):
            selected -= members
        else:
            selected |= members
    if not selected:
        raise argparse.ArgumentTypeError(f"{text!r} selects nothing")
    return selected


def rate_argument(text: str) -> tuple[Signal, float]:
    """One ``--rate SIGNAL=FPS`` occurrence."""
    name, _, fps = text.partition("=")
    if not fps:
        raise argparse.ArgumentTypeError(f"{text!r} is not SIGNAL=FPS")
    return _signal(name), float(fps)


def model_argument(text: str) -> tuple[str, Path]:
    """One ``--model NAME=PATH`` occurrence."""
    name, _, path = text.partition("=")
    if not path:
        raise argparse.ArgumentTypeError(f"{text!r} is not NAME=PATH")
    return name, Path(path)


def time_argument(text: str) -> float:
    """A moment as seconds or ``h:mm:ss`` (fractions allowed)."""
    total = 0.0
    for part in text.split(":"):
        total = total * 60 + float(part)
    return total


def configure_logging(verbose: bool, quiet: bool) -> None:
    """Configure the root logger, and silence the warnings stream.

    Library warnings (``warnings.warn``) are suppressed unless
    ``verbose``. Log records are not affected.
    """
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        stream=sys.stderr, level=level, format="%(levelname)s %(name)s: %(message)s"
    )
    if not verbose:
        warnings.filterwarnings("ignore")


def parse_command(args: argparse.Namespace) -> int:
    """Parse a video into its log file and print the written paths."""
    from tqdm import tqdm

    from hudini.parser import Parser, log_path
    from hudini.video import probe_video

    configure_logging(verbose=args.verbose, quiet=args.quiet)
    parser = Parser(
        signals=args.signals,
        ocr_model_size=args.ocr_model_size,
        locale=args.locale,
        detector_threshold=args.detector_threshold,
        models=dict(args.model),
        batch_size=args.batch_size,
    )
    duration = probe_video(args.video).duration_s
    with tqdm(
        total=duration,
        unit="s",
        bar_format=PROGRESS_BAR_FORMAT,
        file=sys.stderr,
        disable=args.quiet,
    ) as bar:

        def advance(time_s: float) -> None:
            bar.update(round(time_s - bar.n, 3))

        parser.parse_video(
            args.video,
            output=args.output,
            rates=dict(args.rate),
            fast=args.fast,
            progress=advance,
        )
        if bar.total:
            bar.update(bar.total - bar.n)
    print(log_path(Path(args.video), args.output))
    return 0


def _frame_state_at(log: Log, at_s: float) -> FrameState | None:
    """The last sampled frame at or before ``at_s``, or None before the
    first frame."""
    found = None
    for fs in iter_frame_states(log):
        if fs.time_s > at_s:
            break
        found = fs
    return found


def filter_token(text: str) -> str:
    """A ``--signal`` filter: a signal name, or a lane name for
    ``--show lanes``."""
    valid = {member.value for member in Signal} | {member.value for member in LaneName}
    if text not in valid:
        raise argparse.ArgumentTypeError(f"unknown signal {text!r}; valid: {sorted(valid)}")
    return text


def _observation_rows(observations: list[Observation], signal: str | None) -> list[dict]:
    return [
        msgspec.to_builtins(obs) for obs in observations if signal is None or obs.signal == signal
    ]


def _query_rows(log: Log, show: str, at_s: float, signal: str | None) -> list[dict]:
    if show == "state":
        held = sorted(snapshot(log, at_s).values(), key=lambda obs: (obs.signal, obs.key))
        return _observation_rows(held, signal)
    fs = _frame_state_at(log, at_s)
    if fs is None:
        return []
    if show == "fresh":
        return _observation_rows(list(fs.fresh), signal)
    catalog = Catalog.load(locale=_log_locale(log))
    return [
        {
            "lane": lane.name,
            "arm": lane.arm,
            "column": lane.column,
            "key": list(lane.key),
            "value": value,
        }
        for lane, value in lanes(fs, catalog).items()
        if signal is None or lane.name == signal
    ]


def _log_locale(log: Log) -> str:
    catalog_engine = log.header.engines.get(Engine.CATALOG)
    if catalog_engine is None or catalog_engine.locale is None:
        return "en"
    return catalog_engine.locale


def query_command(args: argparse.Namespace) -> int:
    """Print one moment of a log as formatted JSON."""
    log = load(Path(args.log))
    at_s = args.at
    if at_s is None:
        at_s = log.observations[-1].time_s if log.observations else 0.0
    rows = _query_rows(log, show=args.show, at_s=at_s, signal=args.signal)
    print(json.dumps({"at_s": at_s, args.show: rows}, indent=2))
    return 0


def artifact_stem(log_file: Path) -> str:
    """The name a log's derived artifacts share: the log name without
    its ``.hudini.jsonl.gz`` tail."""
    return log_file.name.removesuffix(".gz").removesuffix(".jsonl").removesuffix(".hudini")


def write_artifacts(log: Log, log_file: Path, *, frames: bool, timeline: bool) -> list[Path]:
    """Write the derived artifacts next to a log: the frames export and
    the standalone timeline page. Applies the header's correction
    rules, then masks matched sensitive popups. Returns the written
    paths."""
    catalog = Catalog.load(locale=_log_locale(log))
    rules = rules_from_settings(log.header.corrections)
    patches = correct(log, rules, catalog)
    corrected = apply_patches(log, patches)
    patches = [*patches, *mask_patches(corrected, load_templates())]
    stem = artifact_stem(log_file)
    written: list[Path] = []
    if frames:
        frames_file = log_file.with_name(stem + ".frames.jsonl.gz")
        write_frames(
            frames_file,
            frame_view(log, patches, catalog),
            video=log.header.video,
            corrections=[rule.settings() for rule in rules],
        )
        written.append(frames_file)
    if timeline:
        timeline_file = log_file.with_name(stem + ".html")
        intervals = interval_view(log, patches, catalog)
        write_timeline(
            timeline_file, video=_bake_row(log, stem, intervals, catalog), intervals=intervals
        )
        written.append(timeline_file)
    return written


def _bake_row(log: Log, name: str, intervals: list, catalog: Catalog) -> dict:
    """The baked page's video row: the ``/api/videos`` shape, with the
    summary recomputed from the same corrected intervals."""
    footer = log.footer
    return {
        "name": name,
        "state": "complete",
        "duration_s": log.header.video.duration_s,
        "parsed_at": log.header.created_at,
        "version": log.header.version,
        "log_version": log.header.log_version,
        "achieved_fps": footer.achieved_fps if footer is not None else None,
        "progress": None,
        "video_file": log.header.video.filename,
        "summary": summarize(intervals, catalog).to_wire(),
    }


def timeline_command(args: argparse.Namespace) -> int:
    """Build the timeline page from a stored log and print each path."""
    log_file = Path(args.log)
    log = load(log_file)
    if log.footer is None:
        print(f"{log_file} is a partial parse; a timeline needs a finished log", file=sys.stderr)
        return 2
    for file in write_artifacts(log, log_file, frames=args.frames, timeline=not args.no_timeline):
        print(file)
    return 0


def _screen_parse(args: argparse.Namespace) -> Log:
    """The screen's targeted parse: layout and popups, popups on every
    decoded frame. Writes the log next to the video and reports its path
    on stderr."""
    from tqdm import tqdm

    from hudini.parser import Parser, log_path
    from hudini.video import probe_video

    parser = Parser(signals={Signal.POPUPS}, ocr_model_size=args.ocr_model_size)
    info = probe_video(args.input)
    rates: dict[Signal | str, float] = {Signal.POPUPS: info.frame_rate or 60.0}
    rates.update(dict(args.rate))
    with tqdm(
        total=info.duration_s,
        unit="s",
        bar_format=PROGRESS_BAR_FORMAT,
        file=sys.stderr,
        disable=args.quiet,
    ) as bar:

        def advance(time_s: float) -> None:
            bar.update(round(time_s - bar.n, 3))

        log = parser.parse_video(args.input, output=args.output, rates=rates, progress=advance)
        if bar.total:
            bar.update(bar.total - bar.n)
    print(f"log written to {log_path(args.input, args.output)}", file=sys.stderr)
    return log


def _finding_line(finding: Finding) -> str:
    box = finding.union_box
    return (
        f"{finding.template.id} ({finding.template.locale}) column {finding.column} "
        f"{finding.start_s:.3f}s-{finding.end_s:.3f}s score {finding.score:.0f} "
        f"box {box.x},{box.y} {box.w}x{box.h}"
    )


def screen_command(args: argparse.Namespace) -> int:
    """Screen a video or a stored log for PII popups.

    Exit codes follow grep: 0 is clean, 1 is findings, 2 is an error.
    """
    configure_logging(verbose=args.verbose, quiet=args.quiet)
    templates = load_templates()
    if args.input.name.endswith(".hudini.jsonl.gz") or args.input.name.endswith(".partial"):
        log = load(args.input)
        if log.footer is None:
            print(
                f"{args.input} is a partial parse; the screen needs a finished log", file=sys.stderr
            )
            return 2
    else:
        log = _screen_parse(args)
    findings = screen(log, templates)
    if args.json:
        print(json.dumps([msgspec.to_builtins(finding) for finding in findings], indent=2))
        return 1 if findings else 0
    popup_rate = log.header.rates.get(Signal.POPUPS)
    if not findings:
        print(f"clean: no sensitive popup matched (popups read at {popup_rate} fps)")
        return 0
    for finding in findings:
        print(_finding_line(finding))
        if args.show_text:
            for text in matched_texts(log, finding):
                print(f"  text: {text}")
    return 1


def frame_command(args: argparse.Namespace) -> int:
    """Parse one image and print its frame record as JSON."""
    import cv2

    from hudini.parser import Parser

    configure_logging(verbose=args.verbose, quiet=args.quiet)

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        print(f"cannot read image {args.image}", file=sys.stderr)
        return 2
    parser = Parser(
        signals=args.signals,
        ocr_model_size=args.ocr_model_size,
        locale=args.locale,
        detector_threshold=args.detector_threshold,
        models=dict(args.model),
    )
    observations = parser.parse_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    catalog = Catalog.load(locale=args.locale)
    (record,) = frame_view(observations, (), catalog)
    print(json.dumps(record, indent=2))
    return 0


def serve_command(args: argparse.Namespace) -> int:
    """Serve an archive directory in the browser until interrupted."""
    if not Path(args.directory).expanduser().is_dir():
        print(f"{args.directory} is not a directory", file=sys.stderr)
        return 2
    try:
        server = make_server(args.directory, host=args.host, port=args.port)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        print(
            f"port {args.port} is already in use, maybe another hudini serve; pick one with --port",
            file=sys.stderr,
        )
        return 2
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print(url)
    if not args.no_browser:
        webbrowser.open(url)
    # KeyboardInterrupt is the intended way to stop a foreground server.
    try:
        with server:
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


class CatalogTopic(StrEnum):
    """What ``hudini catalog`` can list."""

    PODS = "pods"
    POPUPS = "popups"


NO_ACTION_CELL = "—"
UNKNOWN_PEDALS_CELL = "?"
LABEL_SWITCH_ARROW = "→"


def _table(headers: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
    """Align rows under headers, with a rule line between them."""
    widths = [
        max(len(header), *(len(row[column]) for row in rows)) if rows else len(header)
        for column, header in enumerate(headers)
    ]

    def formatted(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()

    rule = "  ".join("─" * width for width in widths)
    return "\n".join([formatted(headers), rule, *(formatted(row) for row in rows)])


def _pedal_cell(entry: CatalogEntry, color: PedalColor) -> str:
    if entry.pedals is None:
        return UNKNOWN_PEDALS_CELL
    pedal = entry.pedals.of(color)
    if pedal.idle_label is None:
        return NO_ACTION_CELL
    if not pedal.press_dependent:
        return pedal.idle_label
    pressed = "/".join(pedal.labels[1:])
    return f"{pedal.idle_label}{LABEL_SWITCH_ARROW}{pressed}"


def _pedal_json(pedal: PedalAction) -> dict:
    return {
        "labels": list(pedal.labels),
        "idle_label": pedal.idle_label,
        "press_dependent": pedal.press_dependent,
    }


def _pods_output(catalog: Catalog, machine: bool) -> str:
    """The pod catalog as a table, or as JSON rows."""
    entries = sorted(
        catalog.entries,
        key=lambda entry: (entry.is_system_message, entry.type, entry.display_name),
    )
    if machine:
        rows = [
            {
                "name": entry.name,
                "display_name": entry.display_name,
                "type": entry.type,
                "pedals": None
                if entry.pedals is None
                else {
                    PedalColor.YELLOW.value: _pedal_json(entry.pedals.yellow),
                    PedalColor.BLUE.value: _pedal_json(entry.pedals.blue),
                },
            }
            for entry in entries
        ]
        return json.dumps(rows, indent=2)
    instruments = [entry for entry in entries if not entry.is_system_message]
    messages = [entry for entry in entries if entry.is_system_message]
    cells = [
        (
            entry.display_name,
            entry.type,
            _pedal_cell(entry, PedalColor.YELLOW),
            _pedal_cell(entry, PedalColor.BLUE),
        )
        for entry in instruments
    ]
    lines = [_table(("Instrument", "Type", "Yellow", "Blue"), cells)]
    if messages:
        lines.append("")
        lines.append(_table(("System message",), [(entry.display_name,) for entry in messages]))
    pedal_cells = [cell for row in cells for cell in row[2:]]
    footnotes = []
    if any(LABEL_SWITCH_ARROW in cell for cell in pedal_cells):
        footnotes.append(f"{LABEL_SWITCH_ARROW}  the label can switch from the first to the second")
    if UNKNOWN_PEDALS_CELL in pedal_cells:
        footnotes.append(f"{UNKNOWN_PEDALS_CELL}  no pedal data in the catalog")
    if footnotes:
        lines.append("")
        lines.extend(footnotes)
    return "\n".join(lines)


def _popups_output(locale: str, machine: bool) -> str | None:
    """The sensitive-popup templates of one locale, or None when it has none."""
    templates = sorted(
        (template for template in load_templates() if template.locale == locale),
        key=lambda template: template.id,
    )
    if not templates:
        return None
    if machine:
        rows = [
            {"id": template.id, "text": template.text, "tail": template.tail}
            for template in templates
        ]
        return json.dumps(rows, indent=2)
    cells = [(template.id, template.text) for template in templates]
    return _table(("Template", "Text"), cells)


def catalog_command(args: argparse.Namespace) -> int:
    """List the bundled knowledge of one locale: pod strings, or popup templates."""
    if CatalogTopic(args.topic) is CatalogTopic.POPUPS:
        output = _popups_output(locale=args.locale, machine=args.json)
        if output is None:
            print(f"no popup templates for locale {args.locale!r}", file=sys.stderr)
            return 2
        print(output)
        return 0
    catalog = Catalog.load(locale=args.locale)
    print(_pods_output(catalog, machine=args.json))
    return 0


def fetch_command(args: argparse.Namespace) -> int:
    """Download every checkpoint and the PP-OCRv6 pair, and print each
    cached path."""
    from hudini.ocr import ensure_models
    from hudini.sensors import REGISTRY, resolve_checkpoint

    files = sorted(
        {checkpoint.file for sensor in REGISTRY.values() for checkpoint in sensor.spec.checkpoints}
    )
    for file in files:
        print(resolve_checkpoint(file))
    for model_path in ensure_models(args.ocr_model_size):
        print(model_path)
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="hudini", description="Da Vinci Xi UI parsing.")
    root.add_argument("--version", action="version", version=f"hudini {metadata.version('hudini')}")
    commands = root.add_subparsers(required=True)

    parse = commands.add_parser("parse", help="parse a video into its observation log")
    parse.add_argument("video", type=Path)
    parse.add_argument("-o", "--output", type=Path, default=None)
    parse.add_argument("--signals", type=signal_selection, default=None)
    parse.add_argument(
        "--rate", action="append", type=rate_argument, default=[], metavar="SIGNAL=FPS"
    )
    parse.add_argument("--fast", action="store_true", help="thrift rates")
    parse.add_argument(
        "--model", action="append", type=model_argument, default=[], metavar="NAME=PATH"
    )
    parse.add_argument("--ocr-model-size", default="tiny", choices=["tiny", "small", "medium"])
    parse.add_argument("--locale", default="en")
    parse.add_argument("--detector-threshold", type=float, default=0.5)
    parse.add_argument("--batch-size", type=int, default=1)
    parse.add_argument("-v", "--verbose", action="store_true")
    parse.add_argument("-q", "--quiet", action="store_true")
    parse.set_defaults(handler=parse_command)

    frame = commands.add_parser("frame", help="parse one image, JSON to stdout")
    frame.add_argument("image", type=Path)
    frame.add_argument("--signals", type=signal_selection, default=None)
    frame.add_argument(
        "--model", action="append", type=model_argument, default=[], metavar="NAME=PATH"
    )
    frame.add_argument("--ocr-model-size", default="tiny", choices=["tiny", "small", "medium"])
    frame.add_argument("--locale", default="en")
    frame.add_argument("--detector-threshold", type=float, default=0.5)
    frame.add_argument("-v", "--verbose", action="store_true")
    frame.add_argument("-q", "--quiet", action="store_true")
    frame.set_defaults(handler=frame_command)

    screen = commands.add_parser(
        "screen", help="find privacy-sensitive popups in a video or a stored log"
    )
    screen.add_argument("input", type=Path)
    screen.add_argument(
        "-o", "--output", type=Path, default=None, help="log destination for a video input"
    )
    screen.add_argument(
        "--rate", action="append", type=rate_argument, default=[], metavar="SIGNAL=FPS"
    )
    screen.add_argument("--ocr-model-size", default="tiny", choices=["tiny", "small", "medium"])
    screen.add_argument("--json", action="store_true")
    screen.add_argument(
        "--show-text", action="store_true", help="also print the raw matched text (contains PII)"
    )
    screen.add_argument("-v", "--verbose", action="store_true")
    screen.add_argument("-q", "--quiet", action="store_true")
    screen.set_defaults(handler=screen_command)

    timeline = commands.add_parser("timeline", help="build the timeline page from a stored log")
    timeline.add_argument("log", type=Path)
    timeline.add_argument("--frames", action="store_true", help="also write the per-frame export")
    timeline.add_argument("--no-timeline", action="store_true", help="skip the HTML page")
    timeline.set_defaults(handler=timeline_command)

    query = commands.add_parser("query", help="inspect a log, .partial included")
    query.add_argument("log", type=Path)
    query.add_argument("--at", type=time_argument, default=None, metavar="SECONDS|H:MM:SS")
    query.add_argument("--show", choices=["state", "fresh", "lanes"], default="state")
    query.add_argument("--signal", type=filter_token, default=None)
    query.set_defaults(handler=query_command)

    serve = commands.add_parser("serve", help="serve an archive directory in the browser")
    serve.add_argument("directory", nargs="?", default=".", type=Path)
    serve.add_argument("--host", default=DEFAULT_HOST, help="bind address; localhost on purpose")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--no-browser", action="store_true", help="do not open the browser")
    serve.set_defaults(handler=serve_command)

    catalog = commands.add_parser(
        "catalog", help="list the bundled knowledge: pod strings, or popup templates"
    )
    catalog.add_argument(
        "topic",
        nargs="?",
        choices=[topic.value for topic in CatalogTopic],
        default=CatalogTopic.PODS.value,
        help="what to list (default: pods)",
    )
    catalog.add_argument("--locale", default="en")
    catalog.add_argument("--json", action="store_true")
    catalog.set_defaults(handler=catalog_command)

    fetch = commands.add_parser(
        "fetch", help="download every model checkpoint into the local cache"
    )
    fetch.add_argument("--ocr-model-size", default="tiny", choices=["tiny", "small", "medium"])
    fetch.set_defaults(handler=fetch_command)

    return root


def main(argv: list[str] | None = None) -> int:
    """The hudini entry point."""
    args = build_argument_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    # Die silently on a closed pipe (`hudini catalog | head`), like any
    # well-behaved unix tool. Windows has no SIGPIPE.
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    raise SystemExit(main())
