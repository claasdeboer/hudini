"""Finds popups with personal data in a stored log.

The Xi writes the configured account name into some popup messages.
This module matches the popup texts of a log against bundled templates.
It reports each match as an episode with boxes, ready for redaction. A
finding never contains the variable part of the message. This module
does not import torch or paddleocr.
"""

import re
import string
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import msgspec
from rapidfuzz import fuzz

from hudini.schema import Box, Observation, PopupMessage, PopupStack, Signal
from hudini.views import Patch, PatchAction

MATCH_THRESHOLD = 80
MIN_TAIL_LENGTH = 10
MASK_PREFIX = "masked:"
MASK_RULE = "privacy:mask"


class _RawEntry(msgspec.Struct, forbid_unknown_fields=True):
    id: str
    templates: dict[str, str]


_file_decoder = msgspec.json.Decoder(tuple[_RawEntry, ...])


def _bundled_templates_path() -> Path:
    return Path(__file__).parent / "assets" / "sensitive_popups.json"


@dataclass(frozen=True, slots=True)
class PopupTemplate:
    """One PII popup template for one locale.

    Attributes:
        id: stable identifier of the message, shared across locales.
        locale: the UI language of ``text``.
        text: the full message, lowercase, with ``str.format``
            placeholders for the variable spans.
        tail: the constant text after the last placeholder, normalized
            to lowercase words. The matcher anchors on it, because
            everything before it can be PII.
    """

    id: str
    locale: str
    text: str
    tail: str


def _normalize(text: str) -> str:
    """The form matching sees: casefolded words, no punctuation."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _derive_tail(text: str) -> str | None:
    """The literal text after the last placeholder, whitespace-normalized.

    Returns None when the template has no placeholder.
    """
    segments = list(string.Formatter().parse(text))
    if all(field_name is None for _literal, field_name, _spec, _conv in segments):
        return None
    literal, field_name, _spec, _conv = segments[-1]
    if field_name is not None:
        return ""
    return _normalize(literal)


def load_templates(path: Path | None = None) -> tuple[PopupTemplate, ...]:
    """Load and validate a template file, flattened per locale.

    Args:
        path: a template file. None loads the bundled
            ``sensitive_popups.json``.

    Raises:
        ValueError: a duplicate id, a duplicate tail, a template that is
            not lowercase or has no placeholder, or a derived tail
            shorter than ``MIN_TAIL_LENGTH`` characters. A malformed
            placeholder raises from the template parser.
        msgspec.ValidationError: the file does not match the schema.
    """
    file = path if path is not None else _bundled_templates_path()
    entries = _file_decoder.decode(file.read_bytes())
    templates: list[PopupTemplate] = []
    seen_tails: dict[str, str] = {}
    seen_ids: set[str] = set()
    for entry in entries:
        if entry.id in seen_ids:
            raise ValueError(f"duplicate template id {entry.id!r}")
        seen_ids.add(entry.id)
        for locale, text in entry.templates.items():
            where = f"template {entry.id!r} ({locale})"
            if text != text.lower():
                raise ValueError(f"{where} is not lowercase")
            tail = _derive_tail(text)
            if tail is None:
                raise ValueError(f"{where} has no placeholder")
            if len(tail) < MIN_TAIL_LENGTH:
                raise ValueError(
                    f"{where} derives the tail {tail!r}, shorter than "
                    f"{MIN_TAIL_LENGTH} characters. A short tail over-matches"
                )
            if tail in seen_tails:
                raise ValueError(f"{where} derives the same tail as {seen_tails[tail]}")
            seen_tails[tail] = where
            templates.append(PopupTemplate(id=entry.id, locale=locale, text=text, tail=tail))
    return tuple(templates)


@dataclass(frozen=True, slots=True)
class TimedBox:
    """One matched popup box at one sampled time."""

    time_s: float
    box: Box


@dataclass(frozen=True, slots=True)
class Finding:
    """One episode of one matched PII popup in one column.

    ``start_s`` and ``end_s`` round outward to the sampled frames around
    the episode, so a redaction over ``[start_s, end_s)`` never
    under-covers. A finding carries the template and the boxes, never
    the variable part of the message.

    Attributes:
        score: the best match ratio of the episode, 0-100.
        boxes: one entry per matched message per sampled frame.
        union_box: the smallest box that covers every entry of ``boxes``.
    """

    template: PopupTemplate
    score: float
    column: int
    start_s: float
    end_s: float
    boxes: tuple[TimedBox, ...]
    union_box: Box


def _best_match(
    text: str, templates: Iterable[PopupTemplate]
) -> tuple[PopupTemplate, float] | None:
    normalized = _normalize(text)
    best: tuple[PopupTemplate, float] | None = None
    for template in templates:
        score = fuzz.partial_ratio(template.tail, normalized)
        if score >= MATCH_THRESHOLD and (best is None or score > best[1]):
            best = (template, score)
    return best


def _union(boxes: list[Box]) -> Box:
    x0 = min(box.x for box in boxes)
    y0 = min(box.y for box in boxes)
    x1 = max(box.x + box.w for box in boxes)
    y1 = max(box.y + box.h for box in boxes)
    return Box(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


@dataclass
class _Episode:
    template: PopupTemplate
    column: int
    first_s: float
    last_s: float
    score: float
    boxes: list[TimedBox] = field(default_factory=list)


def _grid_spacing(sample_times: list[float]) -> float:
    if len(sample_times) < 2:
        return 0.0
    return (sample_times[-1] - sample_times[0]) / (len(sample_times) - 1)


def _close(episode: _Episode, sample_times: list[float]) -> Finding:
    before = bisect_left(sample_times, episode.first_s)
    start_s = sample_times[before - 1] if before > 0 else episode.first_s
    after = bisect_right(sample_times, episode.last_s)
    if after < len(sample_times):
        end_s = sample_times[after]
    else:
        end_s = episode.last_s + _grid_spacing(sample_times)
    return Finding(
        template=episode.template,
        score=episode.score,
        column=episode.column,
        start_s=start_s,
        end_s=end_s,
        boxes=tuple(episode.boxes),
        union_box=_union([timed.box for timed in episode.boxes]),
    )


def _stack_matches(
    obs: Observation, templates: tuple[PopupTemplate, ...]
) -> dict[tuple[str, str], tuple[PopupTemplate, float, list[Box]]]:
    """The matched templates of one popup observation, with their boxes."""
    matches: dict[tuple[str, str], tuple[PopupTemplate, float, list[Box]]] = {}
    if not isinstance(obs.value, PopupStack):
        return matches
    for message in obs.value.messages:
        best = _best_match(message.text, templates)
        if best is None:
            continue
        template, score = best
        entry = (template.id, template.locale)
        if entry in matches:
            kept_template, kept_score, boxes = matches[entry]
            matches[entry] = (kept_template, max(kept_score, score), [*boxes, message.box])
        else:
            matches[entry] = (template, score, [message.box])
    return matches


def screen(log: Iterable[Observation], templates: tuple[PopupTemplate, ...]) -> list[Finding]:
    """Find every PII popup episode in a log.

    Walks the popup observations per column. Consecutive observations
    that match one template form one episode, and any popup observation
    of the column without that match closes it. The tails of all locales
    match at once.

    Args:
        log: time-sorted observations, e.g. a stored ``Log``.

    Returns:
        The findings, sorted by ``start_s``.
    """
    observations = list(log)
    sample_times = [obs.time_s for obs in observations if obs.signal is Signal.LAYOUT]
    open_episodes: dict[tuple[int, str, str], _Episode] = {}
    findings: list[Finding] = []
    for obs in observations:
        if obs.signal is not Signal.POPUPS:
            continue
        column = obs.key[0]
        if not isinstance(column, int):
            continue
        matches = _stack_matches(obs, templates)
        for entry_key in [key for key in open_episodes if key[0] == column]:
            _column, template_id, locale = entry_key
            if (template_id, locale) not in matches:
                findings.append(_close(open_episodes.pop(entry_key), sample_times))
        for (template_id, locale), (template, score, boxes) in matches.items():
            entry_key = (column, template_id, locale)
            episode = open_episodes.get(entry_key)
            if episode is None:
                episode = _Episode(
                    template=template,
                    column=column,
                    first_s=obs.time_s,
                    last_s=obs.time_s,
                    score=score,
                )
                open_episodes[entry_key] = episode
            episode.last_s = obs.time_s
            episode.score = max(episode.score, score)
            episode.boxes.extend(TimedBox(time_s=obs.time_s, box=box) for box in boxes)
    for episode in open_episodes.values():
        findings.append(_close(episode, sample_times))
    return sorted(findings, key=lambda finding: (finding.start_s, finding.column))


def matched_texts(log: Iterable[Observation], finding: Finding) -> tuple[str, ...]:
    """The raw texts behind one finding, for local review only.

    The texts contain the PII the finding withholds, so they must not
    enter a shared report.
    """
    texts: list[str] = []
    for obs in log:
        if obs.signal is not Signal.POPUPS or obs.key != (finding.column,):
            continue
        if not finding.start_s <= obs.time_s <= finding.end_s:
            continue
        if not isinstance(obs.value, PopupStack):
            continue
        for message in obs.value.messages:
            best = _best_match(message.text, (finding.template,))
            if best is not None and message.text not in texts:
                texts.append(message.text)
    return tuple(texts)


def mask_patches(log: Iterable[Observation], templates: tuple[PopupTemplate, ...]) -> list[Patch]:
    """Point patches that mask every matched popup text.

    Each patch replaces a matched message text with ``masked:<id>`` and
    keeps its box, so the shared views and the timeline show that a
    sensitive popup was there without showing what it said. Compute the
    patches over the log the views will read, after the corrections, so
    a gap fill cannot reintroduce raw text.
    """
    patches: list[Patch] = []
    for obs in log:
        if obs.signal is not Signal.POPUPS or not isinstance(obs.value, PopupStack):
            continue
        masked: list[PopupMessage] = []
        touched = False
        for message in obs.value.messages:
            best = _best_match(message.text, templates)
            if best is None:
                masked.append(message)
                continue
            template, _score = best
            masked.append(PopupMessage(text=f"{MASK_PREFIX}{template.id}", box=message.box))
            touched = True
        if not touched:
            continue
        patches.append(
            Patch(
                signal=Signal.POPUPS,
                key=obs.key,
                start_s=obs.time_s,
                end_s=obs.time_s,
                action=PatchAction.SET,
                by=MASK_RULE,
                value=PopupStack(messages=tuple(masked)),
            )
        )
    return patches
