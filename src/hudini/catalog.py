"""Names the strings a status pod can show, and matches OCR text to them.

The catalog file ``pods.json`` stores the strings the Xi UI can show on
a status pod. Loading expands each entry into its aliases.
:meth:`Catalog.resolve` answers for an exact string, and
:meth:`Catalog.match` for OCR text with errors. :meth:`Catalog.pedal_action`
names the action a pressed pedal delivers. This module does not import
torch or paddleocr.
"""

from dataclasses import dataclass
from pathlib import Path

import msgspec
from rapidfuzz import fuzz, process

from hudini.schema import PedalColor

DEFAULT_MATCH_THRESHOLD = 70
LABEL_MATCH_THRESHOLD = 55


class _RawPedals(msgspec.Struct, forbid_unknown_fields=True):
    yellow: tuple[str, ...]
    blue: tuple[str, ...]


class _RawEntry(msgspec.Struct, forbid_unknown_fields=True):
    name: str
    type: str
    localized: dict[str, str] = {}
    pedals: _RawPedals | None = None
    reload_colors: tuple[str, ...] = ()


class _CatalogFile(msgspec.Struct):
    color_locales: dict[str, dict[str, str]]
    entries: tuple[_RawEntry, ...]


_file_decoder = msgspec.json.Decoder(_CatalogFile)


def _bundled_catalog_path() -> Path:
    return Path(__file__).parent / "assets" / "pods.json"


@dataclass(frozen=True, slots=True)
class PedalAction:
    """The action labels one pedal can show for one instrument.

    Attributes:
        labels: every label the pedal can show, the idle label first. A
            label can switch with a press or with the instrument pose. Empty
            means the pedal has no action on this instrument.
    """

    labels: tuple[str, ...]

    @property
    def idle_label(self) -> str | None:
        """The label the pedal shows at rest."""
        return self.labels[0] if self.labels else None

    @property
    def press_dependent(self) -> bool:
        """Whether the shown label can switch, with a press or with the
        instrument pose."""
        return len(self.labels) > 1


@dataclass(frozen=True, slots=True)
class PedalPair:
    """The two pedals of an instrument."""

    yellow: PedalAction
    blue: PedalAction

    def of(self, color: PedalColor) -> PedalAction:
        """The pedal of one color."""
        return self.yellow if color is PedalColor.YELLOW else self.blue


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One catalog entry, resolved for one locale.

    Attributes:
        name: the canonical English name, the key ``pods.json`` uses.
        display_name: the catalog spelling in this locale.
        type: the clinical or functional class; ``system_message`` for
            non-instrument strings.
        pedals: the pedal facts, or None when the file states none.
            None means unknown, not "no pedals".
    """

    name: str
    display_name: str
    type: str
    pedals: PedalPair | None

    @property
    def is_system_message(self) -> bool:
        """Whether the entry is a system message, not an instrument."""
        return self.type == "system_message"


@dataclass(frozen=True, slots=True)
class ResolvedAlias:
    """The facts an alias was made from.

    Attributes:
        alias: the alias, in catalog spelling.
        reload_color: the canonical color key when the alias is a stapler
            reload variant, None otherwise. Locale-independent, unlike the
            localized color inside ``alias``.
    """

    entry: CatalogEntry
    alias: str
    reload_color: str | None


@dataclass(frozen=True, slots=True)
class ScoredAlias:
    """A resolved alias with its match quality.

    Attributes:
        score: the ``fuzz.ratio`` similarity, rounded to an integer.
    """

    resolution: ResolvedAlias
    score: int


@dataclass(frozen=True, slots=True)
class CatalogMatch:
    """One fuzzy match's full contest, best first.

    Attributes:
        candidates: every entry at or above the threshold, each shown by
            its best alias, ranked by score. Never empty: a match with no
            candidate is returned as None, not as an empty contest.
    """

    candidates: tuple[ScoredAlias, ...]

    @property
    def best(self) -> ScoredAlias:
        """The candidate with the highest score."""
        return self.candidates[0]

    @property
    def margin(self) -> int | None:
        """Points between the winner and the nearest different entry.

        None when no other entry reached the threshold.
        """
        if len(self.candidates) < 2:
            return None
        return self.candidates[0].score - self.candidates[1].score


def _expand_entry(raw: _RawEntry, locale: str, color_names: dict[str, str]) -> list[ResolvedAlias]:
    localized = raw.localized.get(locale)
    pedals = None
    if raw.pedals is not None:
        pedals = PedalPair(
            yellow=PedalAction(labels=raw.pedals.yellow),
            blue=PedalAction(labels=raw.pedals.blue),
        )
    entry = CatalogEntry(
        name=raw.name, display_name=localized or raw.name, type=raw.type, pedals=pedals
    )
    resolutions = [ResolvedAlias(entry=entry, alias=entry.name, reload_color=None)]
    if localized:
        resolutions.append(ResolvedAlias(entry=entry, alias=localized, reload_color=None))
    for key in raw.reload_colors:
        if key not in color_names:
            raise ValueError(
                f"{entry.name!r} uses reload color {key!r}, which color_locales does not "
                f"translate for this locale"
            )
        alias = f"{entry.name} [{color_names[key]}]"
        resolutions.append(ResolvedAlias(entry=entry, alias=alias, reload_color=key))
    return resolutions


class Catalog:
    """The pod catalog for one locale.

    Attributes:
        locale: the locale the catalog was resolved for.
        entries: every entry, in file order.
        aliases: every string form an entry can be found under, in catalog
            spelling. Resolution ignores letter case.

    Raises:
        ValueError: an unknown locale, or two entries claiming one alias.
    """

    def __init__(self, data: _CatalogFile, locale: str = "en") -> None:
        if locale not in data.color_locales:
            raise ValueError(
                f"locale {locale!r} not in this catalog (has: {sorted(data.color_locales)})"
            )
        self.locale = locale
        color_names = data.color_locales[locale]

        entries: list[CatalogEntry] = []
        self._resolutions: dict[str, ResolvedAlias] = {}
        for raw in data.entries:
            resolutions = _expand_entry(raw, locale, color_names)
            entries.append(resolutions[0].entry)
            for resolution in resolutions:
                key = resolution.alias.casefold()
                claimed = self._resolutions.get(key)
                if claimed is not None and claimed.entry is not resolution.entry:
                    raise ValueError(
                        f"alias {resolution.alias!r} belongs to both {claimed.entry.name!r} "
                        f"and {resolution.entry.name!r}"
                    )
                self._resolutions.setdefault(key, resolution)

        self.entries: tuple[CatalogEntry, ...] = tuple(entries)
        self.aliases: tuple[str, ...] = tuple(r.alias for r in self._resolutions.values())

    @classmethod
    def load(cls, path: Path | str | None = None, locale: str = "en") -> "Catalog":
        """Load a catalog file, the bundled ``pods.json`` when no path is given.

        Raises:
            ValueError: the file does not match the catalog file shape.
        """
        data = _file_decoder.decode(Path(path or _bundled_catalog_path()).read_bytes())
        return cls(data, locale=locale)

    def resolve(self, alias: str) -> ResolvedAlias | None:
        """The facts of an exact alias, or None for an unknown one.

        Names in frame records and intervals are aliases too, so they
        resolve here without any string parsing.
        """
        return self._resolutions.get(alias.casefold())

    def match(self, text: str, threshold: int = DEFAULT_MATCH_THRESHOLD) -> CatalogMatch | None:
        """Fuzzily match OCR text against every alias.

        Returns None when no alias reaches ``threshold``. Candidates are
        deduplicated per entry: aliases of one entry are not opponents.
        """
        text = " ".join(text.split())
        if not text:
            return None
        found = process.extract(
            text,
            self.aliases,
            scorer=fuzz.ratio,
            processor=str.casefold,
            score_cutoff=threshold,
            limit=None,
        )
        if not found:
            return None
        best_per_entry: dict[int, ScoredAlias] = {}
        for alias, score, _index in found:
            resolution = self._resolutions[alias.casefold()]
            entry_id = id(resolution.entry)
            if entry_id not in best_per_entry:
                best_per_entry[entry_id] = ScoredAlias(resolution=resolution, score=round(score))
        return CatalogMatch(candidates=tuple(best_per_entry.values()))

    def pedal_action(self, *, instrument: str, color: PedalColor, label: str | None) -> str | None:
        """The action a pressed pedal delivers, or None when unknown.

        None means the catalog cannot answer: an unknown instrument, no
        pedal data, a pedal with no action, or a press-dependent pedal
        whose ``label`` is missing or matches nothing. A pedal that is not
        press-dependent needs no label.
        """
        resolution = self.resolve(instrument)
        if resolution is None or resolution.entry.pedals is None:
            return None
        pedal = resolution.entry.pedals.of(color)
        if not pedal.labels:
            return None
        if not pedal.press_dependent:
            return pedal.idle_label
        if label is None:
            return None
        found = process.extractOne(
            label.lower(), pedal.labels, scorer=fuzz.ratio, score_cutoff=LABEL_MATCH_THRESHOLD
        )
        return found[0] if found else None
