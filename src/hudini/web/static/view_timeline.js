"use strict";
// The timeline view: viewer, transport, overview, ruler, lanes, legend.
// The plot is one canvas; its layout is a pure function of the
// intervals and the enabled lane layers.

const LAYERS = [
  { key: "banner", label: "Banner" },
  { key: "popups", label: "Popups" },
  { key: "instruments", label: "Instruments" },
  { key: "pedals", label: "Pedal Presses" },
  { key: "laser", label: "Laser" },
  { key: "offscreen", label: "Off-screen" },
  { key: "association", label: "Tool Association" },
  { key: "warning", label: "Warning" },
];
const DEFAULT_LAYERS = {
  banner: true,
  popups: true,
  instruments: true,
  pedals: true,
  laser: true,
  offscreen: false,
  association: false,
  warning: false,
};
const INDICATOR_KEYS = ["offscreen", "association", "warning"];
// The layer that must be on for a lane's runs to count as events (w/b).
// Lanes not listed (no_ui) always count. Status is special-cased: it
// feeds both the instrument band and the warning strip.
const LANE_LAYER = {
  banner: "banner",
  popup: "popups",
  instrument: "instruments",
  role: "instruments",
  pedal_yellow: "pedals",
  pedal_blue: "pedals",
  laser: "laser",
  offscreen: "offscreen",
  tool_association: "association",
};
const LAYER_STORE = "hudini.layers";

const STRIP_H = { popup: 10, band: 16, indicator: 6, pedals: 5, laser: 5 };
const GROUP_PAD = 4;
const BANNER_ROW_H = 18;
const MIN_SPAN_S = 1.0;
const MIN_MARK_PX = 2;
// Marks outside the active filter fade almost out; INACTIVE_ALPHA is
// too gentle to separate matches from the rest.
const FILTERED_ALPHA = 0.1;

function loadLayers() {
  const stored = window.localStorage.getItem(LAYER_STORE);
  const layers = { ...DEFAULT_LAYERS };
  if (!stored) return layers;
  for (const [key, value] of Object.entries(JSON.parse(stored))) {
    if (key in layers) layers[key] = Boolean(value);
  }
  return layers;
}

function byLane(intervals, lane) {
  return intervals.filter((entry) => entry.lane === lane);
}

function forArm(intervals, arm) {
  return intervals.filter((entry) => entry.arm === arm);
}

function runAt(runs, t) {
  for (const run of runs) if (run.start_s <= t && t < run.end_s) return run;
  return null;
}

// Merge overlapping runs into fixed-height blocks with a count — the
// popup strip is one row, always.
function mergeRuns(runs) {
  const sorted = [...runs].sort((a, b) => a.start_s - b.start_s);
  const merged = [];
  for (const run of sorted) {
    const last = merged[merged.length - 1];
    if (last && run.start_s < last.end_s) {
      last.end_s = Math.max(last.end_s, run.end_s);
      last.members.push(run);
    } else {
      merged.push({ start_s: run.start_s, end_s: run.end_s, members: [run] });
    }
  }
  return merged;
}

function armIdentities(intervals) {
  const arms = new Set();
  for (const entry of intervals) {
    if (entry.arm !== null && entry.lane !== "no_ui" && entry.lane !== "banner") {
      arms.add(entry.arm);
    }
  }
  // The Xi always shows four columns; assume them until intervals say otherwise,
  // so the empty skeleton already has the final lane heights.
  if (!arms.size) return [1, 2, 3, 4];
  return [...arms].sort((a, b) => a - b);
}

// The per-arm band: elementary segments colored by instrument class
// (endoscope while the arm holds the camera), dimmed when the pod is
// not active. Hue says identity, brightness says activity.
function bandSegments(intervals, arm, typeOf) {
  const status = byLane(forArm(intervals, arm), "status");
  const instrument = byLane(forArm(intervals, arm), "instrument");
  const role = byLane(forArm(intervals, arm), "role");
  const bounds = new Set();
  for (const run of [...status, ...instrument, ...role]) {
    bounds.add(run.start_s);
    bounds.add(run.end_s);
  }
  const edges = [...bounds].sort((a, b) => a - b);
  const segments = [];
  for (let i = 0; i + 1 < edges.length; i += 1) {
    const mid = (edges[i] + edges[i + 1]) / 2;
    const statusRun = runAt(status, mid);
    const instrumentRun = runAt(instrument, mid);
    const roleRun = runAt(role, mid);
    if (!statusRun && !instrumentRun && !roleRun) continue;
    const camera = roleRun !== null && roleRun.value === "camera";
    const color = camera ? MARK.endoscope : classColor(instrumentRun ? typeOf(instrumentRun.value) : null);
    const active = statusRun === null || statusRun.value === "active" || statusRun.value === "warning";
    segments.push({
      t0: edges[i],
      t1: edges[i + 1],
      color,
      alpha: active ? 1 : INACTIVE_ALPHA,
      name: camera ? "Endoscope" : instrumentRun ? titleCase(instrumentRun.value) : "Unknown instrument",
    });
  }
  return segments;
}

// The vertical layout: banner row, then one uniform group per arm.
// Heights change with the enabled layers, never between arms.
function laneLayout(intervals, layers, typeOf) {
  const rows = [];
  let y = 0;
  if (layers.banner) {
    rows.push({ kind: "banner", top: y, height: BANNER_ROW_H, sep: true });
    y += BANNER_ROW_H;
  }
  const strips = [];
  if (layers.popups) strips.push({ kind: "popup", height: STRIP_H.popup });
  if (layers.instruments) strips.push({ kind: "band", height: STRIP_H.band });
  for (const key of INDICATOR_KEYS) {
    if (layers[key]) strips.push({ kind: key, height: STRIP_H.indicator });
  }
  if (layers.pedals) strips.push({ kind: "pedals", height: STRIP_H.pedals });
  if (layers.laser) strips.push({ kind: "laser", height: STRIP_H.laser });
  let offset = GROUP_PAD;
  for (const strip of strips) {
    strip.top = offset;
    offset += strip.height + GROUP_PAD;
  }
  const groupHeight = Math.max(offset + 1, BANNER_ROW_H);
  const arms = armIdentities(intervals);
  arms.forEach((arm, position) => {
    rows.push({
      kind: "arm",
      arm,
      top: y,
      height: groupHeight,
      strips,
      sep: position < arms.length - 1,
      segments: bandSegments(intervals, arm, typeOf),
    });
    y += groupHeight;
  });
  return { rows, height: y, arms, strips };
}

function stripRect(row, kind) {
  const strip = row.strips.find((entry) => entry.kind === kind);
  return strip ? { top: row.top + strip.top, height: strip.height } : null;
}

function renderTimelineView(root, data, name, navigateBack) {
  const state = {
    layers: loadLayers(),
    view: { t0: 0, span: 1 },
    duration: 1,
    intervals: [],
    row: null,
    marks: [],
    fps: 1,
    pendingG: false,
    dragging: false,
    suppressClick: false,
    filter: null,
    matches: null,
    vocabulary: [],
    suggestions: [],
    suggestIndex: -1,
    cleanup: [],
  };

  const typeMap = new Map();
  const typeOf = (name) => typeMap.get(name) || null;

  // -- static skeleton -----------------------------------------------------
  const hstats = el("span.hstats");
  const video = el("video", { preload: "metadata" });
  const videoWrap = el("div.video-wrap", {}, video);
  const videoHint = el("div.video-hint", { hidden: "hidden" });
  const timecode = el("span.tc-big", { text: "00:00:00.000 " }, el("small"));
  const zoomLabel = el("span.zoom", { text: "1.0×" });
  const lanePanel = el("div.lanepanel");
  const lanesButton = el("button.select", {
    onclick: () => lanePanel.classList.toggle("open"),
  });
  const overviewCanvas = el("canvas");
  const miniWindow = el("div.mini-window");
  const ticks = el("div.ticks");
  const rail = el("div.rail");
  const plotCanvas = el("canvas");
  const playhead = el("div.playhead");
  const selectBox = el("div.selection", { hidden: "hidden" });
  const hoverLine = el("div.hoverline", { hidden: "hidden" }, el("span.t"));
  const plotWrap = el("div.plot-wrap", {}, plotCanvas, playhead, selectBox, hoverLine);
  const legend = el("div.legend");
  const colophon = el("div.colophon");
  const tooltip = el("div.tooltip");
  const searchInput = el("input", { type: "text", placeholder: "filter events…" });
  const searchSuggest = el("div.suggest");
  const searchBox = el(
    "span.t-search",
    {},
    icon("magnifying_glass", 13),
    searchInput,
    el("kbd", { text: "/" }),
    searchSuggest
  );
  const filterChip = el("button.filter-chip", { hidden: "hidden", onclick: () => clearFilter() });
  const keyhelp = keyHelpPanel([
    ["Space", "play or pause"],
    ["h / l · , / .", "one frame back / forward"],
    ["j / k · ← / →", "5 seconds back / forward"],
    ["w / b", "next / previous event"],
    ["gg / G · Home / End", "video start / end"],
    ["zi / zo · + / −", "zoom in / out at the playhead"],
    ["zz", "center on the playhead"],
    ["zf", "fit the whole video"],
    ["i", "toggle the indicator layers"],
    ["/", "filter events"],
    ["?", "this overlay"],
    ["Esc", "close, else back to the index"],
    ["drag", "zoom into the selected span"],
    ["double-click", "fit the whole video"],
    ["scroll", "pan the view"],
    ["ctrl + scroll", "zoom at the cursor"],
  ]);

  root.replaceChildren(
    el(
      "header.page-header",
      {},
      el("span.brand", {}, "hudini", el("small", { text: name })),
      el("span"),
      el("span.h-right", {}, hstats)
    ),
    el(
      "div.tl-shell",
      {},
      videoWrap,
      videoHint,
      el(
        "div.transport",
        {},
        el("span.t-left", {}, lanesButton, lanePanel, searchBox, filterChip),
        timecode,
        el(
          "span.t-right",
          {},
          zoomLabel,
          el("button.plain", { text: "Fit", onclick: () => setView(0, state.duration) }),
          el(
            "button.iconbtn",
            { title: "Keyboard shortcuts (?)", onclick: () => keyhelp.classList.toggle("open") },
            icon("question_mark_circle", 17)
          )
        )
      ),
      el("div.overview-row", {}, el("div.overview-wrap", {}, overviewCanvas, miniWindow)),
      ticks,
      el("div.lanes", {}, rail, plotWrap),
      legend,
      colophon
    ),
    tooltip,
    keyhelp
  );

  // -- layers --------------------------------------------------------------
  function paintLanePanel() {
    const on = LAYERS.filter((layer) => state.layers[layer.key]).length;
    lanesButton.replaceChildren(
      `Lanes ${on}/${LAYERS.length} `,
      el("span.chev", {}, icon("chevron_down", 11))
    );
    lanePanel.replaceChildren(
      ...LAYERS.map((layer) =>
        el(
          `button.lanerow${state.layers[layer.key] ? ".on" : ""}`,
          { onclick: () => toggleLayer(layer.key) },
          icon(state.layers[layer.key] ? "eye" : "eye_slash", 13),
          layer.label
        )
      )
    );
    if (on < LAYERS.length) {
      lanePanel.append(
        el("button.lanerow.all", { onclick: showAllLanes }, icon("eye", 13), "Show all")
      );
    }
  }

  function showAllLanes() {
    for (const layer of LAYERS) state.layers[layer.key] = true;
    window.localStorage.setItem(LAYER_STORE, JSON.stringify(state.layers));
    paintLanePanel();
    layoutAndDraw();
  }

  function toggleLayer(key) {
    state.layers[key] = !state.layers[key];
    window.localStorage.setItem(LAYER_STORE, JSON.stringify(state.layers));
    paintLanePanel();
    layoutAndDraw();
  }

  function toggleIndicators() {
    const anyOff = INDICATOR_KEYS.some((key) => !state.layers[key]);
    for (const key of INDICATOR_KEYS) state.layers[key] = anyOff;
    window.localStorage.setItem(LAYER_STORE, JSON.stringify(state.layers));
    paintLanePanel();
    layoutAndDraw();
  }

  // -- search --------------------------------------------------------------
  // Folded text is the match space: lowercase, umlauts and ß flattened,
  // hyphenated line breaks joined, punctuation collapsed to single
  // spaces. OCR variants of one popup
  // ("gemäss", "gemass", "schneidefunk- tion") fold to one form, so a
  // query in any spelling finds them all.
  function foldText(text) {
    return String(text)
      .toLowerCase()
      .replace(/(\w)-\s+(\w)/g, "$1$2")
      .replace(/ß/g, "ss")
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-z0-9]+/g, " ")
      .trim();
  }

  // One pass builds a folded haystack per run and the suggestion
  // vocabulary: only things this log contains, with counts. Suggestions
  // carry a canonical needle so labels can be pretty ("Off-screen")
  // while the match runs on folded text. Entries dedup by needle, so
  // popup spelling variants merge into one suggestion.
  function buildSearchIndex() {
    const vocabulary = new Map();
    const note = (kind, label, needle) => {
      const key = `${kind}|${needle}`;
      const entry = vocabulary.get(key) || { kind, label, needle, count: 0 };
      entry.count += 1;
      vocabulary.set(key, entry);
    };
    for (const run of state.intervals) {
      const type = run.lane === "instrument" ? typeOf(run.value) : null;
      run.search = foldText(`${run.lane} ${run.value ?? ""} ${type ?? ""}`);
      if (run.lane === "instrument" && run.value) {
        note("instrument", titleCase(run.value), foldText(run.value));
      } else if (run.lane === "pedal_yellow" || run.lane === "pedal_blue") {
        const fallback = run.lane === "pedal_yellow" ? "Yellow press" : "Blue press";
        note("pedal", run.value ? run.value.toUpperCase() : fallback, foldText(run.value || run.lane));
      } else if (run.lane === "laser") note("lane", "Laser", "laser");
      else if (run.lane === "offscreen") note("lane", "Off-screen", "offscreen");
      else if (run.lane === "tool_association") {
        note("lane", "Tool Association", foldText("tool_association"));
      } else if (run.lane === "popup") {
        note("lane", "Popups", "popup");
        if (run.value) note("popup", run.value, foldText(run.value));
      } else if (run.lane === "no_ui") note("lane", "No UI", foldText("no_ui"));
      else if (run.lane === "status" && run.value === "warning") {
        note("lane", "Warning", "warning");
      }
    }
    state.vocabulary = [...vocabulary.values()].sort((a, b) => b.count - a.count);
  }

  function applyFilter(needle, label) {
    needle = foldText(needle);
    state.matches = new Set(state.intervals.filter((run) => run.search.includes(needle)));
    state.filter = { needle, label };
    // A filter states intent: matches on a hidden lane turn that lane on
    // instead of filtering into a blank view.
    let revealed = false;
    for (const run of state.matches) {
      if (visibleEvent(run)) continue;
      const layer =
        run.lane === "status"
          ? run.value === "warning"
            ? "warning"
            : "instruments"
          : LANE_LAYER[run.lane];
      if (layer && !state.layers[layer]) {
        state.layers[layer] = true;
        revealed = true;
      }
    }
    filterChip.hidden = false;
    filterChip.replaceChildren(`${label} · ${state.matches.size}`, el("span.x", { text: "✕" }));
    searchBox.hidden = true;
    searchSuggest.classList.remove("open");
    searchInput.value = "";
    searchInput.blur();
    if (revealed) {
      window.localStorage.setItem(LAYER_STORE, JSON.stringify(state.layers));
      paintLanePanel();
      layoutAndDraw();
    } else draw();
    const next = eventBoundaries().find((t) => t > (video.currentTime || 0) + 1e-6);
    if (next !== undefined) seek(next);
  }

  function clearFilter() {
    if (!state.filter) return;
    state.filter = null;
    state.matches = null;
    filterChip.hidden = true;
    searchBox.hidden = false;
    draw();
  }

  // A mark survives the dim when no filter is set, when it is a matching
  // run, when any merged member matches (popup blocks), or — for band
  // segments, which are not runs — when its instrument name matches.
  function markMatches(mark) {
    if (!state.matches) return true;
    if (mark.members) return mark.members.some((member) => state.matches.has(member));
    if (mark.search !== undefined) return state.matches.has(mark);
    return foldText(mark.name || "").includes(state.filter.needle);
  }

  function paintSuggest() {
    const query = foldText(searchInput.value);
    state.suggestions = state.vocabulary
      .filter((option) => option.needle.includes(query) || foldText(option.label).includes(query))
      .slice(0, 8);
    if (state.suggestIndex >= state.suggestions.length) {
      state.suggestIndex = state.suggestions.length - 1;
    }
    searchSuggest.replaceChildren(
      ...state.suggestions.map((option, index) =>
        el(
          `button.suggest-row${index === state.suggestIndex ? ".sel" : ""}`,
          { onclick: () => applyFilter(option.needle, option.label) },
          el("small", { text: option.kind }),
          el("span.sg-label", { text: option.label }),
          el("span.sg-count", { text: String(option.count) })
        )
      )
    );
    searchSuggest.classList.toggle("open", state.suggestions.length > 0);
  }

  function onSearchKey(event) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      const count = state.suggestions.length;
      if (count) {
        const step = event.key === "ArrowDown" ? 1 : -1;
        state.suggestIndex =
          state.suggestIndex === -1 && step === -1
            ? count - 1
            : (state.suggestIndex + step + count) % count;
      }
      paintSuggest();
    } else if (event.key === "Enter") {
      const chosen = state.suggestions[state.suggestIndex];
      const query = searchInput.value.trim();
      if (chosen) applyFilter(chosen.needle, chosen.label);
      else if (query) applyFilter(query.toLowerCase(), query);
    } else if (event.key === "Escape") {
      searchInput.blur();
    } else return;
    event.preventDefault();
  }

  // -- view state ----------------------------------------------------------
  function setView(t0, span) {
    span = Math.min(state.duration, Math.max(MIN_SPAN_S, span));
    t0 = Math.max(0, Math.min(state.duration - span, t0));
    state.view = { t0, span };
    zoomLabel.textContent = `${(state.duration / span).toFixed(1)}×`;
    draw();
  }

  function toX(t, width) {
    return ((t - state.view.t0) / state.view.span) * width;
  }

  // -- layout + drawing ----------------------------------------------------
  function layoutAndDraw() {
    state.layout = laneLayout(state.intervals, state.layers, typeOf);
    paintRail();
    sizeCanvases();
    draw();
  }

  function paintRail() {
    rail.replaceChildren();
    for (const row of state.layout.rows) {
      const node = el(`div.rail-row${row.sep ? ".sep" : ""}`, {
        style: `height:${row.height}px`,
      });
      if (row.kind === "banner") node.append(el("small", { text: "Banner" }));
      else node.append(el("span.armglyph", { text: String(row.arm) }));
      rail.append(node);
    }
  }

  function sizeCanvases() {
    const ratio = window.devicePixelRatio || 1;
    const width = plotWrap.clientWidth || 800;
    plotCanvas.width = Math.round(width * ratio);
    plotCanvas.height = Math.round(state.layout.height * ratio);
    plotCanvas.style.height = `${state.layout.height}px`;
    const overviewWidth = overviewCanvas.parentElement.clientWidth || 800;
    overviewCanvas.width = Math.round(overviewWidth * ratio);
    overviewCanvas.height = Math.round(30 * ratio);
  }

  function draw() {
    drawPlot();
    drawOverview();
    drawTicks();
    positionPlayhead();
  }

  function drawPlot() {
    const context = plotCanvas.getContext("2d");
    const ratio = window.devicePixelRatio || 1;
    const width = plotCanvas.width / ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.fillStyle = SURFACE.surface;
    context.fillRect(0, 0, width, state.layout.height);
    state.marks = [];

    const offscreenActive = stripePattern(context, MARK.warning, "#451a03");
    const offscreenInactive = stripePattern(context, "#52525b", SURFACE.raised);

    const drawRun = (run, rect, style, lines, minWidth = MIN_MARK_PX) => {
      const left = Math.max(0, toX(run.t0 ?? run.start_s, width));
      const right = Math.min(width, toX(run.t1 ?? run.end_s, width));
      if (right < 0 || left > width) return;
      const alpha = context.globalAlpha;
      if (!markMatches(run)) context.globalAlpha = Math.min(alpha, FILTERED_ALPHA);
      context.fillStyle = style;
      context.fillRect(left, rect.top, Math.max(right - left, minWidth), rect.height);
      context.globalAlpha = alpha;
      if (lines) {
        state.marks.push({
          t0: run.t0 ?? run.start_s,
          t1: run.t1 ?? run.end_s,
          top: rect.top,
          height: rect.height,
          lines,
        });
      }
    };

    for (const row of state.layout.rows) {
      if (row.kind === "banner") {
        const rect = { top: row.top + 4, height: 10 };
        for (const block of mergeRuns(byLane(state.intervals, "banner"))) {
          drawRun(block, rect, MARK.popup, [
            block.members.map((run) => run.value || "Banner").join(" · "),
            formatSpan(block.start_s, block.end_s),
          ]);
        }
        continue;
      }
      const armRuns = forArm(state.intervals, row.arm);
      const popupRect = stripRect(row, "popup");
      if (popupRect) {
        for (const block of mergeRuns(byLane(armRuns, "popup"))) {
          const texts = [...new Set(block.members.map((run) => run.value || "Popup"))];
          drawRun(block, popupRect, MARK.popup, [texts.join(" · "), formatSpan(block.start_s, block.end_s)]);
          if (block.members.length > 1) {
            const x = (toX(block.start_s, width) + toX(block.end_s, width)) / 2;
            if (!markMatches(block)) context.globalAlpha = FILTERED_ALPHA;
            context.fillStyle = INK.muted;
            context.font = "9px ui-monospace, monospace";
            context.textAlign = "center";
            context.fillText(String(block.members.length), x, popupRect.top + 8);
            context.globalAlpha = 1;
          }
        }
      }
      const bandRect = stripRect(row, "band");
      if (bandRect) {
        for (const segment of row.segments) {
          context.globalAlpha = segment.alpha;
          drawRun(segment, bandRect, segment.color, [
            segment.name,
            formatSpan(segment.t0, segment.t1),
          ], 1);
          context.globalAlpha = 1;
        }
        // A 1px ground seam where the instrument changes, so each run
        // reads as its own object.
        context.fillStyle = SURFACE.surface;
        let previous = null;
        for (const segment of row.segments) {
          if (previous && previous.name !== segment.name && previous.t1 === segment.t0) {
            context.fillRect(Math.round(toX(segment.t0, width)), bandRect.top, 1, bandRect.height);
          }
          previous = segment;
        }
      }
      const offRect = stripRect(row, "offscreen");
      if (offRect) {
        for (const run of byLane(armRuns, "offscreen")) {
          const inactive = run.value === "inactive";
          const what = `Off-screen${run.value ? `, bar ${run.value}` : ""}`;
          drawRun(run, offRect, inactive ? offscreenInactive : offscreenActive, [
            what,
            formatSpan(run.start_s, run.end_s),
          ]);
        }
      }
      const assocRect = stripRect(row, "association");
      if (assocRect) {
        for (const run of byLane(armRuns, "tool_association")) {
          drawRun(run, assocRect, INK.secondary, [
            "Tool Association",
            formatSpan(run.start_s, run.end_s),
          ]);
        }
      }
      const warnRect = stripRect(row, "warning");
      if (warnRect) {
        for (const run of byLane(armRuns, "status").filter((entry) => entry.value === "warning")) {
          drawRun(run, warnRect, MARK.warning, ["Warning", formatSpan(run.start_s, run.end_s)]);
        }
      }
      const pedalRect = stripRect(row, "pedals");
      if (pedalRect) {
        for (const color of ["yellow", "blue"]) {
          for (const run of byLane(armRuns, `pedal_${color}`)) {
            const action = run.value ? run.value.toUpperCase() : `${titleCase(color)} press`;
            const held = (run.end_s - run.start_s).toFixed(1);
            drawRun(run, pedalRect, color === "yellow" ? MARK.pedalYellow : MARK.pedalBlue, [
              action,
              `${formatClock(run.start_s)} · ${held} s`,
            ]);
          }
        }
      }
      const laserRect = stripRect(row, "laser");
      if (laserRect) {
        for (const run of byLane(armRuns, "laser")) {
          drawRun(run, laserRect, MARK.laser, ["Laser On", formatSpan(run.start_s, run.end_s)]);
        }
      }
    }

    // The void: no-UI spans interrupt every lane. Opaque — runs end at
    // its edges — with the group hairlines drawn back across it.
    for (const run of byLane(state.intervals, "no_ui")) {
      const left = Math.max(0, toX(run.start_s, width));
      const right = Math.min(width, toX(run.end_s, width));
      if (right < 0 || left > width) continue;
      const voidWidth = Math.max(right - left, 1);
      context.fillStyle = SURFACE.page;
      context.fillRect(left, 0, voidWidth, state.layout.height);
      // A matched void cannot brighten the way a mark does — it is
      // absence — so the accent moves to its border instead.
      context.strokeStyle = state.matches?.has(run) ? MARK.accent : MARK.hairline;
      context.strokeRect(left + 0.5, -1, voidWidth - 1, state.layout.height + 2);
      state.marks.push({
        t0: run.start_s,
        t1: run.end_s,
        top: 0,
        height: state.layout.height,
        lines: ["No UI", formatSpan(run.start_s, run.end_s)],
      });
    }
    context.strokeStyle = MARK.hairline;
    for (const row of state.layout.rows) {
      if (!row.sep) continue;
      const bottom = row.top + row.height - 0.5;
      context.beginPath();
      context.moveTo(0, bottom);
      context.lineTo(width, bottom);
      context.stroke();
    }
  }

  function drawOverview() {
    const context = overviewCanvas.getContext("2d");
    const ratio = window.devicePixelRatio || 1;
    const width = overviewCanvas.width / ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.fillStyle = SURFACE.surface;
    context.fillRect(0, 0, width, 30);
    const scale = (t) => (t / state.duration) * width;
    state.layout.arms.slice(0, 4).forEach((arm, position) => {
      const top = 4 + position * 6;
      const row = state.layout.rows.find((entry) => entry.arm === arm);
      for (const segment of row.segments) {
        context.globalAlpha = 0.8 * segment.alpha;
        context.fillStyle = segment.color;
        context.fillRect(scale(segment.t0), top, Math.max(scale(segment.t1) - scale(segment.t0), 1), 5);
      }
    });
    context.globalAlpha = 1;
    context.fillStyle = SURFACE.page;
    for (const run of byLane(state.intervals, "no_ui")) {
      context.fillRect(scale(run.start_s), 0, Math.max(scale(run.end_s) - scale(run.start_s), 1), 30);
    }
    if (state.matches) {
      context.fillStyle = MARK.accent;
      for (const run of state.matches) {
        context.fillRect(scale(run.start_s), 25, 2, 5);
      }
    }
    const left = scale(state.view.t0);
    miniWindow.style.left = `${left}px`;
    miniWindow.style.width = `${Math.max(scale(state.view.t0 + state.view.span) - left, 6)}px`;
  }

  function drawTicks() {
    ticks.replaceChildren();
    if (!state.row) return;
    const step = chooseTickInterval(state.view.span);
    const first = Math.ceil(state.view.t0 / step) * step;
    const width = ticks.clientWidth || 800;
    for (let t = first; t <= state.view.t0 + state.view.span; t += step) {
      const x = (toX(t, width) / width) * 100;
      ticks.append(el("span", { style: `left:${x}%`, text: formatClock(t) }));
    }
  }

  function positionPlayhead() {
    const width = plotWrap.clientWidth || 800;
    const x = toX(video.currentTime || 0, width);
    playhead.style.display = x < 0 || x > width ? "none" : "block";
    playhead.style.left = `${x}px`;
    timecode.replaceChildren(
      `${formatClock(video.currentTime || 0, true)} `,
      el("small", { text: `/ ${formatClock(state.duration)}` })
    );
  }

  // -- chrome below the plot ----------------------------------------------
  // Two legend sections: the mark grammar, identical on every page, and
  // the instrument colors this case happens to use.
  function paintLegend() {
    legend.replaceChildren();
    const group = (label) => {
      const items = el("span.lg-items");
      const node = el("span.lg", {}, el("span.lg-label", { text: label }), items);
      legend.append(node);
      return {
        node,
        item: (swatchStyle, text) =>
          items.append(el("span.item", {}, el("span.sw", { style: swatchStyle }), text)),
      };
    };
    const marks = group("Marks");
    marks.item(`background:${MARK.pedalYellow}`, "Yellow Press");
    marks.item(`background:${MARK.pedalBlue}`, "Blue Press");
    marks.item(`background:${MARK.laser}`, "Laser On");
    marks.item(`background:${MARK.warning}`, "Warning");
    marks.item(
      `width:14px;background:repeating-linear-gradient(135deg, ${MARK.warning} 0 3px, #451a03 3px 6px)`,
      "Off-screen (Bar Active)"
    );
    marks.item(
      "width:14px;background:repeating-linear-gradient(135deg, #52525b 0 3px, #27272a 3px 6px)",
      "Off-screen (Bar Inactive)"
    );
    marks.item(`background:${INK.secondary}`, "Tool Association");
    marks.item(`width:12px;background:${MARK.popup}`, "Popup / Banner");
    const instruments = group("Instruments");
    const seen = new Set();
    for (const type of typeMap.values()) {
      if (seen.has(type)) continue;
      seen.add(type);
      instruments.item(`background:${classColor(type)}`, titleCase(type));
    }
    if (state.layout.rows.some((row) => row.kind === "arm" && row.segments.some((s) => s.name === "Endoscope"))) {
      instruments.item(`background:${MARK.endoscope}`, "Endoscope");
      seen.add("endoscope");
    }
    if (!seen.size) instruments.node.remove();
  }

  function paintHeaderStats() {
    const summary = state.row?.summary;
    hstats.replaceChildren();
    if (!summary) return;
    const stat = (color, label, value) =>
      hstats.append(
        el(
          "span.item",
          {},
          el("span.dot", { style: `background:${color}` }),
          label,
          el("span.v", { text: String(value) })
        )
      );
    stat(MARK.pedalYellow, "", summary.presses.yellow.count);
    stat(MARK.pedalBlue, "", summary.presses.blue.count);
    stat(MARK.laser, "Laser", summary.laser.count);
  }

  function paintColophon() {
    const row = state.row || {};
    const fps = row.achieved_fps === null || row.achieved_fps === undefined ? "—" : row.achieved_fps;
    colophon.textContent =
      `Parsed ${formatDateTime(row.parsed_at)} · hudini ${row.version || "—"}` +
      ` · Log v${row.log_version ?? "—"} · ${fps} fps achieved`;
  }

  // -- interaction ---------------------------------------------------------
  function visibleEvent(entry) {
    if (entry.lane === "status") {
      return state.layers.instruments || (state.layers.warning && entry.value === "warning");
    }
    const layer = LANE_LAYER[entry.lane];
    return layer === undefined || state.layers[layer];
  }

  function eventBoundaries() {
    const times = new Set();
    for (const entry of state.intervals) {
      if (!visibleEvent(entry)) continue;
      if (state.matches && !state.matches.has(entry)) continue;
      times.add(entry.start_s);
      times.add(entry.end_s);
    }
    return [...times].sort((a, b) => a - b);
  }

  function seek(t) {
    video.currentTime = Math.max(0, Math.min(state.duration, t));
    positionPlayhead();
  }

  function onPlotMove(event) {
    if (state.dragging) return;
    const bounds = plotCanvas.getBoundingClientRect();
    const t = state.view.t0 + ((event.clientX - bounds.left) / bounds.width) * state.view.span;
    const y = event.clientY - bounds.top;
    hoverLine.hidden = !state.row;
    hoverLine.style.left = `${event.clientX - bounds.left}px`;
    hoverLine.firstChild.textContent = formatClock(t, state.view.span < 60);
    const hit = state.marks.findLast(
      (mark) => mark.t0 <= t && t < mark.t1 && mark.top <= y && y <= mark.top + mark.height
    );
    if (!hit) {
      tooltip.style.display = "none";
      return;
    }
    tooltip.replaceChildren(el("b", { text: hit.lines[0] }), el("span.when", { text: hit.lines[1] }));
    tooltip.style.display = "block";
    tooltip.style.left = `${Math.min(event.clientX + 14, window.innerWidth - 240)}px`;
    tooltip.style.top = `${event.clientY + 14}px`;
  }

  function onWheel(event) {
    event.preventDefault();
    if (event.ctrlKey || event.altKey) {
      const bounds = plotCanvas.getBoundingClientRect();
      const anchor = state.view.t0 + ((event.clientX - bounds.left) / bounds.width) * state.view.span;
      const factor = event.deltaY > 0 ? 2 : 0.5;
      const span = state.view.span * factor;
      setView(anchor - (anchor - state.view.t0) * factor, span);
    } else {
      const delta = (event.deltaY || event.deltaX) * (state.view.span / 800);
      setView(state.view.t0 + delta, state.view.span);
    }
  }

  function onRulerClick(event) {
    const bounds = ticks.getBoundingClientRect();
    seek(state.view.t0 + ((event.clientX - bounds.left) / bounds.width) * state.view.span);
  }

  function onPlotClick(event) {
    if (state.suppressClick) {
      state.suppressClick = false;
      return;
    }
    onRulerClick(event);
  }

  // Drag horizontally on the lanes to zoom into the selected span; a
  // movement under the threshold stays a click and seeks.
  function onPlotDown(event) {
    if (event.button !== 0) return;
    event.preventDefault();
    const bounds = plotCanvas.getBoundingClientRect();
    const startX = event.clientX;
    const edges = (x) => [
      Math.max(0, Math.min(startX, x) - bounds.left),
      Math.min(bounds.width, Math.max(startX, x) - bounds.left),
    ];
    const move = (moved) => {
      if (!state.dragging && Math.abs(moved.clientX - startX) < 5) return;
      state.dragging = true;
      tooltip.style.display = "none";
      hoverLine.hidden = true;
      const [left, right] = edges(moved.clientX);
      selectBox.hidden = false;
      selectBox.style.left = `${left}px`;
      selectBox.style.width = `${right - left}px`;
    };
    const stop = (release) => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", stop);
      if (!state.dragging) return;
      state.dragging = false;
      selectBox.hidden = true;
      state.suppressClick = true;
      window.setTimeout(() => (state.suppressClick = false), 0);
      const [left, right] = edges(release.clientX);
      const t0 = state.view.t0 + (left / bounds.width) * state.view.span;
      const t1 = state.view.t0 + (right / bounds.width) * state.view.span;
      setView(t0, t1 - t0);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", stop);
  }

  function onOverviewClick(event) {
    const bounds = overviewCanvas.getBoundingClientRect();
    const t = ((event.clientX - bounds.left) / bounds.width) * state.duration;
    setView(t - state.view.span / 2, state.view.span);
  }

  function onWindowDrag(event) {
    event.preventDefault();
    const bounds = overviewCanvas.getBoundingClientRect();
    const startT = state.view.t0;
    const startX = event.clientX;
    const move = (moved) => {
      const delta = ((moved.clientX - startX) / bounds.width) * state.duration;
      setView(startT + delta, state.view.span);
    };
    const stop = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", stop);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", stop);
  }

  function onKey(event) {
    if (event.target instanceof HTMLInputElement) return;
    if (event.ctrlKey || event.altKey || event.metaKey) return;
    const pendingG = state.pendingG;
    const pendingZ = state.pendingZ;
    state.pendingG = false;
    state.pendingZ = false;
    const playheadT = video.currentTime || 0;
    if (pendingZ) {
      if (event.key === "i") setView(playheadT - state.view.span / 4, state.view.span / 2);
      else if (event.key === "o") setView(playheadT - state.view.span, state.view.span * 2);
      else if (event.key === "z") setView(playheadT - state.view.span / 2, state.view.span);
      else if (event.key === "f") setView(0, state.duration);
      else return;
    } else if (event.key === " ") {
      if (video.paused) video.play();
      else video.pause();
    } else if (event.key === "h" || event.key === ",") seek(playheadT - 1 / state.fps);
    else if (event.key === "l" || event.key === ".") seek(playheadT + 1 / state.fps);
    else if (event.key === "j" || event.key === "ArrowLeft") seek(playheadT - 5);
    else if (event.key === "k" || event.key === "ArrowRight") seek(playheadT + 5);
    else if (event.key === "w") {
      const next = eventBoundaries().find((t) => t > playheadT + 1e-6);
      if (next !== undefined) seek(next);
    } else if (event.key === "b") {
      const previous = eventBoundaries().filter((t) => t < playheadT - 1e-6).pop();
      if (previous !== undefined) seek(previous);
    } else if (event.key === "g" && !pendingG) state.pendingG = true;
    else if ((event.key === "g" && pendingG) || event.key === "0" || event.key === "Home") seek(0);
    else if (event.key === "G" || event.key === "$" || event.key === "End") seek(state.duration);
    else if (event.key === "+" || event.key === "=") {
      setView(playheadT - state.view.span / 4, state.view.span / 2);
    } else if (event.key === "-") setView(playheadT - state.view.span, state.view.span * 2);
    else if (event.key === "z") state.pendingZ = true;
    else if (event.key === "i") toggleIndicators();
    else if (event.key === "/") {
      clearFilter();
      searchInput.focus();
    } else if (event.key === "?") keyhelp.classList.toggle("open");
    else if (event.key === "Escape") {
      if (keyhelp.classList.contains("open")) keyhelp.classList.remove("open");
      else if (lanePanel.classList.contains("open")) lanePanel.classList.remove("open");
      else if (state.filter) clearFilter();
      else if (navigateBack) navigateBack();
    } else return;
    event.preventDefault();
  }

  // -- wiring --------------------------------------------------------------
  async function start() {
    paintLanePanel();
    layoutAndDraw();
    videoHint.hidden = false;
    videoHint.textContent = "Deriving the intervals from the log — a long case takes a few seconds…";
    const [listing, intervals] = await Promise.all([data.listVideos(""), data.intervals(name)]);
    videoHint.hidden = true;
    state.row = listing.videos.find((row) => row.name === name) || null;
    state.intervals = intervals;
    const lastEnd = intervals.reduce((end, entry) => Math.max(end, entry.end_s), 0);
    state.duration = state.row?.duration_s || lastEnd || 1;
    state.fps = state.row?.achieved_fps || 1;
    for (const use of state.row?.summary?.instruments || []) typeMap.set(use.name, use.type);
    buildSearchIndex();
    video.src = data.videoUrl(name);
    state.view = { t0: 0, span: state.duration };
    paintLanePanel();
    layoutAndDraw();
    paintLegend();
    paintHeaderStats();
    paintColophon();
    zoomLabel.textContent = "1.0×";
  }

  video.addEventListener("click", () => (video.paused ? video.play() : video.pause()));
  video.addEventListener("timeupdate", positionPlayhead);
  video.addEventListener("error", () => {
    videoWrap.hidden = true;
    videoHint.hidden = false;
    videoHint.textContent =
      "No video file next to this log — the lanes still work; put the video in the archive directory to scrub it.";
  });
  searchInput.addEventListener("input", () => {
    state.suggestIndex = -1;
    paintSuggest();
  });
  searchInput.addEventListener("focus", () => {
    state.suggestIndex = -1;
    paintSuggest();
  });
  searchInput.addEventListener("blur", () => searchSuggest.classList.remove("open"));
  searchInput.addEventListener("keydown", onSearchKey);
  searchSuggest.addEventListener("mousedown", (event) => event.preventDefault());
  plotCanvas.addEventListener("mousemove", onPlotMove);
  plotCanvas.addEventListener("mouseleave", () => {
    tooltip.style.display = "none";
    hoverLine.hidden = true;
  });
  plotCanvas.addEventListener("click", onPlotClick);
  plotCanvas.addEventListener("dblclick", () => setView(0, state.duration));
  plotWrap.addEventListener("mousedown", onPlotDown);
  plotWrap.addEventListener("wheel", onWheel, { passive: false });
  ticks.addEventListener("click", onRulerClick);
  overviewCanvas.addEventListener("click", onOverviewClick);
  miniWindow.addEventListener("mousedown", onWindowDrag);
  document.addEventListener("keydown", onKey);
  const onResize = () => {
    sizeCanvases();
    draw();
  };
  window.addEventListener("resize", onResize);
  start();

  return () => {
    document.removeEventListener("keydown", onKey);
    window.removeEventListener("resize", onResize);
    video.pause();
    video.removeAttribute("src");
  };
}
