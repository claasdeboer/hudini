"use strict";
// The index view: one sortable, searchable list row per video.

const INDEX_COLUMNS = [
  { key: "name", label: "Name", numeric: false },
  { key: "duration_s", label: "Duration", numeric: true },
  { key: "instruments", label: "Instruments", numeric: false },
  { key: "yellow", label: "Yellow", numeric: true },
  { key: "blue", label: "Blue", numeric: true },
  { key: "laser", label: "Laser", numeric: true },
  { key: "parsed_at", label: "Parsed", numeric: true },
];

function rowCounts(row) {
  const summary = row.summary;
  return {
    yellow: summary ? summary.presses.yellow.count : null,
    blue: summary ? summary.presses.blue.count : null,
    laser: summary ? summary.laser.count : null,
  };
}

function sortValue(row, key) {
  if (key in rowCounts(row)) return rowCounts(row)[key] ?? -1;
  if (key === "instruments") {
    if (!row.summary) return -1;
    // unique instruments first, tool traffic (swaps) breaks ties
    return row.summary.instruments.length * 1000 + Math.min(row.summary.instrument_changes, 999);
  }
  const value = row[key];
  return value === null || value === undefined ? -1 : value;
}

// The view state lives at module level, so navigating to a timeline and
// back finds the list exactly as it was: same rows, query, sort, focus.
const INDEX_STATE = {
  query: "",
  sort: { key: "name", descending: false },
  focus: 0,
  rows: null,
  total: undefined,
  directory: null,
  vocabulary: null,
};

function renderIndexView(root, data, navigate) {
  const state = INDEX_STATE;
  state.pendingG = false;
  state.suggestions = null;
  state.loading = false;

  const searchInput = el("input", {
    type: "search",
    placeholder: "Search instruments, actions, texts…",
    value: state.query,
    oninput: () => {
      state.query = searchInput.value;
      window.clearTimeout(state.debounce);
      state.debounce = window.setTimeout(fetchListing, 150);
      paintSuggestions();
    },
  });
  const suggestPanel = el("div.suggest");
  const count = el("span.count");
  const listNode = el("div.list");
  const brandSmall = el("small");
  const keyhelp = keyHelpPanel([
    ["j / k · ↓ / ↑", "move the selection"],
    ["Enter", "open the selected log"],
    ["gg / G · Home / End", "first / last row"],
    ["/", "search"],
    ["r", "rescan the archive"],
    ["?", "this overlay"],
    ["Esc", "close, else clear the search"],
  ]);
  const refreshButton = el(
    "button.iconbtn",
    { title: "Rescan the archive (r)", onclick: rescan },
    icon("arrow_path", 17)
  );
  root.replaceChildren(
    el(
      "header.page-header",
      {},
      el("span.brand", {}, "hudini", brandSmall),
      el(
        "span.h-center",
        {},
        el("span.search", {}, icon("magnifying_glass", 14), searchInput, el("kbd", { text: "/" }), suggestPanel),
        count
      ),
      el(
        "span.h-right",
        {},
        refreshButton,
        el(
          "button.iconbtn",
          { title: "Keyboard shortcuts (?)", onclick: () => keyhelp.classList.toggle("open") },
          icon("question_mark_circle", 17)
        )
      )
    ),
    listNode,
    keyhelp
  );

  function sorted() {
    const { key, descending } = state.sort;
    const rows = [...state.rows];
    rows.sort((a, b) => {
      const left = sortValue(a, key);
      const right = sortValue(b, key);
      const order = left < right ? -1 : left > right ? 1 : a.name < b.name ? -1 : 1;
      return descending ? -order : order;
    });
    return rows;
  }

  function setSort(key) {
    if (state.sort.key === key) state.sort.descending = !state.sort.descending;
    else state.sort = { key, descending: key !== "name" };
    paint();
  }

  function chipFor(use) {
    return el(
      "span.chip",
      {},
      el("span.dot", { style: `background:${classColor(use.type)}` }),
      titleCase(use.name)
    );
  }

  function instrumentsCell(row) {
    if (row.summary === undefined) return el("span.num.none", { text: "—" });
    const cell = el("span.rowchips");
    const uses = row.summary ? row.summary.instruments : [];
    for (const use of uses.slice(0, 3)) cell.append(chipFor(use));
    if (uses.length > 3) {
      const rest = uses.slice(3).map((use) => titleCase(use.name)).join(", ");
      cell.append(el("span.chip.more", { title: rest, text: `+${uses.length - 3}` }));
    }
    return cell;
  }

  function numberCell(value, color, title) {
    if (value === null || value === undefined) return el("span.num.none", { text: "—" });
    if (value === 0) return el("span.num.none", { text: "0" });
    const cell = el(
      "span.num",
      {},
      el("span.dot", { style: `background:${color}` }),
      String(value)
    );
    if (title) cell.title = title;
    return cell;
  }

  // The per-action split of one pedal's presses, as the cell's hover
  // text — one "12× COAG" line per action.
  function pedalActionsTitle(row, color) {
    const actions = row.summary?.actions_by_pedal?.[color];
    if (!actions) return null;
    return (
      Object.entries(actions)
        .map(([action, count]) => `${count}× ${action.toUpperCase()}`)
        .join("\n") || null
    );
  }

  // The no-UI marker: a warning triangle in the name cell when the log
  // has spans where the parser saw no interface at all. Pinned to the
  // cell edge so long names cannot truncate it and the markers align
  // vertically.
  function nameCell(row) {
    const cell = el("span.name", {}, el("span.ntext", { text: row.name }));
    const noUi = row.summary?.no_ui_duration_s;
    if (noUi > 0 && row.duration_s > 0) {
      const share = Math.round((100 * noUi) / row.duration_s);
      const marker = el("span.noui-warn", {
        title: `No UI for ${formatClock(noUi)} (${share < 1 ? "<1" : share}% of the video)`,
      });
      marker.append(icon("exclamation_triangle", 13));
      cell.append(marker);
    }
    return cell;
  }

  function parsedCell(row) {
    if (row.state === "running") {
      const percent = row.progress === null ? "…" : `${Math.round(row.progress * 100)}%`;
      // The percent gets a fixed slot, so the dot keeps one x-position
      // whether the number has one digit or three.
      return el("span.pulse", {}, el("span.pct", { text: percent }));
    }
    if (row.state === "outdated") return el("span.ldate", { text: "re-stamp" });
    return el("span.ldate", { text: formatDateTime(row.parsed_at) });
  }

  function rowNode(row, position) {
    const counts = rowCounts(row);
    const focused = position === state.focus;
    const node = el(
      `div.lrow${focused ? ".focused" : ""}`,
      { onclick: () => navigate(row.name), role: "link", tabindex: "-1" },
      nameCell(row),
      el("span.dur", { text: row.duration_s === null ? "—" : formatClock(row.duration_s) }),
      row.state === "running"
        ? el("span.num.none", { text: "—" })
        : instrumentsCell(row),
      numberCell(counts.yellow, MARK.pedalYellow, pedalActionsTitle(row, "yellow")),
      numberCell(counts.blue, MARK.pedalBlue, pedalActionsTitle(row, "blue")),
      row.summary ? numberCell(counts.laser, MARK.laser) : el("span.num.none", { text: "—" }),
      parsedCell(row)
    );
    return node;
  }

  function paint() {
    if (!state.rows) return;
    const rows = sorted();
    state.painted = rows;
    state.focus = Math.min(state.focus, Math.max(0, rows.length - 1));
    count.replaceChildren(
      ...(state.query
        ? [el("b", { text: String(rows.length) }), ` of ${state.total} videos`]
        : [el("b", { text: String(state.total ?? 0) }), " videos"])
    );
    refreshButton.classList.toggle("busy", state.loading);
    const header = el("div.cols");
    for (const column of INDEX_COLUMNS) {
      const active = column.key === state.sort.key;
      const arrow = state.sort.descending ? "↓" : "↑";
      // The arrow goes on the un-anchored side, so the label never jumps:
      // right-aligned columns prepend it, left-aligned columns append it.
      const label = !active
        ? column.label
        : column.numeric
          ? `${arrow} ${column.label}`
          : `${column.label} ${arrow}`;
      const cell = el(`span${column.numeric ? ".num" : ""}${active ? ".active-sort" : ""}`);
      cell.append(el("button", { text: label, onclick: () => setSort(column.key) }));
      header.append(cell);
    }
    listNode.replaceChildren(header, ...rows.map(rowNode));
    if (!rows.length) {
      listNode.append(
        el(
          "div.empty-note",
          {},
          "No videos match. ",
          el("button", {
            text: "Clear the search",
            onclick: () => {
              searchInput.value = "";
              state.query = "";
              fetchListing();
            },
          })
        )
      );
    }
  }

  function adopt(listing) {
    state.rows = listing.videos;
    state.directory = listing.directory;
    brandSmall.textContent = state.directory || "";
    if (state.total === undefined || !state.query) state.total = listing.videos.length;
    if (!state.query) buildVocabulary(listing.videos);
    paint();
  }

  async function fetchListing() {
    state.loading = true;
    paint();
    let listing;
    try {
      listing = await data.listVideos(state.query);
    } catch {
      state.loading = false;
      if (!state.rows) {
        listNode.replaceChildren(el("div.empty-note", { text: "The server is not reachable." }));
      }
      paint();
      return;
    }
    state.loading = false;
    adopt(listing);
    if (!state.query) data.saveListing(listing);
  }

  async function rescan() {
    if (state.loading) return;
    state.loading = true;
    paint();
    try {
      await data.refresh();
    } catch {
      // fetchListing reports the unreachable server.
    }
    state.loading = false;
    await fetchListing();
  }

  // The suggestion vocabulary: every searchable string the summaries
  // hold, so the search box doubles as corpus exploration.
  function buildVocabulary(rows) {
    const seen = new Set();
    const vocabulary = [];
    const add = (field, text) => {
      const key = `${field}\u0000${text}`;
      if (text && !seen.has(key)) {
        seen.add(key);
        vocabulary.push({ field, text });
      }
    };
    for (const row of rows) {
      const summary = row.summary;
      if (!summary) continue;
      for (const use of summary.instruments) {
        add("instrument", use.name);
        add("instrument", use.type);
      }
      for (const action of Object.keys(summary.presses_by_action)) add("action", action);
      for (const text of summary.popup_texts) add("popup", text);
      for (const text of summary.banner_texts) add("banner", text);
    }
    state.vocabulary = vocabulary;
  }

  function typedToken() {
    const value = searchInput.value;
    const tail = value.slice(value.lastIndexOf(" ") + 1);
    return tail.slice(tail.indexOf(":") + 1).replaceAll('"', "");
  }

  function paintSuggestions() {
    const token = typedToken();
    if (token.length < 2 || !state.vocabulary) {
      closeSuggestions();
      return;
    }
    const needle = token.toLowerCase();
    state.suggestions = state.vocabulary
      .filter((entry) => entry.text.toLowerCase().includes(needle))
      .slice(0, 8);
    state.suggestIndex = 0;
    if (!state.suggestions.length) {
      closeSuggestions();
      return;
    }
    suggestPanel.classList.add("open");
    suggestPanel.replaceChildren(
      ...state.suggestions.map((entry, position) =>
        el(
          `button.suggest-row${position === state.suggestIndex ? ".sel" : ""}`,
          { onmousedown: (event) => { event.preventDefault(); accept(entry); } },
          el("small", { text: entry.field }),
          el("span", {
            text:
              entry.field === "instrument"
                ? titleCase(entry.text)
                : entry.field === "action"
                  ? entry.text.toUpperCase()
                  : entry.text,
          })
        )
      )
    );
  }

  function closeSuggestions() {
    state.suggestions = null;
    suggestPanel.classList.remove("open");
  }

  function accept(entry) {
    const value = searchInput.value;
    const kept = value.slice(0, value.lastIndexOf(" ") + 1);
    const text = entry.text.includes(" ") ? `"${entry.text}"` : entry.text;
    searchInput.value = `${kept}${entry.field}:${text} `;
    state.query = searchInput.value;
    closeSuggestions();
    searchInput.focus();
    fetchListing();
  }

  function moveSuggestion(delta) {
    if (!state.suggestions?.length) return;
    state.suggestIndex =
      (state.suggestIndex + delta + state.suggestions.length) % state.suggestions.length;
    suggestPanel.querySelectorAll(".suggest-row").forEach((node, position) => {
      node.classList.toggle("sel", position === state.suggestIndex);
    });
  }

  function moveFocus(delta) {
    if (!state.painted?.length) return;
    state.focus = Math.max(0, Math.min(state.painted.length - 1, state.focus + delta));
    paint();
    listNode.querySelectorAll(".lrow")[state.focus]?.scrollIntoView({ block: "nearest" });
  }

  function onKey(event) {
    if (event.target === searchInput) {
      if (state.suggestions?.length) {
        if (event.key === "ArrowDown") moveSuggestion(1);
        else if (event.key === "ArrowUp") moveSuggestion(-1);
        else if (event.key === "Tab" || event.key === "Enter") {
          accept(state.suggestions[state.suggestIndex]);
        } else if (event.key === "Escape") closeSuggestions();
        else return;
        event.preventDefault();
        return;
      }
      if (event.key === "Escape") {
        searchInput.value = "";
        state.query = "";
        searchInput.blur();
        fetchListing();
      }
      if (event.key === "Enter") searchInput.blur();
      return;
    }
    if (event.ctrlKey || event.altKey || event.metaKey) return;
    const pendingG = state.pendingG;
    state.pendingG = false;
    if (event.key === "j" || event.key === "ArrowDown") moveFocus(1);
    else if (event.key === "k" || event.key === "ArrowUp") moveFocus(-1);
    else if (event.key === "g" && !pendingG) state.pendingG = true;
    else if ((event.key === "g" && pendingG) || event.key === "Home") {
      state.focus = 0;
      paint();
    } else if (event.key === "G" || event.key === "End") {
      state.focus = (state.painted?.length || 1) - 1;
      paint();
    } else if (event.key === "Enter" && state.painted?.length) {
      navigate(state.painted[state.focus].name);
    } else if (event.key === "/") {
      event.preventDefault();
      searchInput.focus();
    } else if (event.key === "r") rescan();
    else if (event.key === "?") keyhelp.classList.toggle("open");
    else if (event.key === "Escape" && keyhelp.classList.contains("open")) {
      keyhelp.classList.remove("open");
    } else return;
    event.preventDefault();
  }

  document.addEventListener("keydown", onKey);
  searchInput.addEventListener("blur", () => window.setTimeout(closeSuggestions, 100));
  // Paint whatever is already known — the rows from the last visit, or
  // the localStorage snapshot — then revalidate against the server.
  if (state.rows) {
    brandSmall.textContent = state.directory || "";
    paint();
  } else {
    const snapshot = data.cachedListing();
    if (snapshot) adopt(snapshot);
    else {
      listNode.replaceChildren(
        el("div.empty-note", {}, el("span.pulse", { text: "Scanning the archive…" }))
      );
    }
  }
  fetchListing();
  return () => {
    document.removeEventListener("keydown", onKey);
  };
}
