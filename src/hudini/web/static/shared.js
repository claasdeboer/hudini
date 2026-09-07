"use strict";
// Shared vocabulary of both views: the mark colors, the class-color
// mapping, time formatting, and small DOM helpers. The CSS custom
// properties in app.css are the one source of the values; this module
// reads them so canvas drawing and DOM styling cannot drift.

const CSS = getComputedStyle(document.documentElement);
const token = (name) => CSS.getPropertyValue(name).trim();

const INK = { primary: token("--ink"), secondary: token("--ink-2"), muted: token("--ink-3") };
const SURFACE = { page: token("--page"), surface: token("--surface"), raised: token("--raised") };
const MARK = {
  pedalYellow: token("--pedal-yellow"),
  pedalBlue: token("--pedal-blue"),
  laser: token("--laser"),
  warning: token("--warning"),
  accent: token("--accent"),
  hairline: token("--hairline"),
  hairlineStrong: token("--hairline-strong"),
  popup: token("--hairline-strong"),
  endoscope: token("--c-endoscope"),
  other: token("--c-other"),
};

// One color per instrument class, fixed across all pages and videos.
const CLASS_COLORS = {
  grasper: token("--c-grasper"),
  vessel_sealer: token("--c-vessel-sealer"),
  bipolar_grasper: token("--c-bipolar-grasper"),
  cold_scissors: token("--c-cold-scissors"),
  clip_applier: token("--c-clip-applier"),
  stapler: token("--c-stapler"),
  needle_driver: token("--c-needle-driver"),
  monopolar_cautery: token("--c-monopolar-cautery"),
  endoscope: token("--c-endoscope"),
};

// Catalog strings are stored lowercase; the UI is the display boundary.
// Instrument names and classes render in title case, pedal actions upper.
function titleCase(text) {
  return text
    .replaceAll("_", " ")
    .replace(/\S+/g, (word) => word[0].toUpperCase() + word.slice(1));
}

function classColor(type) {
  if (!type) return MARK.other;
  const key = String(type).toLowerCase().replace(/[^a-z]+/g, "_");
  return CLASS_COLORS[key] || MARK.other;
}

const INACTIVE_ALPHA = 0.24;

function withAlpha(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgb(${(n >> 16) & 255} ${(n >> 8) & 255} ${n & 255} / ${alpha})`;
}

function formatClock(seconds, withMs) {
  const clamped = Math.max(0, seconds);
  const h = Math.floor(clamped / 3600);
  const m = Math.floor((clamped % 3600) / 60);
  const s = Math.floor(clamped % 60);
  const base = `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  if (!withMs) return base;
  const ms = Math.round((clamped - Math.floor(clamped)) * 1000);
  return `${base}.${String(ms).padStart(3, "0")}`;
}

function formatSpan(t0, t1) {
  return `${formatClock(t0)} – ${formatClock(t1)}`;
}

function formatDate(iso) {
  return iso ? iso.slice(0, 10) : "—";
}

function formatDateTime(iso) {
  return iso ? iso.slice(0, 16).replace("T", " ") : "—";
}

function chooseTickInterval(span) {
  for (const step of [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]) {
    const count = span / step;
    if (count >= 4 && count <= 14) return step;
  }
  return span / 8;
}

// el("div.lrow", {onclick: fn}, child, "text") -> HTMLElement
function el(spec, attrs, ...children) {
  const [tag, ...classes] = spec.split(".");
  const node = document.createElement(tag || "div");
  if (classes.length) node.className = classes.join(" ");
  for (const [name, value] of Object.entries(attrs || {})) {
    if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else if (name === "text") node.textContent = value;
    else node.setAttribute(name, value);
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(child);
  }
  return node;
}

// The "?" overlay: one table of key rows, shared by both views.
function keyHelpPanel(bindings) {
  const table = el("table");
  for (const [keys, action] of bindings) {
    table.append(el("tr", {}, el("td", {}, el("kbd", { text: keys })), el("td", { text: action })));
  }
  return el("div.keyhelp", {}, el("div.panel", {}, el("h2", { text: "Keys" }), table));
}

// Heroicons outline paths (heroicons.com, MIT), drawn on the 24px grid.
const ICONS = {
  magnifying_glass: "M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z",
  chevron_down: "M19.5 8.25l-7.5 7.5-7.5-7.5",
  arrow_path:
    "M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 " +
    "3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 " +
    "3.182m0-4.991v4.99",
  question_mark_circle:
    "M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 " +
    "3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 " +
    "0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z",
  eye:
    "M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 " +
    "9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007" +
    "-9.963-7.178z M15 12a3 3 0 11-6 0 3 3 0 016 0z",
  eye_slash:
    "M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 " +
    "1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 " +
    "7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 " +
    "7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88",
  exclamation_triangle:
    "M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 " +
    "2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 " +
    "16.126zM12 15.75h.007v.008H12v-.008z",
};

// icon("chevron_down", 12) -> inline SVG stroked with currentColor
function icon(name, size) {
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", size);
  svg.setAttribute("height", size);
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(svgNS, "path");
  path.setAttribute("d", ICONS[name]);
  svg.append(path);
  return svg;
}

// Diagonal stripe fills for off-screen marks, one pattern per bar state.
function stripePattern(context, bright, dark) {
  const tile = document.createElement("canvas");
  tile.width = 8;
  tile.height = 8;
  const draw = tile.getContext("2d");
  draw.fillStyle = dark;
  draw.fillRect(0, 0, 8, 8);
  draw.strokeStyle = bright;
  draw.lineWidth = 3;
  for (const offset of [-8, 0, 8]) {
    draw.beginPath();
    draw.moveTo(offset, 8);
    draw.lineTo(offset + 8, 0);
    draw.stroke();
  }
  return context.createPattern(tile, "repeat");
}
