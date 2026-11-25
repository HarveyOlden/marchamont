import { CONFIG } from "./config.js";

const $ = (id) => document.getElementById(id);
const els = {
  src: $("src"), tally: $("tally"), year: $("year"), roman: $("roman"), slider: $("slider"), ticks: $("ticks"),
  en: $("t-en"), sc: $("t-sc"), go: $("go"), goLabel: $("go-label"), status: $("status"), hourglass: $("hourglass"),
  sand: $("sand"), region: $("region"), hand: $("hand"), roll: $("roll"), nohand: $("nohand"), liberty: $("liberty"),
  out: $("out"), tail: $("tail"), tools: $("tools"), copy: $("copy"), save: $("save"), band: $("era-band"),
};
$("source").href = CONFIG.source;
$("weights").href = CONFIG.weights;

const state = { scots: false, ready: false, working: false, meta: null, hand: 1 + Math.floor(Math.random() * 9999), noHand: false };
let ranges = { english: [1380, 1790], scots: [1500, 1905] };
const REGIONS = ["<unk>", "england", "scotland", "ireland", "wales", "colonies", "continent"];

/* ---------- year dial ---------- */
function roman(n) {
  const t = [[1000, "M"], [900, "CM"], [500, "D"], [400, "CD"], [100, "C"], [90, "XC"], [50, "L"], [40, "XL"], [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]];
  let s = ""; for (const [v, r] of t) while (n >= v) { s += r; n -= v; } return s;
}
const BANDS = [[1450, "prayer_header"], [1550, "lion_header"], [1650, "king_header"], [1750, "eagle_header"], [9999, "devil_header"]];
function bandFor(y) { return `assets/cuts/${BANDS.find(([lim]) => y < lim)[1]}.png`; }

function drawTicks() {
  const [lo, hi] = state.scots ? ranges.scots : ranges.english;
  els.ticks.innerHTML = "";
  for (let y = Math.ceil(lo / 100) * 100; y <= hi; y += 100) {
    const s = document.createElement("span");
    s.textContent = y; s.style.left = `${((y - lo) / (hi - lo)) * 100}%`;
    els.ticks.appendChild(s);
  }
}
function setYear(y) {
  const [lo, hi] = state.scots ? ranges.scots : ranges.english;
  y = Math.max(lo, Math.min(hi, Math.round(y)));
  els.slider.min = lo; els.slider.max = hi; els.slider.value = y;
  els.year.textContent = y; els.roman.textContent = roman(y);
  const b = bandFor(y);
  if (!els.band.src.endsWith(b)) { els.band.style.opacity = 0; setTimeout(() => { els.band.src = b; els.band.style.opacity = .85; }, 250); }
  const col = els.region.querySelector('option[value="colonies"]');
  col.disabled = y < 1640;
  if (col.disabled && els.region.value === "colonies") els.region.value = "auto";
}
els.slider.addEventListener("input", () => setYear(+els.slider.value));

/* ---------- tongue ---------- */
function setTongue(scots) {
  state.scots = scots;
  els.en.classList.toggle("on", !scots); els.sc.classList.toggle("on", scots);
  els.en.setAttribute("aria-checked", !scots); els.sc.setAttribute("aria-checked", scots);
  drawTicks(); setYear(+els.slider.value);
}
els.en.onclick = () => setTongue(false);
els.sc.onclick = () => setTongue(true);

/* ---------- hand ---------- */
function showHand() { els.hand.value = state.hand; els.hand.disabled = state.noHand; els.nohand.classList.toggle("on", state.noHand); }
els.roll.onclick = () => { state.hand = 1 + Math.floor(Math.random() * 9999); state.noHand = false; showHand(); };
els.nohand.onclick = () => { state.noHand = !state.noHand; showHand(); };
els.hand.onchange = () => { const v = Math.round(+els.hand.value); if (v >= 1 && v <= 9999) state.hand = v; showHand(); };

/* ---------- input ---------- */
function tally() {
  const n = els.src.value.length;
  els.tally.textContent = n > CONFIG.maxChars * 0.8 ? `${n} / ${CONFIG.maxChars}` : "";
}
els.src.addEventListener("input", tally);
els.src.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") go(); });

/* ---------- chunking ---------- */
const bytes = (s) => new TextEncoder().encode(s).length;
function splitLong(s) {
  const out = [];
  while (bytes(s) > CONFIG.chunkHard) {
    let cut = -1;
    const window = s.slice(0, Math.floor(CONFIG.chunkHard * 0.95));
    for (const re of [/[;:]\s/g, /,\s/g, /\s/g]) {
      let m, last = -1; re.lastIndex = 0;
      while ((m = re.exec(window))) if (m.index > CONFIG.chunkTarget * 0.4) last = m.index + 1;
      if (last > 0) { cut = last; break; }
    }
    if (cut < 0) cut = window.length;
    out.push(s.slice(0, cut).trim()); s = s.slice(cut).trim();
  }
  if (s) out.push(s);
  return out;
}
function chunk(text) {
  const chunks = [];
  text.replace(/\r/g, "").split(/\n\s*\n/).forEach((para, p) => {
    const flat = para.replace(/\s*\n\s*/g, " ").trim();
    if (!flat) return;
    const sents = flat.match(/[^.!?]+(?:[.!?]+["'”’)\]]*|$)\s*/g) || [flat];
    let cur = "";
    for (let s of sents) {
      s = s.trim(); if (!s) continue;
      const parts = bytes(s) > CONFIG.chunkHard ? splitLong(s) : [s];
      for (const part of parts) {
        if (cur && bytes(cur + " " + part) > CONFIG.chunkTarget) { chunks.push({ p, text: cur }); cur = part; }
        else cur = cur ? cur + " " + part : part;
      }
    }
    if (cur) chunks.push({ p, text: cur });
  });
  return chunks;
}

/* ---------- output ---------- */
const TAILS = ["tree", "turnip", "crest", "geometry", "pseudo_d20", "creatures", "angel", "gods", "beseech", "ship", "battlefield"];
let pieces = [], chunkList = [];
function render(liveIndex = -1) {
  els.out.innerHTML = "";
  let para = null, lastP = -1;
  pieces.forEach((t, i) => {
    if (t == null) return;
    if (chunkList[i].p !== lastP) { para = document.createElement("p"); para.style.margin = "0 0 .9em"; els.out.appendChild(para); lastP = chunkList[i].p; }
    const span = document.createElement("span");
    span.textContent = (para.childNodes.length ? " " : "") + t;
    if (i === liveIndex) span.className = "live";
    para.appendChild(span);
  });
}
const plain = () => {
  const ps = []; pieces.forEach((t, i) => { if (t == null) return; const p = chunkList[i].p; (ps[p] = ps[p] || []).push(t); });
  return ps.filter(Boolean).map((a) => a.join(" ")).join("\n\n");
};
els.copy.onclick = async () => { try { await navigator.clipboard.writeText(plain()); els.copy.textContent = "❧ copied"; setTimeout(() => (els.copy.textContent = "❧ copy"), 1400); } catch (_) {} };
els.save.onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([plain() + "\n"], { type: "text/plain;charset=utf-8" }));
  a.download = `marchamont-${els.slider.value}${state.scots ? "-scots" : ""}.txt`;
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
};

/* ---------- worker ---------- */
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
function status(t) { els.status.textContent = t || ""; }
worker.onmessage = ({ data }) => {
  if (data.type === "progress") {
    els.hourglass.hidden = false; els.sand.style.transform = `scaleX(${data.frac})`;
    status(`preparing the press · ${Math.round(data.frac * 100)}%`);
  } else if (data.type === "ready") {
    state.ready = true; state.meta = data.meta;
    if (data.meta.years) { ranges = data.meta.years; drawTicks(); setYear(+els.slider.value); }
    els.hourglass.hidden = true; els.go.disabled = false; status("");
  } else if (data.type === "start") {
    els.sand.style.transform = `scaleX(${data.i / chunkList.length})`;
  } else if (data.type === "partial") {
    pieces[data.i] = data.text; render(data.i);
  } else if (data.type === "chunk") {
    pieces[data.i] = data.text; render();
    els.sand.style.transform = `scaleX(${(data.i + 1) / chunkList.length})`;
  } else if (data.type === "done") {
    finish();
  } else if (data.type === "error") {
    if (data.message === "nometa") { status("the press awaits its type"); return; }
    status("the press has jammed"); console.error(data.message); finish();
  }
};
worker.postMessage({ type: "load", base: new URL(CONFIG.model, location.href).href });

function finish() {
  state.working = false;
  els.go.classList.remove("working"); els.goLabel.textContent = "Transmute";
  els.hourglass.hidden = true;
  if (pieces.some((t) => t)) {
    els.tools.hidden = false;
    els.tail.src = `assets/cuts/${TAILS[Math.floor(Math.random() * TAILS.length)]}.png`;
    els.tail.classList.add("shown");
  }
}

function go() {
  if (!state.ready) return;
  if (state.working) { worker.postMessage({ type: "cancel" }); return; }
  const text = els.src.value.trim().slice(0, CONFIG.maxChars);
  if (!text) { els.src.focus(); return; }
  chunkList = chunk(text); pieces = new Array(chunkList.length).fill(null);
  const region = els.region.value === "auto" ? (state.scots ? "scotland" : "england") : els.region.value;
  const settings = {
    year: +els.slider.value, scots: state.scots ? 1 : 0, region: Math.max(0, REGIONS.indexOf(region)),
    hand: state.noHand ? 0 : state.hand, temp: (+els.liberty.value / 100) * 1.1, seed: (state.hand * 31 + +els.slider.value) >>> 0,
  };
  state.working = true;
  els.go.classList.add("working"); els.goLabel.textContent = "Stay";
  els.tools.hidden = true; els.tail.classList.remove("shown"); els.out.innerHTML = "";
  els.hourglass.hidden = false; els.sand.style.transform = "scaleX(0)";
  status("");
  worker.postMessage({ type: "translate", chunks: chunkList, settings });
}
els.go.onclick = go;

drawTicks(); setYear(1650); showHand(); tally();
