import { CONFIG } from "./config.js";

const $ = (id) => document.getElementById(id);
const els = {
  src: $("src"), tally: $("tally"), year: $("year"), roman: $("roman"), slider: $("slider"), ticks: $("ticks"),
  en: $("t-en"), sc: $("t-sc"), go: $("go"), goLabel: $("go-label"), status: $("status"), hourglass: $("hourglass"),
  sand: $("sand"), region: $("region"), hand: $("hand"), roll: $("roll"), liberty: $("liberty"),
  out: $("out"), tail: $("tail"), tools: $("tools"), copy: $("copy"), save: $("save"), band: $("era-band"),
};
$("source").href = CONFIG.source;
$("weights").href = CONFIG.weights;

const state = { scots: false, ready: false, working: false, meta: null, roll: Math.floor(Math.random() * 1e6), famous: [], rerun: false };
let ranges = { english: [1380, 1790], scots: [1500, 1905] };
const REGIONS = ["<unk>", "england", "scotland", "ireland", "wales", "colonies", "continent"];

/* ---------- year dial ---------- */
function roman(n) {
  const t = [[1000, "M"], [900, "CM"], [500, "D"], [400, "CD"], [100, "C"], [90, "XC"], [50, "L"], [40, "XL"], [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]];
  let s = ""; for (const [v, r] of t) while (n >= v) { s += r; n -= v; } return s;
}
const BANDS = [[1480, "prayer_header"], [1600, "king_header"], [1700, "eagle_header"], [9999, "devil_header"]];
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
els.slider.addEventListener("change", () => {
  const v = els.hand.value;
  if (v.startsWith("f") && Math.abs(state.famous[+v.slice(1)].y - +els.slider.value) > 40) els.hand.value = "random";
  changed();
});
let rerunTimer = null;
function changed() {
  if (!state.ready || !pieces.some((t) => t)) return;
  clearTimeout(rerunTimer);
  rerunTimer = setTimeout(() => {
    if (state.working) { state.rerun = true; worker.postMessage({ type: "cancel" }); }
    else go();
  }, 350);
}

/* ---------- tongue ---------- */
function setTongue(scots) {
  state.scots = scots;
  els.en.classList.toggle("on", !scots); els.sc.classList.toggle("on", scots);
  els.en.setAttribute("aria-checked", !scots); els.sc.setAttribute("aria-checked", scots);
  drawTicks(); setYear(+els.slider.value);
}
els.en.onclick = () => { if (state.scots) { setTongue(false); changed(); } };
els.sc.onclick = () => { if (!state.scots) { setTongue(true); changed(); } };
els.region.addEventListener("change", () => changed());
els.liberty.addEventListener("change", () => changed());

/* ---------- hand ---------- */
function fillHands(famous, pool) {
  state.famous = famous;
  els.hand.innerHTML = '<option value="none">∅</option>' + (pool ? '<option value="random">⚄ of the period</option>' : "");
  if (famous.length) {
    const g = document.createElement("optgroup"); g.label = "—";
    famous.forEach((f, i) => { const o = document.createElement("option"); o.value = `f${i}`; o.textContent = `${f.n}, ${f.y}`; g.appendChild(o); });
    els.hand.appendChild(g);
  }
  els.hand.value = pool ? "random" : "none"; els.roll.disabled = !pool;
}
function handSpec() {
  const v = els.hand.value;
  if (v.startsWith("f")) return { mode: "famous", idx: +v.slice(1) };
  return v === "random" ? { mode: "random", roll: state.roll } : { mode: "none" };
}
els.roll.onclick = () => { state.roll = Math.floor(Math.random() * 1e6); if ([...els.hand.options].some((o) => o.value === "random")) els.hand.value = "random"; changed(); };
els.hand.onchange = () => {
  const v = els.hand.value;
  if (v.startsWith("f")) {
    const f = state.famous[+v.slice(1)];
    if (f.s && !state.scots) setTongue(true); else if (!f.s && state.scots) setTongue(false);
    setYear(f.y);
  }
  changed();
};

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
    fillHands(data.famous || [], data.pool || 0);
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
    if (state.rerun) { state.rerun = false; go(); }
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
  const text = els.src.value.trim().slice(0, CONFIG.maxChars)
    .replace(/(^|[\s"“'‘(\-])i(?=['’](?:m|ll|d|ve)\b|[\s,;:!?)”"]|\.(?!\w)|$)/g, "$1I");
  if (!text) { els.src.focus(); return; }
  chunkList = chunk(text); pieces = new Array(chunkList.length).fill(null);
  const region = els.region.value === "auto" ? (state.scots ? "scotland" : "england") : els.region.value;
  const settings = {
    year: +els.slider.value, scots: state.scots ? 1 : 0, region: Math.max(0, REGIONS.indexOf(region)),
    hand: handSpec(), temp: (+els.liberty.value / 100) * 1.1, seed: (state.roll * 31 + +els.slider.value) >>> 0,
  };
  state.working = true;
  els.go.classList.add("working"); els.goLabel.textContent = "Stay";
  els.tools.hidden = true; els.tail.classList.remove("shown"); els.out.innerHTML = "";
  els.hourglass.hidden = false; els.sand.style.transform = "scaleX(0)";
  status("");
  worker.postMessage({ type: "translate", chunks: chunkList, settings });
}
els.go.onclick = go;

drawTicks(); setYear(1650); tally();
