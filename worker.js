import * as ort from "./vendor/ort.wasm.min.mjs";

ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let enc, dec, meta, cancelled = false;

async function fetchBytes(url, version, onProgress) {
  const key = `${url}?v=${version}`;
  const cache = self.caches ? await caches.open("marchamont") : null;
  if (cache) {
    const hit = await cache.match(key);
    if (hit) { const b = new Uint8Array(await hit.arrayBuffer()); onProgress(b.length, b.length); return b; }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  const total = +res.headers.get("content-length") || 0;
  const reader = res.body.getReader();
  const parts = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length; onProgress(got, total);
  }
  const bytes = new Uint8Array(got); let o = 0;
  for (const p of parts) { bytes.set(p, o); o += p.length; }
  if (cache) {
    for (const k of await cache.keys()) if (k.url.includes(url.split("/").pop()) && !k.url.endsWith(`v=${version}`)) await cache.delete(k);
    try { await cache.put(key, new Response(bytes)); } catch (_) {}
  }
  return bytes;
}

async function load(base) {
  const r = await fetch(base + "meta.json", { cache: "no-cache" });
  if (!r.ok) throw new Error("nometa");
  meta = await r.json();
  const files = meta.files || { encoder: ["encoder.onnx"], decoder: ["decoder.onnx"] };
  const all = [...files.encoder, ...files.decoder];
  const sizes = Object.fromEntries(all.map((f) => [f, [0, 0]]));
  const report = () => {
    let got = 0, tot = 0;
    for (const [g, t] of Object.values(sizes)) { got += g; tot += t || 2.4e7; }
    postMessage({ type: "progress", frac: Math.min(1, got / tot) });
  };
  const fetched = await Promise.all(all.map((f) => fetchBytes(base + f, meta.built, (g, t) => { sizes[f] = [g, t]; report(); })));
  const join = (names) => {
    const ps = names.map((n) => fetched[all.indexOf(n)]);
    const out = new Uint8Array(ps.reduce((a, p) => a + p.length, 0)); let o = 0;
    for (const p of ps) { out.set(p, o); o += p.length; }
    return out;
  };
  const eb = join(files.encoder), db = join(files.decoder);
  const opt = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  enc = await ort.InferenceSession.create(eb, opt);
  dec = await ort.InferenceSession.create(db, opt);
  postMessage({ type: "ready", meta });
}

function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function scribe(hand) {
  const z = new Float32Array(meta.z_dim);
  if (!hand) return z;
  const r = rng(hand * 2654435761);
  for (let i = 0; i < z.length; i++) {
    const n = Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
    z[i] = meta.z_mu[i] + meta.z_sd[i] * n;
  }
  return z;
}

function pick(logits, temp, rand) {
  const l = Float32Array.from(logits);
  l[meta.pad] = -Infinity; l[meta.bos] = -Infinity;
  if (temp <= 0.001) { let b = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[b]) b = i; return b; }
  let mx = -Infinity; for (const v of l) if (v > mx) mx = v;
  const p = []; let s = 0;
  for (let i = 0; i < l.length; i++) { const e = Math.exp((l[i] - mx) / temp); if (e > 0) { p.push([e, i]); s += e; } }
  p.sort((a, b) => b[0] - a[0]);
  let acc = 0, kept = [];
  for (const [e, i] of p) { if (acc / s > 0.95) break; kept.push([e, i]); acc += e; }
  let u = rand() * acc;
  for (const [e, i] of kept) { u -= e; if (u <= 0) return i; }
  return kept[kept.length - 1][1];
}

function looping(bytes) {
  const n = bytes.length;
  if (n < 48) return false;
  for (let p = 1; p <= 16; p++) {
    let reps = 1;
    while (reps < 6 && n - (reps + 1) * p >= 0) {
      let same = true;
      for (let k = 0; k < p; k++) if (bytes[n - 1 - k] !== bytes[n - 1 - k - reps * p]) { same = false; break; }
      if (!same) break; reps++;
    }
    if (reps >= (p < 4 ? 6 : 4)) return true;
  }
  return false;
}

const yieldNow = () => new Promise((r) => setTimeout(r, 0));

async function translateChunk(text, s, seed, onPartial) {
  const src = new TextEncoder().encode(text).slice(0, meta.max_src);
  const ids = new BigInt64Array(src.length + 1);
  src.forEach((b, i) => (ids[i] = BigInt(b + meta.offset)));
  ids[src.length] = BigInt(meta.eos);
  const e = await enc.run({
    src: new ort.Tensor("int64", ids, [1, ids.length]),
    year: new ort.Tensor("float32", new Float32Array([s.year]), [1]),
    region: new ort.Tensor("int64", new BigInt64Array([BigInt(s.region)]), [1]),
    scots: new ort.Tensor("float32", new Float32Array([s.scots]), [1]),
    z: new ort.Tensor("float32", scribe(s.hand), [1, meta.z_dim]),
  });
  const L = meta.layers, H = meta.heads, hd = meta.head_dim;
  let pk = new ort.Tensor("float32", new Float32Array(L * H * hd), [L, 1, H, 1, hd]);
  let pv = new ort.Tensor("float32", new Float32Array(L * H * hd), [L, 1, H, 1, hd]);
  const rand = rng(seed);
  const maxNew = Math.min(meta.max_pos - 2, Math.ceil(src.length * 1.7) + 24);
  const decoder = new TextDecoder("utf-8");
  const out = []; let shown = "", tok = meta.bos;
  for (let t = 0; t < maxNew; t++) {
    if (cancelled) break;
    const bias = new Float32Array(t + 2); bias[0] = meta.neg;
    const r = await dec.run({
      token: new ort.Tensor("int64", new BigInt64Array([BigInt(tok)]), [1, 1]),
      pos: new ort.Tensor("int64", new BigInt64Array([BigInt(t)]), [1]),
      bias: new ort.Tensor("float32", bias, [1, 1, 1, t + 2]),
      cross_k: e.cross_k, cross_v: e.cross_v, dec_mod: e.dec_mod, past_k: pk, past_v: pv,
    });
    pk = r.present_k; pv = r.present_v;
    tok = pick(r.logits.data, s.temp, rand);
    if (tok === meta.eos) break;
    out.push(tok - meta.offset);
    shown += decoder.decode(new Uint8Array([tok - meta.offset]), { stream: true });
    if (t % 3 === 0) onPartial(shown);
    if (looping(out)) break;
    if (t % 12 === 0) await yieldNow();
  }
  return shown + decoder.decode();
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") await load(data.base);
    else if (data.type === "cancel") cancelled = true;
    else if (data.type === "translate") {
      cancelled = false;
      const { chunks, settings } = data;
      for (let i = 0; i < chunks.length && !cancelled; i++) {
        postMessage({ type: "start", i });
        const text = await translateChunk(chunks[i].text, settings, settings.seed + i * 7919, (p) => postMessage({ type: "partial", i, text: p }));
        postMessage({ type: "chunk", i, text });
      }
      postMessage({ type: "done", cancelled });
    }
  } catch (err) {
    postMessage({ type: "error", message: String(err && err.message || err) });
  }
};
