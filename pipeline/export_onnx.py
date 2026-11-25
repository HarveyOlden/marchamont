"""Export a Marchamont checkpoint to the two ONNX graphs the site runs in the browser.

    python pipeline/export_onnx.py --ckpt pipeline/weights/final.pt --out model

Writes the encoder/decoder graphs (int8 by default) to model/ in parts under 24 MB (so they can be uploaded
through GitHub's web interface) plus model/meta.json, then checks the exported graphs against the PyTorch model.
"""
import argparse, difflib, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from nedham.model import Nedham, ModelConfig            # noqa: E402
from nedham.constants import REGIONS, SRC_TYPES         # noqa: E402
from nedham.tokenizer import ByteTokenizer              # noqa: E402

TOK = ByteTokenizer()
NEG = -1e9


def lin3(layer, x2d):
    return layer(x2d.unsqueeze(1)).squeeze(1)


def rot(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


def mod(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


def byte_layers(m, blocks, src, c, H, hd):
    S = src.shape[1]
    cos = m.rope.cos[:S][None, None]
    sin = m.rope.sin[:S][None, None]
    x = m.embed(src)
    for b in blocks:
        md = lin3(b.mod[1], F.silu(c)).chunk(6, dim=-1)
        h = mod(b.norms[0](x), md[0], md[1])
        q = b.self_attn.q(h).view(1, -1, H, hd).transpose(1, 2)
        kv = b.self_attn.kv(h).view(1, -1, 2, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = rot(q, cos, sin), rot(kv[0], cos, sin), kv[1]
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), dim=-1) @ v
        x = x + md[2][:, None] * b.self_attn.o(a.transpose(1, 2).reshape(1, -1, H * hd))
        h = mod(b.norms[1](x), md[3], md[4])
        x = x + md[5][:, None] * b.ff(h)
    return x


def cross_out(m, mem, c, H, hd):
    ks, vs, mods = [], [], []
    for b in m.dec:
        kv = b.cross_attn.kv(mem).view(1, -1, 2, H, hd).permute(2, 0, 3, 1, 4)
        ks.append(kv[0]); vs.append(kv[1])
        mods.append(torch.stack(lin3(b.mod[1], F.silu(c)).chunk(9, dim=-1), 0))
    return torch.stack(ks), torch.stack(vs), torch.stack(mods)


class Encoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        cfg = m.cfg
        self.H, self.hd = cfg.n_heads, cfg.d_model // cfg.n_heads

    def forward(self, src, year, region, scots, z):
        m, H, hd = self.m, self.H, self.hd
        c = m.cond(year=year, region=region, src_type=torch.ones_like(region), scots=scots, z=z, train_drop=False)
        return cross_out(m, m.enc_norm(byte_layers(m, m.enc, src, c, H, hd)), c, H, hd)


class DualEncoder(Encoder):
    """T5 states (meaning) followed by byte-layer states (spelling), as one memory for the decoder."""

    def forward(self, t5_ids, src, year, region, scots, z):
        m, H, hd = self.m, self.H, self.hd
        c = m.cond(year=year, region=region, src_type=torch.ones_like(region), scots=scots, z=z, train_drop=False)
        m1 = m.adapt(m.t5(input_ids=t5_ids, attention_mask=torch.ones_like(t5_ids)).last_hidden_state)
        m2 = m.benc_norm(byte_layers(m, m.benc, src, c, H, hd)) + m.stream
        return cross_out(m, torch.cat([m1, m2], 1), c, H, hd)


class DecoderStep(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        cfg = m.cfg
        self.H, self.hd = cfg.n_heads, cfg.d_model // cfg.n_heads

    def forward(self, token, pos, bias, cross_k, cross_v, dec_mod, past_k, past_v):
        m, H, hd = self.m, self.H, self.hd
        cos = m.rope.cos.index_select(0, pos)[None, None]
        sin = m.rope.sin.index_select(0, pos)[None, None]
        x = m.embed(token)
        pk, pv = [], []
        for l, b in enumerate(m.dec):
            md = dec_mod[l]
            h = mod(b.norms[0](x), md[0], md[1])
            q = rot(b.self_attn.q(h).view(1, 1, H, hd).transpose(1, 2), cos, sin)
            kv = b.self_attn.kv(h).view(1, 1, 2, H, hd).permute(2, 0, 3, 1, 4)
            K = torch.cat([past_k[l], rot(kv[0], cos, sin)], dim=2)
            V = torch.cat([past_v[l], kv[1]], dim=2)
            a = torch.softmax(q @ K.transpose(-1, -2) / math.sqrt(hd) + bias, dim=-1) @ V
            x = x + md[2][:, None] * b.self_attn.o(a.transpose(1, 2).reshape(1, 1, H * hd))
            h = mod(b.norms[1](x), md[3], md[4])
            q = b.cross_attn.q(h).view(1, 1, H, hd).transpose(1, 2)
            a = torch.softmax(q @ cross_k[l].transpose(-1, -2) / math.sqrt(hd), dim=-1) @ cross_v[l]
            x = x + md[5][:, None] * b.cross_attn.o(a.transpose(1, 2).reshape(1, 1, H * hd))
            h = mod(b.norms[2](x), md[6], md[7])
            x = x + md[8][:, None] * b.ff(h)
            pk.append(K); pv.append(V)
        logits = m.head(m.dec_norm(x))[:, 0]
        return logits, torch.stack(pk), torch.stack(pv)


def export(fn, args, path, names_in, names_out, axes):
    kw = dict(input_names=names_in, output_names=names_out, dynamic_axes=axes, opset_version=17, do_constant_folding=True)
    try:
        torch.onnx.export(fn, args, str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(fn, args, str(path), **kw)


class OrtModel:
    def __init__(self, enc_path, dec_path, cfg, t5tok=None):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = max(1, os.cpu_count() // 2)
        self.enc = ort.InferenceSession(str(enc_path), so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(dec_path), so, providers=["CPUExecutionProvider"])
        self.L, self.H, self.hd = cfg.n_dec, cfg.n_heads, cfg.d_model // cfg.n_heads
        self.t5tok = t5tok

    def greedy(self, text, year, region, scots, z, max_new=400):
        src = np.array([TOK.encode(text)], dtype=np.int64)
        feed = {"src": src, "year": np.array([year], np.float32), "region": np.array([REGIONS.index(region)], np.int64),
                "scots": np.array([scots], np.float32), "z": z[None].astype(np.float32)}
        if self.t5tok is not None:
            feed["t5_ids"] = np.array([self.t5tok.encode(text)], dtype=np.int64)
        ck, cv, dm = self.enc.run(None, feed)
        pk = np.zeros((self.L, 1, self.H, 1, self.hd), np.float32); pv = pk.copy()
        tok, out = TOK.BOS, []
        for t in range(max_new):
            bias = np.zeros((1, 1, 1, t + 2), np.float32); bias[..., 0] = NEG
            logits, pk, pv = self.dec.run(None, {"token": np.array([[tok]], np.int64), "pos": np.array([t], np.int64),
                                                 "bias": bias, "cross_k": ck, "cross_v": cv, "dec_mod": dm,
                                                 "past_k": pk, "past_v": pv})
            lg = logits[0].copy(); lg[TOK.PAD] = lg[TOK.BOS] = -np.inf
            tok = int(lg.argmax())
            if tok == TOK.EOS:
                break
            out.append(tok)
        return TOK.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path(__file__).parent / "weights" / "final.pt"))
    ap.add_argument("--out", default="model")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--english", default="1380,1790")
    ap.add_argument("--scots", default="1500,1905")
    ap.add_argument("--part-mb", type=float, default=24.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); tmp = out / "_fp32"; tmp.mkdir(exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["cfg"])
    hyb, t5tok = ck.get("hybrid"), None
    if hyb:
        from nedham.hybrid import DualNedham
        from nedham.t5tok import T5Tok, vocab_from_hf
        name = hyb["t5"] if "/" in hyb["t5"] else "google/" + hyb["t5"]
        m = DualNedham(cfg, name, hyb.get("byte_layers", 2))
        missing, unexpected = m.load_state_dict(ck["model"], strict=False)
        missing = [k for k in missing if k != "t5.encoder.embed_tokens.weight"]
        assert not missing and not unexpected, (missing[:6], unexpected[:6])
        vocab = vocab_from_hf(m.t5_tok, m.max_src_tokens); t5tok = T5Tok(vocab)
        print(f"dual model: T5 ({name}) + {len(m.benc)} byte layers, {m.n_params() / 1e6:.1f}M params")
    else:
        m = Nedham(cfg); m.load_state_dict(ck["model"])
    m.eval()
    L, H, hd, S, P = cfg.n_dec, cfg.n_heads, cfg.d_model // cfg.n_heads, 12, 3
    z0 = torch.zeros(1, cfg.cond.z_dim)

    enc, dec = (DualEncoder(m) if hyb else Encoder(m)).eval(), DecoderStep(m).eval()
    with torch.no_grad():
        src = torch.randint(4, 200, (1, S))
        args_e = (src, torch.tensor([1600.0]), torch.tensor([1]), torch.tensor([0.0]), z0)
        names_e, axes_e = ["src", "year", "region", "scots", "z"], {"src": {1: "S"}, "cross_k": {3: "M"}, "cross_v": {3: "M"}}
        if hyb:
            args_e = (torch.tensor([t5tok.encode("A short test sentence, for tracing.")]),) + args_e
            names_e, axes_e = ["t5_ids"] + names_e, {"t5_ids": {1: "St"}, **axes_e}
        ck_, cv_, dm_ = enc(*args_e)
        args_d = (torch.tensor([[TOK.BOS]]), torch.tensor([P - 1]), torch.zeros(1, 1, 1, P + 1),
                  ck_, cv_, dm_, torch.zeros(L, 1, H, P, hd), torch.zeros(L, 1, H, P, hd))
        export(enc, args_e, tmp / "encoder.onnx", names_e, ["cross_k", "cross_v", "dec_mod"], axes_e)
        export(dec, args_d, tmp / "decoder.onnx", ["token", "pos", "bias", "cross_k", "cross_v", "dec_mod", "past_k", "past_v"],
               ["logits", "present_k", "present_v"],
               {"bias": {3: "P1"}, "cross_k": {3: "S"}, "cross_v": {3: "S"}, "past_k": {3: "P"}, "past_v": {3: "P"},
                "present_k": {3: "P1"}, "present_v": {3: "P1"}})
    print("exported fp32 graphs")

    if a.fp32:
        for f in ("encoder.onnx", "decoder.onnx"): os.replace(tmp / f, out / f)
    else:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        for f in ("encoder.onnx", "decoder.onnx"):
            quantize_dynamic(str(tmp / f), str(out / f), weight_type=QuantType.QInt8, per_channel=True,
                             op_types_to_quantize=["MatMul", "Gemm", "Gather"]) #smaller weights yay
        print("quantised to int8")

    w = m.cond.doc_z.weight[1:].detach()
    en, sc = [int(v) for v in a.english.split(",")], [int(v) for v in a.scots.split(",")]
    meta = {"name": "marchamont", "layers": L, "heads": H, "head_dim": hd, "vocab": cfg.vocab_size,
            "pad": TOK.PAD, "bos": TOK.BOS, "eos": TOK.EOS, "offset": TOK.OFFSET, "neg": NEG,
            "regions": REGIONS, "z_dim": cfg.cond.z_dim,
            "z_mu": [round(float(v), 6) for v in w.mean(0)], "z_sd": [round(float(v), 6) for v in w.std(0)],
            "years": {"english": en, "scots": sc}, "max_src": 600, "max_pos": cfg.max_len,
            "precision": "fp32" if a.fp32 else "int8", "built": time.strftime("%Y-%m-%d %H:%M")}
    if hyb:
        json.dump(vocab, open(out / "t5vocab.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
        meta["t5"] = {"vocab": "t5vocab.json", "max_tokens": vocab["max_tokens"]}
        probe = ["I don't think we should trust him; he has lied to us before.", "i dont think bananas deserve equal rights tbh",
                 "Honestly, the weather’s been awful — we’ve stayed in all week!!", "Sarah texted me at 3pm about the £20 she owes Kevin.",
                 "  Spaces   everywhere\tand\nnewlines ", "Café naïve résumé – “quoted” {braces} ~tilde~", "Ðe weird þorn ȝogh ſ chars",
                 "URL https://example.com/a_b?c=1 & email a@b.co", "Mr. O'Brien-Smith's 1,234.5 kg", "日本語 and emoji 🙂 mixed in"]
        bad = [t for t in probe if t5tok.encode(t) != m.t5_ids([t]).input_ids[0].tolist()]
        print(f"tokenizer port: {len(probe) - len(bad)}/{len(probe)} identical to the reference" + "".join(f"\n   differs: {t!r}" for t in bad))
    json.dump(meta, open(out / "meta.json", "w"), indent=1)

    tests = [("I don't think we should trust him; he has lied to us before.", 1600, "england", 0.0),
             ("Come in and sit down, you look frozen.", 1450, "england", 0.0),
             ("The king has written a letter to his son.", 1750, "england", 0.0),
             ("She said she would not go to church on Sunday.", 1560, "scotland", 1.0)]
    ref = []
    for text, year, region, scots in tests:
        src = torch.tensor([TOK.encode(text)])
        ref.append(m.generate(src, max_new=400, temperature=0, year=torch.tensor([float(year)]),
                              region=torch.tensor([REGIONS.index(region)]), src_type=torch.tensor([SRC_TYPES.index("modern")]),
                              scots=torch.tensor([scots]), z=z0)[0])
    f32 = OrtModel(tmp / "encoder.onnx", tmp / "decoder.onnx", cfg, t5tok) if not a.fp32 else OrtModel(out / "encoder.onnx", out / "decoder.onnx", cfg, t5tok)
    fin = OrtModel(out / "encoder.onnx", out / "decoder.onnx", cfg, t5tok)
    print("\ncheck: pytorch vs onnx (greedy)")
    ok = 0
    for (text, year, region, scots), r in zip(tests, ref):
        o32 = f32.greedy(text, year, region, scots, np.zeros(cfg.cond.z_dim))
        t0 = time.time(); oq = fin.greedy(text, year, region, scots, np.zeros(cfg.cond.z_dim)); dt = time.time() - t0
        same = o32 == r; ok += same
        sim = difflib.SequenceMatcher(None, oq, r).ratio()
        print(f"  {year} {region:9s} torch==onnx-fp32: {same} | final vs torch similarity {sim:.2f} | {len(oq) / max(dt, 1e-6):.0f} bytes/s")
        print(f"     torch: {r}\n     final: {oq}")
    parts = {}
    for f in ("encoder.onnx", "decoder.onnx"):
        data = (out / f).read_bytes(); step = int(a.part_mb * 1e6); names = []
        for old in out.glob(f + ".*"): old.unlink()
        for i in range(0, len(data), step):
            name = f"{f}.{i // step}"; (out / name).write_bytes(data[i:i + step]); names.append(name)
        (out / f).unlink(); parts[f.split(".")[0]] = names
        print(f"{f}: {len(data) / 1e6:.1f} MB in {len(names)} part(s)")
    meta["files"] = parts
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    if not a.fp32:
        for f in tmp.iterdir(): f.unlink()
        tmp.rmdir()
    print(f"\n{'✅' if ok == len(tests) else '⚠'} {ok}/{len(tests)} exact fp32 matches. Files in {out}/")


if __name__ == "__main__":
    main()
