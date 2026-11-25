"""Byte-level encoder-decoder transformer with adaLN-zero (FiLM) conditioning in every block.

src bytes (modern or regularised text) -> encoder -> decoder -> target bytes (period spelling), all conditioned on
c = Conditioner(year, region, src_type, scots, doc/z). Pre-norm blocks, RoPE self-attention, SDPA (flash /
mem-efficient kernels on T4 under fp16 autocast), tied input/output embeddings.
"""
import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import Conditioner, CondConfig
from .tokenizer import ByteTokenizer

TOK = ByteTokenizer()


@dataclass
class ModelConfig:
    vocab_size: int = ByteTokenizer.vocab_size
    d_model: int = 512
    n_heads: int = 8
    n_enc: int = 6
    n_dec: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 1024
    cond: CondConfig = field(default_factory=CondConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["cond"] = CondConfig(**d["cond"])
        return cls(**d)


# ---------------------------------------------------------------- rotary position embeddings
class Rope(nn.Module):
    def __init__(self, head_dim, max_len, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        f = torch.outer(torch.arange(max_len).float(), inv)          # (T, hd/2)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    def forward(self, x, offset=0):                                  # x: (B, H, T, hd)
        T = x.shape[-2]
        cos = self.cos[offset:offset + T][None, None]
        sin = self.sin[offset:offset + T][None, None]
        x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
        return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.hd, self.p = n_heads, d // n_heads, dropout
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, ctx=None, key_pad=None, causal=False, rope=None):
        B, T, D = x.shape
        src = x if ctx is None else ctx
        S = src.shape[1]
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)                  # (B,H,T,hd)
        k, v = self.kv(src).view(B, S, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)  # 2 x (B,H,S,hd)
        if rope is not None:
            q, k = rope(q), rope(k)
        mask = None
        if key_pad is not None:                       # key_pad: (B,S) True = padding
            mask = ~key_pad[:, None, None, :]         # SDPA bool mask: True = may attend
        if causal:
            cm = torch.ones(T, S, dtype=torch.bool, device=x.device).tril()
            mask = cm[None, None] if mask is None else mask & cm[None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    """Pre-norm block; each sublayer gets (shift, scale, gate) from c (adaLN-zero: gates start at 0)."""

    def __init__(self, cfg: ModelConfig, cross: bool):
        super().__init__()
        d = cfg.d_model
        self.cross = cross
        self.n_sub = 3 if cross else 2
        self.norms = nn.ModuleList([nn.LayerNorm(d, elementwise_affine=False, eps=1e-6) for _ in range(self.n_sub)])
        self.self_attn = Attention(d, cfg.n_heads, cfg.dropout)
        self.cross_attn = Attention(d, cfg.n_heads, cfg.dropout) if cross else None
        self.ff = nn.Sequential(nn.Linear(d, cfg.d_ff), nn.GELU(approximate="tanh"),
                                nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, d))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(cfg.cond.d_cond, 3 * self.n_sub * d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, c, rope, key_pad=None, causal=False, mem=None, mem_pad=None):
        m = self.mod(c).chunk(3 * self.n_sub, dim=-1)

        def sub(x, k, fn):
            shift, scale, gate = m[3 * k], m[3 * k + 1], m[3 * k + 2]
            return x + gate[:, None] * self.drop(fn(modulate(self.norms[k](x), shift, scale)))

        x = sub(x, 0, lambda h: self.self_attn(h, key_pad=None if causal else key_pad, causal=causal, rope=rope))
        if self.cross:
            x = sub(x, 1, lambda h: self.cross_attn(h, ctx=mem, key_pad=mem_pad))
        x = sub(x, self.n_sub - 1, self.ff)
        return x


class Nedham(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.cond = Conditioner(cfg.cond)
        self.rope = Rope(d // cfg.n_heads, cfg.max_len)
        self.enc = nn.ModuleList([Block(cfg, cross=False) for _ in range(cfg.n_enc)])
        self.dec = nn.ModuleList([Block(cfg, cross=True) for _ in range(cfg.n_dec)])
        self.enc_norm = nn.LayerNorm(d)
        self.dec_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.apply(self._init)
        self.head.weight = self.embed.weight                      # tie
        for b in list(self.enc) + list(self.dec):                 # adaLN-zero
            nn.init.zeros_(b.mod[-1].weight)
            nn.init.zeros_(b.mod[-1].bias)
        with torch.no_grad():                                     # row 0 of the scribe table = "no document"
            self.cond.doc_z.weight[0].zero_()

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    # ---------------------------------------------------------------- core
    def encode(self, src, c):
        pad = src == TOK.PAD
        x = self.embed(src)
        for b in self.enc:
            x = b(x, c, self.rope, key_pad=pad)
        return self.enc_norm(x), pad

    def decode(self, tgt_in, mem, mem_pad, c):
        x = self.embed(tgt_in)
        for b in self.dec:
            x = b(x, c, self.rope, causal=True, mem=mem, mem_pad=mem_pad)
        return self.head(self.dec_norm(x))

    def forward(self, src, tgt_in, **cond):
        c = self.cond(**cond)
        mem, pad = self.encode(src, c)
        return self.decode(tgt_in, mem, pad, c)

    def loss(self, batch, label_smoothing=0.0):
        """batch: src (B,S), tgt (B,T) = BOS ... EOS PAD..., plus conditioning tensors."""
        cond = {k: batch[k] for k in ("year", "region", "src_type", "scots", "doc") if k in batch}
        tgt = batch["tgt"]
        logits = self(batch["src"], tgt[:, :-1], **cond)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1),
                               ignore_index=TOK.PAD, label_smoothing=label_smoothing)

    # ---------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, src, max_new=600, temperature=0.7, top_p=0.95, seed=0,
                 cfg_scale=1.0, guide_on=("year",), **cond):
        """Sample target bytes. Same seed + same z => the 'same scribe' across slider settings.
        cfg_scale > 1 applies classifier-free guidance on the inputs in guide_on (their null value = unconditional).
        No KV cache: fine for sentence-length outputs; add one if generation speed matters."""
        was_training = self.training
        self.eval()
        B, dev = src.shape[0], src.device
        g = torch.Generator(device=dev).manual_seed(seed)
        c = self.cond(**cond, train_drop=False, batch_size=B)
        mem, pad = self.encode(src, c)
        guided = cfg_scale != 1.0
        if guided:
            ucond = {k: (None if k in guide_on else v) for k, v in cond.items()}
            cu = self.cond(**ucond, train_drop=False, batch_size=B)
            mem_u, _ = self.encode(src, cu)
        out = torch.full((B, 1), TOK.BOS, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(min(max_new, self.cfg.max_len - 1)):
            logits = self.decode(out, mem, pad, c)[:, -1].float()
            if guided:
                lu = self.decode(out, mem_u, pad, cu)[:, -1].float()
                logits = lu + cfg_scale * (logits - lu)
            logits[:, TOK.PAD] = -float("inf")
            logits[:, TOK.BOS] = -float("inf")
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, -1)
                sp, si = probs.sort(-1, descending=True)
                cut = sp.cumsum(-1) - sp > top_p
                sp = sp.masked_fill(cut, 0.0)
                sp = sp / sp.sum(-1, keepdim=True)
                nxt = si.gather(-1, torch.multinomial(sp, 1, generator=g))
            nxt = torch.where(done[:, None], torch.full_like(nxt, TOK.PAD), nxt)
            out = torch.cat([out, nxt], 1)
            done |= nxt.squeeze(1) == TOK.EOS
            if done.all():
                break
        self.train(was_training)
        return [TOK.decode(o[1:]) for o in out]


def make_cond(batch_size, device, year=None, region=None, src_type=None, scots=None, doc=None, z=None):
    """Convenience: python values -> conditioning tensors. region/src_type as names or ids."""
    from .constants import REGIONS, SRC_TYPES
    rep = lambda v, dt: torch.full((batch_size,), v, dtype=dt, device=device)
    out = {}
    if year is not None:
        out["year"] = rep(float(year), torch.float32)
    if region is not None:
        out["region"] = rep(REGIONS.index(region) if isinstance(region, str) else region, torch.long)
    if src_type is not None:
        out["src_type"] = rep(SRC_TYPES.index(src_type) if isinstance(src_type, str) else src_type, torch.long)
    if scots is not None:
        out["scots"] = rep(float(scots), torch.float32)
    if doc is not None:
        out["doc"] = rep(int(doc), torch.long)
    if z is not None:
        out["z"] = z.to(device).expand(batch_size, -1) if z.dim() == 1 or z.shape[0] == 1 else z.to(device)
    return out


def save_checkpoint(path, model, optimizer=None, step=0, extra=None):
    torch.save({"cfg": model.cfg.to_dict(), "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer else None, "step": step, "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = Nedham(ModelConfig.from_dict(ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    return model, ck
