"""Conditioning: year, region, source type, Scots-ness scalar, per-document scribe latent z -> one vector c.

c drives adaLN (FiLM) in every transformer block. Every input has a learned "null" value, used when the
input is unknown and, during training, at random (condition dropout) so the model also learns the
unconditional distribution -> enables classifier-free guidance and missing-metadata inference.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .constants import REGIONS, SRC_TYPES


@dataclass
class CondConfig:
    d_cond: int = 256
    year_lo: float = 1100.0          # year normalisation range (t = 0..1)
    year_hi: float = 2050.0
    n_fourier: int = 6               # LOW frequencies only: smooth in year -> smooth slider, sane extrapolation
    n_regions: int = len(REGIONS)
    n_src_types: int = len(SRC_TYPES)
    n_docs: int = 1                  # size of the per-document z table (set from the data)
    z_dim: int = 16
    p_drop: dict = field(default_factory=lambda: {"year": 0.1, "region": 0.3, "src": 0.1, "scots": 0.3, "z": 0.2})


class Conditioner(nn.Module):
    def __init__(self, cfg: CondConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_cond
        self.register_buffer("freqs", math.pi * (2.0 ** torch.arange(cfg.n_fourier)), persistent=False)
        self.year_mlp = nn.Sequential(nn.Linear(2 * cfg.n_fourier + 1, d), nn.SiLU(), nn.Linear(d, d))
        self.null_year = nn.Parameter(torch.zeros(d))
        self.region = nn.Embedding(cfg.n_regions, d)       # index 0 = unknown / dropped
        self.src = nn.Embedding(cfg.n_src_types, d)        # index 0 = unknown / dropped
        self.scots = nn.Linear(1, d)
        self.null_scots = nn.Parameter(torch.zeros(d))
        # per-document scribe latent (auto-decoder): row 0 = "no document" (generation without a borrowed scribe)
        self.doc_z = nn.Embedding(cfg.n_docs + 1, cfg.z_dim)
        nn.init.normal_(self.doc_z.weight, std=0.01)
        with torch.no_grad():
            self.doc_z.weight[0].zero_()
        self.z_proj = nn.Linear(cfg.z_dim, d)
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(d, d))

    def year_features(self, year):
        t = ((year.float() - self.cfg.year_lo) / (self.cfg.year_hi - self.cfg.year_lo)).unsqueeze(-1)
        ang = t * self.freqs
        return torch.cat([t, torch.sin(ang), torch.cos(ang)], -1)

    def forward(self, year=None, region=None, src_type=None, scots=None, doc=None, z=None, train_drop=None, batch_size=None):
        """All args are (B,) tensors (or None). year: float years; NaN = unknown. scots: float in [0,1]; NaN = unknown.
        region/src_type/doc: long ids (0 = unknown). z: (B, z_dim) explicit scribe latent, overrides doc."""
        ref = next((t for t in (year, region, src_type, scots, doc, z) if t is not None), None)
        if ref is None:
            if batch_size is None:
                raise ValueError("pass at least one conditioning tensor or batch_size")
            B, dev = batch_size, self.null_year.device
        else:
            B, dev = ref.shape[0], ref.device
        train_drop = self.training if train_drop is None else train_drop
        p = self.cfg.p_drop

        def keep(name):  # per-sample Bernoulli mask for condition dropout
            if not train_drop:
                return torch.ones(B, dtype=torch.bool, device=dev)
            return torch.rand(B, device=dev) >= p[name]

        c = torch.zeros(B, self.cfg.d_cond, device=dev, dtype=self.null_year.dtype)
        if year is not None:
            ok = ~torch.isnan(year) & keep("year")
            y = torch.where(ok, year, torch.full_like(year, (self.cfg.year_lo + self.cfg.year_hi) / 2))
            c = c + torch.where(ok[:, None], self.year_mlp(self.year_features(y)).to(c.dtype), self.null_year.to(c.dtype))
        else:
            c = c + self.null_year
        if region is not None:
            c = c + self.region(torch.where(keep("region"), region, torch.zeros_like(region)))
        else:
            c = c + self.region.weight[0]
        if src_type is not None:
            c = c + self.src(torch.where(keep("src"), src_type, torch.zeros_like(src_type)))
        else:
            c = c + self.src.weight[0]
        if scots is not None:
            ok = ~torch.isnan(scots) & keep("scots")
            s = torch.nan_to_num(scots, 0.0).unsqueeze(-1)
            c = c + torch.where(ok[:, None], self.scots(s).to(c.dtype), self.null_scots.to(c.dtype))
        else:
            c = c + self.null_scots
        if z is None:
            d = doc if doc is not None else torch.zeros(B, dtype=torch.long, device=dev)
            d = torch.where(keep("z"), d, torch.zeros_like(d))
            z = self.doc_z(d)
        c = c + self.z_proj(z.to(self.z_proj.weight.dtype)).to(c.dtype)
        return self.out(c)

    def z_prior(self, n, generator=None):
        """Sample scribe latents from a Gaussian fitted to the learned document embeddings (excluding row 0)."""
        w = self.doc_z.weight[1:].detach()
        if len(w) < 2:
            return torch.zeros(n, self.cfg.z_dim, device=w.device)
        mu, sd = w.mean(0), w.std(0)
        eps = torch.randn(n, w.shape[1], generator=generator, device="cpu").to(w.device)
        return mu + sd * eps
