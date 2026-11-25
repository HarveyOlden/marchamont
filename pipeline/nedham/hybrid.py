"""Pretrained T5 encoder (meaning) + two byte-level layers (exact spelling) feeding the decoder."""
import torch
import torch.nn as nn

from .model import Nedham, Block, TOK

NORM = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": " - ", "–": "-", "{": "(", "}": ")", "~": "-"})


class DualNedham(Nedham):
    def __init__(self, cfg, t5_name="google/flan-t5-small", byte_layers=2, max_src_tokens=256):
        super().__init__(cfg)
        from transformers import AutoTokenizer, T5Config, T5EncoderModel
        self.enc = nn.ModuleList()
        self.enc_norm = nn.Identity()
        self.t5 = T5EncoderModel(T5Config.from_pretrained(t5_name))
        self.t5_tok = AutoTokenizer.from_pretrained(t5_name)
        d_t5 = self.t5.config.d_model
        self.adapt = nn.Sequential(nn.LayerNorm(d_t5), nn.Linear(d_t5, cfg.d_model))
        self.benc = nn.ModuleList([Block(cfg, cross=False) for _ in range(byte_layers)])
        self.benc_norm = nn.LayerNorm(cfg.d_model)
        self.stream = nn.Parameter(torch.zeros(cfg.d_model))
        self.max_src_tokens = max_src_tokens

    def t5_ids(self, texts):
        return self.t5_tok([t.translate(NORM) for t in texts], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_src_tokens)

    def encode(self, src, c):
        b = self.t5_ids([TOK.decode(r) for r in src.tolist()])
        ids, att = b.input_ids.to(src.device), b.attention_mask.to(src.device)
        m1 = self.adapt(self.t5(input_ids=ids, attention_mask=att).last_hidden_state)
        pad = src == TOK.PAD
        x = self.embed(src)
        for blk in self.benc:
            x = blk(x, c, self.rope, key_pad=pad)
        return torch.cat([m1, self.benc_norm(x) + self.stream], 1), torch.cat([att == 0, pad], 1)
