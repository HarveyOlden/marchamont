"""Byte-level tokenizer: UTF-8 bytes + 4 specials. Handles ſ, ȝ, þ, combining macrons etc. with no vocab work."""
class ByteTokenizer:
    PAD, BOS, EOS, SEP = 0, 1, 2, 3
    OFFSET = 4
    vocab_size = 256 + OFFSET

    def encode(self, text, add_bos=False, add_eos=True, max_len=None):
        ids = [b + self.OFFSET for b in text.encode("utf-8")]
        if max_len is not None:
            ids = ids[: max_len - int(add_bos) - int(add_eos)]
        return ([self.BOS] if add_bos else []) + ids + ([self.EOS] if add_eos else [])

    def decode(self, ids):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        out = []
        for i in ids:
            if i == self.EOS:
                break
            if i >= self.OFFSET:
                out.append(i - self.OFFSET)
        return bytes(out).decode("utf-8", errors="replace")

    def pad_batch(self, seqs, pad_to=None):
        import torch
        n = pad_to or max(len(s) for s in seqs)
        t = torch.full((len(seqs), n), self.PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            t[i, : len(s)] = torch.tensor(s[:n])
        return t
