"""Plain re-implementation of the T5 (unigram) tokenizer, mirrored line for line by the site's worker.js.
Used by the exporter to write model/t5vocab.json and to check that it matches the reference tokenizer."""
import json
import re
import unicodedata

NORM = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": " - ", "–": "-", "{": "(", "}": ")", "~": "-"})


def vocab_from_hf(tok, max_tokens=256):
    spec = json.loads(tok.backend_tokenizer.to_str())["model"]
    return {"pieces": [p for p, _ in spec["vocab"]], "scores": [s for _, s in spec["vocab"]],
            "unk": spec.get("unk_id", 2), "eos": tok.eos_token_id, "max_tokens": max_tokens}


class T5Tok:
    def __init__(self, v):
        self.map, self.scores = {}, v["scores"]
        self.unk, self.eos, self.max = v["unk"], v["eos"], v["max_tokens"]
        self.max_len, lo = 1, float("inf")
        for i, p in enumerate(v["pieces"]):
            if i < 3 or p.startswith("<extra_id_"):
                continue
            self.map[p] = i
            self.max_len = max(self.max_len, len(p))
            lo = min(lo, self.scores[i])
        self.unk_score = lo - 10.0

    def _word(self, chars):
        n = len(chars)
        best = [float("-inf")] * (n + 1); back = [0] * (n + 1); bid = [0] * (n + 1); best[0] = 0.0
        for i in range(n):
            if best[i] == float("-inf"):
                continue
            piece, single = "", False
            for L in range(1, min(self.max_len, n - i) + 1):
                piece += chars[i + L - 1]
                j = self.map.get(piece)
                if j is None:
                    continue
                if L == 1:
                    single = True
                sc = best[i] + self.scores[j]
                if sc > best[i + L]:
                    best[i + L], back[i + L], bid[i + L] = sc, i, j
            if not single:
                sc = best[i] + self.unk_score
                if sc > best[i + 1]:
                    best[i + 1], back[i + 1], bid[i + 1] = sc, i, self.unk
        seq, j = [], n
        while j > 0:
            seq.append(bid[j]); j = back[j]
        out = []
        for t in reversed(seq):
            if not (t == self.unk and out and out[-1] == self.unk):
                out.append(t)
        return out

    def encode(self, text):
        s = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text.translate(NORM))).strip()
        ids = []
        if s:
            for w in s.split(" "):
                ids += self._word(list("▁" + w))
        return ids[: self.max - 1] + [self.eos]
