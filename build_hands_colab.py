# Colab cell: builds model/hands.json + model/hands_pool.bin (scribe vectors only) from final.pt.
# Paste into one cell, run, then put both downloaded files into the repo's model/ folder.
import json, numpy as np, pandas as pd, torch
from google.colab import drive, files
drive.mount("/content/drive")
B = "/content/drive/MyDrive/Nedham"
REGIONS = ["<unk>", "england", "scotland", "ireland", "wales", "colonies", "continent"]
ck = torch.load(f"{B}/runs/natural_v2/final.pt", map_location="cpu", weights_only=False)
Z = ck["model"]["cond.doc_z.weight"].float().numpy()
idx = json.load(open(f"{B}/runs/natural_v2/doc_index.json"))
pairs = pd.read_parquet(f"{B}/processed/train/pairs.parquet", columns=["doc_key", "year", "region", "split"])

FAMOUS = [  # label, doc keys or key prefix, year, region, scots
    ("Wycliffite Bible", "bible:wycliffite:", 1395, "england", 0), ("Caxton's Chaucer", "A18548", 1477, "england", 0),
    ("Malory, Morte Darthur", "A21703", 1485, "england", 0), ("Nisbet's Scots Testament", "bible:nisbet_c1520:", 1520, "scotland", 1),
    ("Tyndale's Testament", "bible:tyndale_nt_1534:", 1534, "continent", 0), ("Coverdale Bible", "bible:coverdale_1535:", 1535, "continent", 0),
    ("Great Bible", "bible:great_bible_1540:", 1540, "england", 0), ("Book of Common Prayer", "A72693", 1549, "england", 0),
    ("More, Utopia", "A07706", 1551, "england", 0), ("Douglas, Eneados", "A14477", 1553, "scotland", 1),
    ("Geneva Bible", "bible:geneva_1561:", 1561, "continent", 0), ("Bishops' Bible", "bible:bishops_1568:", 1568, "england", 0),
    ("Foxe, Actes and Monuments", "A67922", 1583, "england", 0), ("Sidney, Arcadia", "A12229", 1590, "england", 0),
    ("Marlowe, Tamburlaine", "A07004", 1590, "england", 0), ("Spenser, Faerie Queene", "A12778", 1596, "england", 0),
    ("Hakluyt, Principal Navigations", "A02495", 1599, "england", 0), ("Lindsay, Ane Satyre", "A72573", 1602, "scotland", 1),
    ("Douay Bible", "bible:douay_ot_1609:", 1609, "continent", 0), ("Jonson, Workes", "A04632", 1616, "england", 0),
    ("Barbour, The Bruce", "A03659", 1620, "scotland", 1), ("Burton, Anatomy of Melancholy", "A17310", 1621, "england", 0),
    ("Shakespeare, First Folio", "A11954", 1623, "england", 0), ("Bacon, Essayes", "B11307", 1625, "england", 0),
    ("Donne, Poems", "A69225", 1633, "england", 0), ("Browne, Religio Medici", "A29868", 1642, "england", 0),
    ("Milton, Areopagitica", "A50883", 1644, "england", 0), ("Hobbes, Leviathan", "A43998", 1651, "england", 0),
    ("Walton, Compleat Angler", "A67462", 1653, "england", 0), ("Nedham, Excellencie of a Free-State", "A89878", 1656, "england", 0),
    ("Milton, Paradise Lost", "A50919", 1667, "england", 0), ("Bunyan, Pilgrim's Progress", "A30170", 1678, "england", 0),
    ("Behn, Oroonoko", "A27305", 1688, "england", 0), ("Defoe, Robinson Crusoe", "K061280.000", 1719, "england", 0),
]
famous = []
for name, key, year, region, scots in FAMOUS:
    keys = [k for k in idx if k.startswith(key)] if key.endswith(":") else [k for k in (key, f"ep:{key}", f"tcp:{key}") if k in idx]
    rows = [idx[k] for k in keys if np.abs(Z[idx[k]]).sum() > 0]
    if not rows: print("  not found:", name); continue
    famous.append({"n": name, "y": year, "r": REGIONS.index(region), "s": scots, "z": [round(float(v), 5) for v in Z[rows].mean(0)]})
print(f"{len(famous)}/{len(FAMOUS)} named hands")

# chance pool: every trained scribe (>= 10 training pairs), stored compactly as year, region, 16 int8 numbers
d = pairs[pairs.split == "train"].groupby("doc_key").agg(n=("year", "size"), year=("year", "first"), region=("region", "first"))
d = d[(d.n >= 10) & d.index.isin(idx.keys())].reset_index()
P = Z[[idx[k] for k in d.doc_key]]
lo, hi = P.min(0), P.max(0); scale = np.maximum(hi - lo, 1e-6) / 255
q = (np.clip(np.round((P - lo) / scale), 0, 255) - 128).astype(np.int8)
rec = np.zeros((len(d), 3 + P.shape[1]), np.uint8)
yrs = d.year.astype(int).clip(0, 65535).to_numpy()
rec[:, 0], rec[:, 1] = yrs & 255, yrs >> 8
rec[:, 2] = [REGIONS.index(r) if r in REGIONS else 0 for r in d.region]
rec[:, 3:] = q.view(np.uint8)
open("/content/hands_pool.bin", "wb").write(rec.tobytes())
pool = {"file": "hands_pool.bin", "dim": int(P.shape[1]), "lo": [float(v) for v in lo], "scale": [float(v) for v in scale]}
print(f"{len(d):,} scribes in the chance pool, {yrs.min()}–{yrs.max()} ({rec.nbytes / 1e6:.1f} MB)")
json.dump({"famous": famous, "pool": pool}, open("/content/hands.json", "w"), separators=(",", ":"))
files.download("/content/hands.json"); files.download("/content/hands_pool.bin")
