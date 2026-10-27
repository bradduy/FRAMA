"""Official per-caption CIDEr-D on THumB 1.0 (Kasai et al., NAACL 2022) for the R1 revision.

THumB gives, for 500 COCO Karpathy-test images, five captions per image (Up-Down, Unified-VLP,
VinVL-base, VinVL-large and one held-out human COCO caption) with human Precision, Recall and a
total human score, plus the four remaining COCO references. We rebuild the five-reference setting
of the paper: references R = the four THumB references + the human caption; candidates = the four
machine captions. For every candidate we compute official CIDEr-D against
  all five references (C5), the four references R_{-f} of fold f (C4[f]) and the single held-out
  reference r_f (C1[f]), f = 0..4 with r_4 = the human caption.
Writes <out>.npz with image names, systems, candidate strings and the three score arrays.

Run on GCP2 (rap_venv, Java 11):
  python thumb_official.py --thumb_dir <dir with mscoco_THumB-1.0.jsonl and mscoco_references.json> --out thumb_cider
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/ZeroShotImageCap/macst_v1/v2_upgrade/revision_r1"))
from official_matrices import PTBTokenizer, cider_vec

SYS = ["Up-Down", "Unified-VLP", "VinVL-base", "VinVL-large"]
ap = argparse.ArgumentParser(); ap.add_argument("--thumb_dir", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = [json.loads(l) for l in open(os.path.join(a.thumb_dir, "mscoco_THumB-1.0.jsonl"))]
refs = {r["seg_id"]: r["refs"] for r in (json.loads(l) for l in open(os.path.join(a.thumb_dir, "mscoco_references.json")))}
by = {}
for r in rows:
    by.setdefault(r["seg_id"], {})[r["SYS"]] = r
segs = sorted(by, key=lambda s: by[s]["Human"]["set_id"])
assert all(len(refs[s]) == 4 and set(by[s]) == set(SYS) | {"Human"} for s in segs)
R5 = [refs[s] + [by[s]["Human"]["hyp"]] for s in segs]
N, K = len(segs), len(SYS)
tok = PTBTokenizer()
ct = tok.tokenize({f"{i}_{j}": [{"caption": by[s][SYS[j]]["hyp"]}] for i, s in enumerate(segs) for j in range(K)})
rt = tok.tokenize({f"{i}_{r}": [{"caption": c}] for i in range(N) for r, c in enumerate(R5[i])})
R = [[rt[f"{i}_{r}"][0] for r in range(5)] for i in range(N)]
H = lambda j: [ct[f"{i}_{j}"][0] for i in range(N)]
C5 = np.stack([cider_vec(R, H(j)) for j in range(K)], 1)
C4 = np.stack([np.stack([cider_vec([[x[r] for r in range(5) if r != f] for x in R], H(j)) for j in range(K)], 1) for f in range(5)])
C1 = np.stack([np.stack([cider_vec([[x[f]] for x in R], H(j)) for j in range(K)], 1) for f in range(5)])
hum = {k: np.array([[by[s][SYS[j]][k] for j in range(K)] for s in segs], float) for k in ("human_score", "P", "R")}
np.savez_compressed(a.out + ".npz", segs=np.array(segs), images=np.array([by[s]["Human"]["image"] for s in segs]),
                    systems=np.array(SYS), cands=np.array([[by[s][x]["hyp"] for x in SYS] for s in segs], dtype=object),
                    C5=C5, C4=C4, C1=C1, H=hum["human_score"], P=hum["P"], Rc=hum["R"],
                    human_caption_score=np.array([by[s]["Human"]["human_score"] for s in segs], float))
print("wrote", a.out + ".npz", C5.shape, C4.shape, C1.shape)
