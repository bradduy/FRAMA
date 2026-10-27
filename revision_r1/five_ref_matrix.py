"""Official CIDEr-D of every candidate against the first five references only (the reference
set of the cross-reference protocol), so that the reference-selected pick on the held-out scale
is selected on exactly the five references the protocol uses (NoCaps has ten). Writes
matrices/<out>_five.npz with names and M_five [N, K]; images with fewer than five references
keep NaN rows."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official_matrices import PTBTokenizer, cider_vec
ap = argparse.ArgumentParser(); ap.add_argument("--beam_dump", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
if os.path.exists(a.out + "_five.npz"):
    sys.exit(0)
beams = json.load(open(a.beam_dump)); N = len(beams); K = min(len(b["beam"]) for b in beams)
ok = np.array([len(b["captions"]) >= 5 for b in beams]); idx = np.where(ok)[0]
tok = PTBTokenizer()
ct = tok.tokenize({f"{i}_{j}": [{"caption": beams[i]["beam"][j]["caption"]}] for i in idx for j in range(K)})
rt = tok.tokenize({f"{i}_{r}": [{"caption": beams[i]["captions"][r]}] for i in idx for r in range(5)})
refs = [[rt[f"{i}_{r}"][0] for r in range(5)] for i in idx]
M = np.full((N, K), np.nan)
for j in range(K):
    M[idx, j] = cider_vec(refs, [ct[f"{i}_{j}"][0] for i in idx])
np.savez_compressed(a.out + "_five.npz", names=np.array([b["image_name"] for b in beams]), M_five=M)
print("wrote", a.out + "_five.npz")
