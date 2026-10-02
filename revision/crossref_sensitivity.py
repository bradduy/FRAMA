"""Reviewer item 3: is the cross-reference oracle a stable estimate? Select per image the
candidate with the highest official CIDEr-D against k of the four non-held-out references
(k = 1..4, one rotated subset per fold), score it against the held-out reference alone, and
average over the five folds. If the estimate grows with k and flattens, XRef@N with k = 4 is
a conservative estimate of what reference-like selection information buys; the plain
Oracle@N corresponds to k = 5 including the held-out reference."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/ZeroShotImageCap/IFCap/evaluation"))
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
ap = argparse.ArgumentParser(); ap.add_argument("--beam_dump", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
beams = [b for b in json.load(open(a.beam_dump)) if len(b["captions"]) >= 5]
N = len(beams); K = min(len(b["beam"]) for b in beams)
tok = PTBTokenizer()
ct = tok.tokenize({f"{i}_{j}": [{"caption": beams[i]["beam"][j]["caption"]}] for i in range(N) for j in range(K)})
rt = tok.tokenize({f"{i}_{r}": [{"caption": c}] for i in range(N) for r, c in enumerate(beams[i]["captions"][:5])})
res_all = {f"{i}_{j}": ct[f"{i}_{j}"] for i in range(N) for j in range(K)}
def cid(g, r): return np.asarray(Cider().compute_score(g, r)[1], float)
out = {"n_images": N, "K": K, "by_k": {}}
for k in (1, 2, 3, 4):
    vals = []
    for f in range(5):
        others = [r for r in range(5) if r != f]
        sel = [others[(f + t) % 4] for t in range(k)]
        g = {f"{i}_{j}": [rt[f"{i}_{r}"][0] for r in sel] for i in range(N) for j in range(K)}
        pick = cid(g, res_all).reshape(N, K).argmax(1)
        g1 = {str(i): [rt[f"{i}_{f}"][0]] for i in range(N)}
        vals.append(float(100 * cid(g1, {str(i): ct[f"{i}_{pick[i]}"] for i in range(N)}).mean()))
    out["by_k"][k] = {"mean": float(np.mean(vals)), "folds": vals}
    print(k, round(np.mean(vals), 2), [round(v, 2) for v in vals], flush=True)
json.dump(out, open(a.out, "w"), indent=1)
