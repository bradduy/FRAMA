"""Reviewer items M1, M2 and Q4 for the cross-reference protocol.
M1  human baseline on the same single-held-out-reference scale: for fold f, another human reference
    r_{(f+1) mod 5} is scored against r_f alone; averaged over the five folds.
M2  paired bootstrap over images (10,000 resamples) of the recoverable headroom XRef@20 - s and of
    the surviving share (XRef@20 - s) / (Oracle@20 pick - s), with s the three-agent z-sum,
    all on the single-held-out-reference scale.
Q4  the same protocol with SPICE as both selection and scoring metric (--spice, on a subsample)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/ZeroShotImageCap/IFCap/evaluation"))
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
ap = argparse.ArgumentParser(); ap.add_argument("--beam_dump", required=True); ap.add_argument("--preds_dir", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--spice", action="store_true"); ap.add_argument("--stride", type=int, default=1)
a = ap.parse_args()
beams = [b for b in json.load(open(a.beam_dump)) if len(b["captions"]) >= 5][::a.stride]
N = len(beams); K = min(len(b["beam"]) for b in beams); names = [b["image_name"] for b in beams]
def pred(rule):
    d = {x["image_name"]: x["prediction"] for x in json.load(open(os.path.join(a.preds_dir, f"{rule}__overall.json")))}
    return [d[n] for n in names]
zs = pred("zsum3_lm_bigG_blip")
tok = PTBTokenizer()
ct = tok.tokenize({f"{i}_{j}": [{"caption": beams[i]["beam"][j]["caption"]}] for i in range(N) for j in range(K)})
rt = tok.tokenize({f"{i}_{r}": [{"caption": c}] for i in range(N) for r, c in enumerate(beams[i]["captions"][:5])})
zt = tok.tokenize({str(i): [{"caption": zs[i]}] for i in range(N)})
if a.spice:
    from pycocoevalcap.spice.spice import Spice
    def metric(g, r):
        _, sc = Spice().compute_score(g, r); return np.array([s["All"]["f"] for s in sc], float)
else:
    def metric(g, r): return np.asarray(Cider().compute_score(g, r)[1], float)
res_all = {f"{i}_{j}": ct[f"{i}_{j}"] for i in range(N) for j in range(K)}
g5 = {f"{i}_{j}": [rt[f"{i}_{r}"][0] for r in range(5)] for i in range(N) for j in range(K)}
pick5 = metric(g5, res_all).reshape(N, K).argmax(1)
X, O, Z, H = np.zeros((5, N)), np.zeros((5, N)), np.zeros((5, N)), np.zeros((5, N))
for f in range(5):
    others = [r for r in range(5) if r != f]
    g4 = {f"{i}_{j}": [rt[f"{i}_{r}"][0] for r in others] for i in range(N) for j in range(K)}
    pick = metric(g4, res_all).reshape(N, K).argmax(1)
    g1 = {str(i): [rt[f"{i}_{f}"][0]] for i in range(N)}
    X[f] = metric(g1, {str(i): ct[f"{i}_{pick[i]}"] for i in range(N)})
    O[f] = metric(g1, {str(i): ct[f"{i}_{pick5[i]}"] for i in range(N)})
    Z[f] = metric(g1, {str(i): zt[str(i)] for i in range(N)})
    H[f] = metric(g1, {str(i): rt[f"{i}_{(f + 1) % 5}"] for i in range(N)})
    print("fold", f, "xref %.2f oracle %.2f zsum %.2f human %.2f" % (100 * X[f].mean(), 100 * O[f].mean(), 100 * Z[f].mean(), 100 * H[f].mean()), flush=True)
x, o, zz, h = X.mean(0), O.mean(0), Z.mean(0), H.mean(0)
rng = np.random.default_rng(0); B = 10000
idx = rng.integers(0, N, size=(B, N))
head = (x - zz)[idx].mean(1) * 100
share = ((x - zz)[idx].mean(1)) / ((o - zz)[idx].mean(1))
out = {"metric": "SPICE" if a.spice else "CIDEr", "n_images": N,
       "xref": 100 * x.mean(), "oracle5_pick": 100 * o.mean(), "zsum3": 100 * zz.mean(), "human_single_ref": 100 * h.mean(),
       "recoverable_headroom": 100 * (x - zz).mean(), "recoverable_headroom_ci95": [float(np.percentile(head, 2.5)), float(np.percentile(head, 97.5))],
       "apparent_headroom": 100 * (o - zz).mean(), "share": float((x - zz).mean() / (o - zz).mean()),
       "share_ci95": [float(np.percentile(share, 2.5)), float(np.percentile(share, 97.5))]}
print(json.dumps(out, indent=1)); json.dump(out, open(a.out, "w"), indent=1)
