"""C3 analysis: verifier ladder. For every scores_<scorer>_<bench>.jsonl present,
report within-beam discrimination, complementarity with the existing agents,
and the gain when the scorer replaces or joins them in the equal-weight stack.
Attainable ceiling from the cross-reference oracle is reused from
pilot_agent_stack when the VQA subset is available; otherwise Oracle@20 only."""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import z
from pilot_c2_weighted_consensus import load
import analyze_grounding as AG

HERE = os.path.dirname(os.path.abspath(__file__))

def load_scores(path, beams, K):
    recs = {}
    for line in open(path):
        r = json.loads(line); recs[r["image_name"]] = r["scores"]
    X = np.full((len(beams), K), np.nan)
    for i, b in enumerate(beams):
        v = recs.get(b["image_name"])
        if v is not None:
            X[i, : min(K, len(v))] = v[:K]
    return X

def rho(a, b, mask):
    return np.nanmean([spearmanr(a[i], b[i]).correlation for i in np.where(mask)[0] if np.std(a[i]) > 0 and np.std(b[i]) > 0])

for bench in ("coco", "nocaps"):
    files = sorted(glob.glob(os.path.join(HERE, f"scores_*_{bench}.jsonl")))
    if not files:
        print(f"{bench}: no scores_*_{bench}.jsonl yet"); continue
    feats, M, splits = load(bench)
    beams = json.load(open(AG.PATHS[bench]))
    N, K = M.shape
    LM, BG, BL = feats[..., 0], feats[..., 1], feats[..., 2]
    G, _ = AG.grounding_features(bench, beams, K)
    V = AG.load_vqascore(bench, beams, K)
    S = np.load(os.path.join(HERE, f"mbr_self_{bench}.npy"))
    idx = np.arange(N)
    print(f"\n===== {bench} verifier ladder =====")
    for f in files:
        name = os.path.basename(f).split("_")[1]
        X = load_scores(f, beams, K)
        mask = ~np.isnan(X).any(axis=1)
        if V is not None:
            mask &= ~np.isnan(V).any(axis=1)
        Xf = X.copy(); Xf[~mask] = 0.0
        Vf = None if V is None else np.nan_to_num(V)
        def cid(s):
            pick = s.argmax(1); return M[mask][np.arange(mask.sum()), pick[mask]].mean()
        base3 = z(LM) + z(BG) + z(BL)
        stack6 = base3 + (z(Vf) if Vf is not None else 0) + z(S) + z(G[..., 0])
        print(f"[{name}] n={mask.sum()}  rho(quality)={rho(Xf, M, mask):.3f}  rho(BLIP)={rho(Xf, BL, mask):.3f}  rho(bigG)={rho(Xf, BG, mask):.3f}  rho(LM)={rho(Xf, LM, mask):.3f}"
              + (f"  rho(VQA)={rho(Xf, Vf, mask):.3f}" if Vf is not None else ""))
        rows = [("zsum3", base3), (f"{name} alone", Xf), (f"zsum3 + {name}", base3 + z(Xf)),
                (f"zsum3 with bigG->{name}", z(LM) + z(Xf) + z(BL)), (f"zsum3 with BLIP->{name}", z(LM) + z(BG) + z(Xf)),
                ("stack6", stack6), (f"stack6 + {name}", stack6 + z(Xf))]
        for nm, s in rows:
            print(f"   {nm:28s} CIDEr~{cid(s):.4f}")
        print(f"   {'Oracle@20 (same rows)':28s} CIDEr~{M[mask].max(1).mean():.4f}")
