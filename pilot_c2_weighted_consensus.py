"""C2 pilot: label-free reliability-weighted consensus vs equal-weight fusion.

Weights are estimated from the scorers' outputs alone (no references):
  spectral : leading eigenvector of the mean pairwise rank-correlation matrix
             among scorers (Parisi et al. 2014 style; under conditional
             independence the off-diagonal entries factorise as r_i * r_j)
  mallows  : EM over a Mallows-type model: dispersion of scorer s = mean
             normalised Kendall distance to the current weighted consensus;
             weight_s = log((1-d_s)/d_s) (Klementiev et al. 2008 style)
Also an oracle-weight upper bound (weights fit on approximate CIDEr, inductive
across benchmarks) to show the ceiling of any linear reweighting.
Metric: approximate CIDEr-D (pilot_headroom.CiderD). Values x1 scale.
"""
import json, sys, os, itertools
import numpy as np
from scipy.stats import spearmanr, kendalltau
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import CiderD, tok, borda, z, ROOT

def load(bench):
    if bench == "coco":
        bp = f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_biggFull/coco_beam_dump.json"
        fp = f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_csfn/features.npz"
    else:
        bp = f"{ROOT}/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json"
        fp = f"{ROOT}/inference_result/nocaps-bigg-blip/features.npz"
    beams = json.load(open(bp)); f = np.load(fp, allow_pickle=True)
    pos = {str(n): i for i, n in enumerate(f["image_names"])}
    order = np.array([pos[b["image_name"]] for b in beams])
    feats = f["feats"].astype(np.float64)[order]
    K = feats.shape[1]
    cands = [[x["caption"] for x in b["beam"]][:K] for b in beams]
    cider = CiderD([[tok(c) for c in b["captions"]] for b in beams])
    M = np.array([[cider.score_one(i, tok(cands[i][j])) for j in range(K)] for i in range(len(beams))])
    splits = np.array([b.get("split", "overall") for b in beams])
    return feats, M, splits

def ranks_desc(s):
    order = np.argsort(-s, axis=1); r = np.empty_like(order)
    r[np.arange(s.shape[0])[:, None], order] = np.arange(s.shape[1])[None, :]
    return r

def weighted_borda(sigs, w):
    K = sigs[0].shape[1]
    return sum(wi * (K - 1 - ranks_desc(s)) for wi, s in zip(w, sigs)) / sum(w)

def weighted_zsum(sigs, w):
    return sum(wi * z(s) for wi, s in zip(w, sigs)) / sum(w)

def spectral_weights(sigs):
    S = len(sigs); N = sigs[0].shape[0]
    C = np.zeros((S, S))
    for i, j in itertools.combinations(range(S), 2):
        c = np.nanmean([spearmanr(sigs[i][n], sigs[j][n]).correlation for n in range(N)])
        C[i, j] = C[j, i] = c
    # rank-one fit of off-diagonals: C_ij = r_i r_j  ->  r from leading eigvec
    vals, vecs = np.linalg.eigh(C + np.eye(S) * 0)  # diag is 0
    v = np.abs(vecs[:, -1]) * np.sqrt(max(vals[-1], 1e-9))
    r = np.clip(v, 1e-3, 0.999)
    # Parisi-style: optimal linear weight ~ r / (1 - r^2) for balanced binary case; use it as heuristic
    w = r / (1 - r ** 2)
    return w / w.sum(), C, r

def mallows_em_weights(sigs, iters=10):
    S = len(sigs); N, K = sigs[0].shape
    w = np.ones(S) / S
    R = [ranks_desc(s) for s in sigs]
    for _ in range(iters):
        cons = weighted_borda(sigs, w)
        rc = ranks_desc(cons)
        d = np.zeros(S)
        for s in range(S):
            # normalised Kendall distance to consensus, averaged over images
            taus = np.array([kendalltau(R[s][n], rc[n]).correlation for n in range(N)])
            d[s] = np.nanmean((1 - taus) / 2)  # in (0,1); 0 = identical
        d = np.clip(d, 1e-3, 0.5 - 1e-3)
        w = np.log((1 - d) / d); w = w / w.sum()
    return w, d

def oracle_weights(sigs, M, grid=11):
    best = (-1, None)
    for a in np.linspace(0, 1, grid):
        for b in np.linspace(0, 1 - a, grid):
            w = np.array([a, b, 1 - a - b])
            if w.sum() <= 0: continue
            v = M[np.arange(len(M)), weighted_zsum(sigs, w).argmax(1)].mean()
            if v > best[0]: best = (v, w)
    return best[1]

def evaluate(name, sigs, M, w, splits=None):
    out = {}
    for agg, fn in (("wBorda", weighted_borda), ("wZsum", weighted_zsum)):
        pick = fn(sigs, w).argmax(1)
        v = M[np.arange(len(M)), pick].mean()
        s = f"{name:34s} {agg:6s} w={np.round(w,3)}  CIDEr~{v:.4f}"
        if splits is not None:
            m = splits == "out_domain"; s += f"  out~{M[m][np.arange(m.sum()), pick[m]].mean():.4f}"
        print(s); out[agg] = v
    return out

data = {b: load(b) for b in ("coco", "nocaps")}
for bench, (feats, M, splits) in data.items():
    print(f"\n===== {bench} =====")
    sigs = [feats[..., 0], feats[..., 1], feats[..., 2]]  # LM, bigG, BLIP
    sp = None if bench == "coco" else splits
    equal = np.ones(3) / 3
    evaluate("equal weights (baseline)", sigs, M, equal, sp)
    w_spec, C, r = spectral_weights(sigs)
    print(f"  mean pairwise Spearman among scorers: LM-bigG {C[0,1]:.3f}  LM-BLIP {C[0,2]:.3f}  bigG-BLIP {C[1,2]:.3f}; r={np.round(r,3)}")
    evaluate("spectral (label-free, in-benchmark)", sigs, M, w_spec, sp)
    w_em, d = mallows_em_weights(sigs)
    print(f"  Mallows-EM dispersions d={np.round(d,3)}")
    evaluate("Mallows-EM (label-free, in-benchmark)", sigs, M, w_em, sp)
    other = "nocaps" if bench == "coco" else "coco"
    osigs = [data[other][0][..., k] for k in range(3)]
    w_spec_x, _, _ = spectral_weights(osigs); w_em_x, _ = mallows_em_weights(osigs)
    evaluate(f"spectral fit on {other} (cross-bench)", sigs, M, w_spec_x, sp)
    evaluate(f"Mallows-EM fit on {other} (cross-bench)", sigs, M, w_em_x, sp)
    w_or = oracle_weights(osigs, data[other][1])
    evaluate(f"ORACLE weights fit on {other} (upper bd)", sigs, M, w_or, sp)
    w_or_in = oracle_weights(sigs, M)
    evaluate("ORACLE weights in-benchmark (cheating)", sigs, M, w_or_in, sp)
    print(f"  Oracle@20 ceiling ~{M.max(1).mean():.4f}")
