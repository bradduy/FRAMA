"""Leave-one-out ablation of the six-agent stack and weight-sensitivity analysis,
all three benchmarks, from the stored agent scores (no GPU).

Writes preds/<bench>/loo_minus_<agent>__<split>.json for the five agents whose
removal is not already on disk (minus LM = stack5_noLM..., already official),
preds/<bench>/wtransfer_from_<other>__<split>.json for equal-vs-transferred
weights, and prints approximate CIDeR for every rule plus the Dirichlet sweep
summary. Usage: python3 analyze_ablation.py [n_sweep]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import CiderD, tok, z, ROOT
from pilot_c2_weighted_consensus import load
import analyze_grounding as AG

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = ["LM", "bigG", "BLIP", "VQA", "MBR", "ground"]
NSWEEP = int(sys.argv[1]) if len(sys.argv) > 1 else 300
AG.PATHS["flickr"] = f"{ROOT}/inference_result/flickr-indomain-bigg-blip/flickr30k_beam_dump.json"


def load_bench(bench):
    beams = json.load(open(AG.PATHS[bench]))
    if bench == "flickr":
        N = len(beams); K = min(len(b["beam"]) for b in beams)
        f = np.load(f"{ROOT}/inference_result/flickr-indomain-bigg-blip/features.npz", allow_pickle=True)
        pos = {str(n): i for i, n in enumerate(f["image_names"])}
        order = [pos[b["image_name"]] for b in beams]
        feats = f["feats"].astype(np.float64)[order][:, :K]
        cands_tok = [[tok(x["caption"]) for x in b["beam"][:K]] for b in beams]
        cider = CiderD([[tok(c) for c in b["captions"]] for b in beams])
        M = np.array([[cider.score_one(i, cands_tok[i][j]) for j in range(K)] for i in range(N)])
        splits = None
    else:
        feats, M, splits = load(bench)
    N, K = M.shape
    G, miss = AG.grounding_features(bench, beams, K)
    V = AG.load_vqascore(bench, beams, K)
    assert miss == 0 and V is not None and not np.isnan(V).any(), f"{bench}: agent files incomplete"
    S = np.load(os.path.join(HERE, f"mbr_self_{bench}.npy"))
    Z = np.stack([z(feats[..., 0]), z(feats[..., 1]), z(feats[..., 2]), z(V), z(S), z(G[..., 0])], axis=-1)  # N,K,6
    return beams, M, Z


def write_preds(bench, beams, name, score):
    K = score.shape[1]
    cands = [[x["caption"] for x in b["beam"]][:K] for b in beams]
    pick = score.argmax(1)
    recs = [{"image_name": b["image_name"], "captions": b["captions"], "prediction": cands[i][int(pick[i])]} for i, b in enumerate(beams)]
    splits = ["overall"] if bench in ("coco", "flickr") else ["overall", "in_domain", "near_domain", "out_domain"]
    os.makedirs(os.path.join(HERE, "preds", bench), exist_ok=True)
    for sp in splits:
        sel = recs if sp == "overall" else [r for r, b in zip(recs, beams) if b.get("split") == sp]
        json.dump(sel, open(os.path.join(HERE, "preds", bench, f"{name}__{sp}.json"), "w"))
    return pick


data = {b: load_bench(b) for b in ("coco", "nocaps", "flickr")}
rng = np.random.default_rng(0)
W = rng.dirichlet(np.ones(6), size=NSWEEP) * 6.0  # equal weights = all ones
summary = {}
print("\n===== leave-one-out (approximate CIDEr) =====")
for bench, (beams, M, Z) in data.items():
    N = M.shape[0]; idx = np.arange(N)
    full = Z.sum(-1); pf = full.argmax(1); base = M[idx, pf].mean()
    line = [f"{bench:7s} six-agent {100*base:.2f}"]
    for a, name in enumerate(AGENTS):
        s = np.delete(Z, a, axis=-1).sum(-1)
        p = s.argmax(1)
        d = 100 * (M[idx, p].mean() - base)
        line.append(f"-{name} {d:+.2f}")
        if name != "LM":
            write_preds(bench, beams, f"loo_minus_{name}", s)
    print("  ".join(line))
print("\n===== weight sweep (Dirichlet, sum 6; approximate CIDEr) =====")
sweep = {}
for bench, (beams, M, Z) in data.items():
    N = M.shape[0]; idx = np.arange(N)
    eq = 100 * M[idx, Z.sum(-1).argmax(1)].mean()
    sc = np.array([100 * M[idx, (Z * w).sum(-1).argmax(1)].mean() for w in W])
    best = sc.argmax()
    sweep[bench] = {"equal": eq, "best": float(sc[best]), "best_w": W[best].round(2).tolist(),
                    "pct_equal_beats": float((sc < eq).mean()), "median": float(np.median(sc)),
                    "frac_within_0.5_of_best": float((sc >= sc[best] - 0.5).mean()), "p10": float(np.percentile(sc, 10)), "p90": float(np.percentile(sc, 90))}
    print(f"{bench:7s} equal {eq:.2f} | sweep median {np.median(sc):.2f} p10 {np.percentile(sc,10):.2f} p90 {np.percentile(sc,90):.2f} best {sc[best]:.2f} w={W[best].round(2).tolist()} | equal beats {100*(sc<eq).mean():.0f}% of samples")
print("\n===== cross-bench weight transfer (best sweep weights of one bench applied to the others) =====")
for src in ("coco", "nocaps"):
    w = np.array(sweep[src]["best_w"])
    for bench, (beams, M, Z) in data.items():
        if bench == src: continue
        N = M.shape[0]; idx = np.arange(N)
        s = (Z * w).sum(-1); p = s.argmax(1)
        eq = 100 * M[idx, Z.sum(-1).argmax(1)].mean()
        print(f"weights from {src:6s} -> {bench:7s}: {100*M[idx,p].mean():.2f} vs equal {eq:.2f} ({100*M[idx,p].mean()-eq:+.2f})")
        write_preds(bench, beams, f"wtransfer_from_{src}", s)
json.dump(sweep, open(os.path.join(HERE, "weight_sweep_summary.json"), "w"), indent=1)
print("\npreds written:", sum(len([f for f in os.listdir(os.path.join(HERE, 'preds', b)) if f.startswith(('loo_', 'wtransfer_'))]) for b in data))
