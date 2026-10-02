"""Label-free reliability weighting at six agents (spectral rank-one fit of the agent
agreement matrix; Mallows-EM dispersion weights), three benchmarks, approximate metric,
plus cross-benchmark transfer of the fitted weights. Writes preds/<bench>/rel6_<method>__<split>.json
for the in-benchmark weights so they can be scored officially."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import CiderD, tok, z, ROOT
from pilot_c2_weighted_consensus import load, spectral_weights, mallows_em_weights
import analyze_grounding as AG
HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = ["LM", "bigG", "BLIP", "VQA", "MBR", "ground"]
AG.PATHS["flickr"] = f"{ROOT}/inference_result/flickr-indomain-bigg-blip/flickr30k_beam_dump.json"

def load_bench(bench):
    beams = json.load(open(AG.PATHS[bench]))
    if bench == "flickr":
        N = len(beams); K = min(len(b["beam"]) for b in beams)
        f = np.load(f"{ROOT}/inference_result/flickr-indomain-bigg-blip/features.npz", allow_pickle=True)
        pos = {str(n): i for i, n in enumerate(f["image_names"])}; order = [pos[b["image_name"]] for b in beams]
        feats = f["feats"].astype(np.float64)[order][:, :K]
        ct = [[tok(x["caption"]) for x in b["beam"][:K]] for b in beams]
        cider = CiderD([[tok(c) for c in b["captions"]] for b in beams])
        M = np.array([[cider.score_one(i, ct[i][j]) for j in range(K)] for i in range(N)])
    else:
        feats, M, _ = load(bench)
    N, K = M.shape
    G, miss = AG.grounding_features(bench, beams, K); V = AG.load_vqascore(bench, beams, K)
    assert miss == 0 and V is not None and not np.isnan(V).any()
    S = np.load(os.path.join(HERE, f"mbr_self_{bench}.npy"))
    sigs = [feats[..., 0], feats[..., 1], feats[..., 2], V, S, G[..., 0]]
    return beams, M, sigs

def write(bench, beams, name, score):
    K = score.shape[1]; cands = [[x["caption"] for x in b["beam"]][:K] for b in beams]; pick = score.argmax(1)
    recs = [{"image_name": b["image_name"], "captions": b["captions"], "prediction": cands[i][int(pick[i])]} for i, b in enumerate(beams)]
    for sp in (["overall"] if bench in ("coco", "flickr") else ["overall", "out_domain"]):
        sel = recs if sp == "overall" else [r for r, b in zip(recs, beams) if b.get("split") == sp]
        json.dump(sel, open(os.path.join(HERE, "preds", bench, f"{name}__{sp}.json"), "w"))

data = {b: load_bench(b) for b in ("coco", "nocaps", "flickr")}
W = {}
for bench, (beams, M, sigs) in data.items():
    N = M.shape[0]; idx = np.arange(N)
    Z = [z(s) for s in sigs]
    eq = 100 * M[idx, sum(Z).argmax(1)].mean()
    ws, C, r = spectral_weights(sigs)
    wm, d = mallows_em_weights(sigs, iters=8)
    sc_s = 100 * M[idx, sum(w * zz for w, zz in zip(ws, Z)).argmax(1)].mean()
    sc_m = 100 * M[idx, sum(w * zz for w, zz in zip(wm, Z)).argmax(1)].mean()
    W[bench] = {"spectral": ws.tolist(), "mallows": wm.tolist(), "equal": eq, "spectral_cider": sc_s, "mallows_cider": sc_m,
                "agreement": C.round(3).tolist(), "dispersion": d.round(3).tolist()}
    print(f"\n===== {bench}: equal {eq:.2f} | spectral {sc_s:.2f} ({sc_s-eq:+.2f}) | Mallows-EM {sc_m:.2f} ({sc_m-eq:+.2f})")
    print("  spectral w:", dict(zip(AGENTS, np.round(6*ws, 2))), " Mallows w:", dict(zip(AGENTS, np.round(6*wm, 2))))
    print("  pairwise Spearman:\n   " + "\n   ".join(f"{AGENTS[i]:6s} " + " ".join(f"{C[i,j]:+.2f}" for j in range(6)) for i in range(6)))
    write(bench, beams, "rel6_spectral", sum(w * zz for w, zz in zip(ws, Z)))
    write(bench, beams, "rel6_mallows", sum(w * zz for w, zz in zip(wm, Z)))
print("\n===== cross-benchmark transfer of reliability weights (approximate) =====")
for src in ("coco", "nocaps"):
    for method in ("spectral", "mallows"):
        w = np.array(W[src][method])
        for bench, (beams, M, sigs) in data.items():
            if bench == src: continue
            Z = [z(s) for s in sigs]; N = M.shape[0]; idx = np.arange(N)
            sc = 100 * M[idx, sum(wi * zz for wi, zz in zip(w, Z)).argmax(1)].mean()
            print(f"{method:8s} fit on {src:6s} -> {bench:7s}: {sc:.2f} vs equal {W[bench]['equal']:.2f} ({sc-W[bench]['equal']:+.2f})")
json.dump(W, open(os.path.join(HERE, "reliability6_summary.json"), "w"), indent=1)
