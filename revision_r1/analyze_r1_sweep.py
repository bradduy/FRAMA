"""R1 revision: weight sweep and beam-width study on the OFFICIAL metric, with uncertainty.

Weight sweep: the same 300 Dirichlet draws as the submitted analysis (numpy default_rng(0),
symmetric Dirichlet over six agents, scaled to sum to six), scored with official per-candidate
CIDEr-D (matrices/<bench>.npz, M_all), with the corrected MBR agent. For every draw the paired
bootstrap over images (1,000 resamples, seed 0) gives the interval of its difference from equal
weights, and the percentile of equal weights is recomputed in every resample.
Beam width: Oracle@N and XRef@N for N = 20, 40, 80 on the full COCO test set, official metric,
with bootstrap intervals of the XRef@N differences.
Writes r1_sweep.json.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
sys.path.insert(0, V2); sys.path.insert(0, HERE)
from pilot_headroom import z, ROOT
import analyze_grounding as AG
import mbr_train_idf as MB

FEAT = {"coco": f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_csfn/features.npz",
        "nocaps": f"{ROOT}/inference_result/nocaps-bigg-blip/features.npz",
        "flickr": f"{ROOT}/inference_result/flickr-indomain-bigg-blip/features.npz"}
for b, p in MB.DUMPS.items():
    AG.PATHS[b] = p
NB = 1000


def agents(bench):
    beams = json.load(open(MB.DUMPS[bench])); K = min(len(b["beam"]) for b in beams)
    f = np.load(FEAT[bench], allow_pickle=True); pos = {str(n): i for i, n in enumerate(f["image_names"])}
    F = f["feats"].astype(np.float64)[[pos[b["image_name"]] for b in beams]][:, :K]
    G, _ = AG.grounding_features(bench, beams, K); V = AG.load_vqascore(bench, beams, K)
    S = np.load(os.path.join(V2, f"mbr_trainidf_{bench}.npy"))
    Z = np.stack([z(F[..., 0]), z(F[..., 1]), z(F[..., 2]), z(V), z(S), z(G[..., 0])], -1)
    return [b["image_name"] for b in beams], Z


def main():
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(6), size=300) * 6.0
    out = {"sweep": {}, "transfer": {}, "beam_width": {}}
    best_w = {}
    for bench in ("coco", "nocaps", "flickr"):
        names, Z = agents(bench)
        m = np.load(os.path.join(HERE, "matrices", f"{bench}.npz"), allow_pickle=True)
        assert [str(x) for x in m["names"]] == names
        M = m["M_all"]; N = len(names); ar = np.arange(N)
        P = np.stack([M[ar, (Z * w).sum(-1).argmax(1)] for w in W])           # 300 x N
        e = M[ar, Z.sum(-1).argmax(1)]                                         # equal weights
        C = np.random.default_rng(0).multinomial(N, np.ones(N) / N, size=NB).T  # N x NB resample counts
        bm = (P @ C) / N; be = (e @ C) / N                                     # 300 x NB, NB
        d = bm - be[None, :]
        lo, hi = np.percentile(d, 2.5, axis=1), np.percentile(d, 97.5, axis=1)
        pct = (bm < be[None, :]).mean(0)
        sc = P.mean(1); k = int(sc.argmax()); best_w[bench] = W[k]
        out["sweep"][bench] = {"equal": float(e.mean()), "pct_equal_beats": float((sc < e.mean()).mean()),
                               "ci_pct_equal_beats": np.percentile(pct, [2.5, 97.5]).tolist(),
                               "best": float(sc[k]), "best_minus_equal": float(sc[k] - e.mean()),
                               "ci_best_minus_equal": [float(lo[k]), float(hi[k])], "best_w": W[k].round(2).tolist(),
                               "n_sig_better": int((lo > 0).sum()), "n_sig_worse": int((hi < 0).sum()),
                               "median": float(np.median(sc))}
        out.setdefault("_Z", {})[bench] = None
        globals()[f"_Z_{bench}"] = (Z, M)
    for src in ("coco", "nocaps", "flickr"):
        for dst in ("coco", "nocaps", "flickr"):
            if src == dst:
                continue
            Z, M = globals()[f"_Z_{dst}"]; ar = np.arange(len(M))
            t = M[ar, (Z * best_w[src]).sum(-1).argmax(1)]; e = M[ar, Z.sum(-1).argmax(1)]
            dd = t - e; C = np.random.default_rng(0).multinomial(len(M), np.ones(len(M)) / len(M), size=NB).T
            bd = (dd @ C) / len(M)
            out["transfer"][f"{src}->{dst}"] = {"delta": float(dd.mean()), "ci": np.percentile(bd, [2.5, 97.5]).tolist()}
    # beam width, full COCO test set
    rows = {}
    for tag in ("coco", "coco_beam40", "coco_beam80"):
        p = os.path.join(HERE, "matrices", f"{tag}.npz")
        if not os.path.exists(p):
            continue
        m = np.load(p, allow_pickle=True); ok = m["ok5"]; M1 = m["M_one"][:, ok]; S4 = m["M_sel4"][:, ok]
        ar = np.arange(ok.sum())
        five = os.path.join(HERE, "matrices", f"{tag}_five.npz")
        Mo = (np.load(five)["M_five"] if os.path.exists(five) else m["M_all"])[ok]
        rows[tag] = {"oracle_5ref": float(m["M_all"][ok].max(1).mean()),
                     "xref_img": np.mean([M1[f, ar, S4[f].argmax(1)] for f in range(5)], 0),
                     "oracle_pick_heldout": float(np.mean([M1[f, ar, Mo.argmax(1)] for f in range(5)]))}
    if "coco" in rows:
        base = rows["coco"]["xref_img"]; n = len(base)
        C = np.random.default_rng(0).multinomial(n, np.ones(n) / n, size=NB).T
        for tag, r in rows.items():
            dd = r["xref_img"] - base; bd = (dd @ C) / n
            out["beam_width"][tag] = {"Oracle@N_5ref": r["oracle_5ref"], "XRef@N": float(r["xref_img"].mean()),
                                      "Oracle_pick_heldout": r["oracle_pick_heldout"],
                                      "XRef_minus_N20": float(dd.mean()), "ci": np.percentile(bd, [2.5, 97.5]).tolist()}
    out.pop("_Z", None)
    json.dump(out, open(os.path.join(HERE, "r1_sweep.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
