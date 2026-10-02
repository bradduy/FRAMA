"""C2b analysis: does a detector-grounded scorer add complementary signal?

Inputs: ground_<bench>.jsonl from detector_ground_scorer.py (pulled from GCP2),
the local beam dump and Stage-4 features. Metric: approximate CIDEr-D from
pilot_headroom.py (x1 scale). Also writes cocoeval-ready prediction files for
the admitted combinations under preds/<bench>/ for official scoring on GCP2.
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import CiderD, tok, z, ROOT
from pilot_c2_weighted_consensus import load

HERE = os.path.dirname(os.path.abspath(__file__))
NOUNS = None  # filled by grounding_features: per image, per candidate noun sets
PATHS = {
    "coco": f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_biggFull/coco_beam_dump.json",
    "nocaps": f"{ROOT}/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json",
}


def grounding_features(bench, beams, K, thr_h=0.2):
    """Per-candidate grounding signals from the jsonl.
    g_mean: mean max-detection score over the candidate's nouns
    g_min : weakest noun (hallucination-sensitive)
    g_hall: fraction of nouns with max score below thr_h (higher = worse)
    g_cnt : number of numeral-noun pairs whose detected count disagrees
    Candidates with no nouns get the image mean of that signal (neutral)."""
    path = os.path.join(HERE, f"ground_{bench}.jsonl")
    recs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line); recs[r["image_name"]] = r
    N = len(beams)
    G = np.full((N, K, 4), np.nan)
    global NOUNS
    NOUNS = [[frozenset() for _ in range(K)] for _ in range(N)]
    missing = 0
    for i, b in enumerate(beams):
        r = recs.get(b["image_name"])
        if r is None:
            missing += 1; continue
        ns = r["nouns"]
        for j, c in enumerate(r["cands"][:K]):
            NOUNS[i][j] = frozenset(c["nouns"])
            sc = [ns[n][0] for n in c["nouns"] if n in ns]
            if sc:
                G[i, j, 0] = np.mean(sc); G[i, j, 1] = np.min(sc)
                G[i, j, 2] = np.mean([s < thr_h for s in sc])
            G[i, j, 3] = sum(1 for n, k in c["nums"] if n in ns and ns[n][1] != k)
        for d in range(4):
            col = G[i, :, d]
            if np.isnan(col).any():
                col[np.isnan(col)] = np.nanmean(col) if not np.isnan(col).all() else 0.0
    return G, missing


def load_vqascore(bench, beams, K):
    """Optional per-candidate VQAScore from vqascore_<bench>.jsonl; NaN where absent."""
    path = os.path.join(HERE, f"vqascore_{bench}.jsonl")
    if not os.path.exists(path):
        return None
    recs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line); recs[r["image_name"]] = r["vqascore"]
    V = np.full((len(beams), K), np.nan)
    for i, b in enumerate(beams):
        v = recs.get(b["image_name"])
        if v is not None:
            V[i, : min(K, len(v))] = v[:K]
    print(f"{bench}: VQAScore available for {sum(1 for b in beams if b['image_name'] in recs)} images")
    return V


def rho_masked(S, M, mask):
    return np.nanmean([spearmanr(S[i], M[i]).correlation for i in np.where(mask)[0]
                       if np.std(M[i]) > 0 and np.std(S[i]) > 0])


def pair_rho(A, B, mask):
    return np.nanmean([spearmanr(A[i], B[i]).correlation for i in np.where(mask)[0]
                       if np.std(A[i]) > 0 and np.std(B[i]) > 0])


def report(bench, feats, M, splits, G, V=None):
    N, K, _ = feats.shape
    LM, BG, BL = feats[..., 0], feats[..., 1], feats[..., 2]
    gm, gmin, ghall, gcnt = G[..., 0], G[..., 1], -G[..., 2], -G[..., 3]
    idx = np.arange(N)
    if V is not None:
        has_v = ~np.isnan(V).any(axis=1)
        if has_v.sum() < N:
            print(f"  VQAScore rows restricted to {has_v.sum()} images for the VQA block")
        Vf = V.copy(); Vf[~has_v] = 0.0

    def cid(pick, mask=None):
        if mask is None:
            return M[idx, pick].mean()
        return M[mask][np.arange(mask.sum()), pick[mask]].mean()

    def rho(s):
        return np.nanmean([spearmanr(s[i], M[i]).correlation for i in range(N) if np.std(M[i]) > 0 and np.std(s[i]) > 0])

    print(f"\n===== {bench} =====")
    print("within-beam Spearman with candidate CIDEr:  " + "  ".join(
        f"{nm} {rho(s):.3f}" for nm, s in (("LM", LM), ("bigG", BG), ("BLIP", BL),
                                             ("g_mean", gm), ("g_min", gmin), ("g_hall", ghall))))
    print("scorer-scorer Spearman (complementarity):  " + "  ".join(
        f"{a}-{b} {np.nanmean([spearmanr(x[i], y[i]).correlation for i in range(N) if np.std(x[i])>0 and np.std(y[i])>0]):.3f}"
        for (a, x), (b, y) in (((("BLIP", BL), ("g_mean", gm))), (("bigG", BG), ("g_mean", gm)), (("LM", LM), ("g_mean", gm)))))
    base3 = z(LM) + z(BG) + z(BL)
    rows = [("zsum3 (baseline)", base3)]
    for nm, g in (("g_mean", gm), ("g_min", gmin), ("g_hall", ghall)):
        for w in (0.5, 1.0, 1.5, 2.0):
            rows.append((f"zsum3 + {w:.1f}*z({nm})", base3 + w * z(g)))
    rows.append(("zsum3 + z(g_mean) + 0.5*z(g_cnt)", base3 + z(gm) + 0.5 * z(gcnt)))
    rows.append(("z(BLIP)+z(g_mean) (no LM/bigG)", z(BL) + z(gm)))
    rows.append(("g_mean alone", gm))
    if V is not None:
        print(f"VQAScore within-beam Spearman with CIDEr {rho_masked(Vf, M, has_v):.3f}; "
              f"vs BLIP {pair_rho(Vf, BL, has_v):.3f}  vs bigG {pair_rho(Vf, BG, has_v):.3f}  vs LM {pair_rho(Vf, LM, has_v):.3f}")
        rows.append(("VQAScore alone", Vf))
        for w in (0.5, 1.0, 1.5, 2.0, 3.0):
            rows.append((f"zsum3 + {w:.1f}*z(VQAScore)", base3 + w * z(Vf)))
        rows.append(("z(BLIP)+z(VQAScore)", z(BL) + z(Vf)))
        rows.append(("zsum3 + z(VQAScore) + 0.5*z(g_mean)", base3 + z(Vf) + 0.5 * z(gm)))
    out_mask = None if splits is None else (splits == "out_domain")
    best = None
    for nm, s in rows:
        pick = s.argmax(1)
        mask_rows = has_v if (V is not None and "VQA" in nm) else None
        v = cid(pick, mask_rows) if mask_rows is not None else cid(pick)
        line = f"{nm:36s} CIDEr~{v:.4f}"
        if mask_rows is not None:
            line += f"  (n={mask_rows.sum()}; zsum3 on same rows {cid(base3.argmax(1), mask_rows):.4f})"
        if out_mask is not None:
            om = out_mask if mask_rows is None else (out_mask & mask_rows)
            line += f"  out~{cid(pick, om):.4f}"
        print(line)
        if nm != "zsum3 (baseline)" and mask_rows is None and (best is None or v > best[1]):
            best = (nm, v, pick)
    # Diagnostic: how often do the pick and the oracle-best candidate mention the
    # same noun set? If usually yes, object-presence grounding cannot separate
    # them by construction and the residual gap is attribute/relation/count.
    if NOUNS is not None:
        pick3 = base3.argmax(1); orc_idx = M.argmax(1)
        same = np.array([NOUNS[i][pick3[i]] == NOUNS[i][orc_idx[i]] for i in range(N)])
        sub = np.array([NOUNS[i][orc_idx[i]] <= NOUNS[i][pick3[i]] or NOUNS[i][pick3[i]] <= NOUNS[i][orc_idx[i]] for i in range(N)])
        gap_same = (M[idx, orc_idx] - M[idx, pick3])[same].mean() if same.any() else float("nan")
        gap_diff = (M[idx, orc_idx] - M[idx, pick3])[~same].mean() if (~same).any() else float("nan")
        print(f"noun-set diagnostic: pick vs oracle-best identical noun sets on {100*same.mean():.1f}% of images "
              f"(nested {100*sub.mean():.1f}%); mean CIDEr gap when identical {gap_same:.3f}, when different {gap_diff:.3f}")
    b0 = cid(base3.argmax(1)); orc = M.max(1).mean(); lm0 = cid(np.zeros(N, dtype=int))
    print(f"admission test: best combo {best[0]} = {best[1]:.4f} vs zsum3 {b0:.4f} (delta {100*(best[1]-b0):+.2f} CIDEr x100); "
          f"efficiency {100*(b0-lm0)/(orc-lm0):.1f}% -> {100*(best[1]-lm0)/(orc-lm0):.1f}% of Oracle@20 headroom")
    return best


if __name__ == "__main__":
    for bench in ("coco", "nocaps"):
        if not os.path.exists(os.path.join(HERE, f"ground_{bench}.jsonl")):
            print(f"{bench}: ground_{bench}.jsonl not found, skip"); continue
        feats, M, splits = load(bench)
        beams = json.load(open(PATHS[bench]))
        K = feats.shape[1]
        G, missing = grounding_features(bench, beams, K)
        print(f"{bench}: {len(beams)} images, {missing} without grounding record")
        if missing:  # partial pull: restrict every array to the scored images
            scored = {json.loads(l)["image_name"] for l in open(os.path.join(HERE, f"ground_{bench}.jsonl"))}
            keep = np.array([b["image_name"] in scored for b in beams])
            feats, M, G = feats[keep], M[keep], G[keep]
            NOUNS = [n for n, k in zip(NOUNS, keep) if k]
            splits = None if splits is None else splits[keep]
            beams = [b for b, k in zip(beams, keep) if k]
            print(f"  -> analysing the {keep.sum()} scored images only (partial run)")
        V = load_vqascore(bench, beams, K)
        best = report(bench, feats, M, None if bench == "coco" else splits, G, V)
        # write cocoeval-ready preds for the best combo
        cands = [[x["caption"] for x in b["beam"]][:K] for b in beams]
        recs = [{"image_name": b["image_name"], "captions": b["captions"],
                 "prediction": cands[i][int(best[2][i])]} for i, b in enumerate(beams)]
        tag = "zsum3_plus_ground_best"
        for sp in (["overall"] if bench == "coco" else ["overall", "out_domain"]):
            sel = recs if sp == "overall" else [r for r, b in zip(recs, beams) if b.get("split") == sp]
            json.dump(sel, open(os.path.join(HERE, "preds", bench, f"{tag}__{sp}.json"), "w"))
        print(f"wrote preds/{bench}/{tag}__*.json  ({best[0]})")
        # Paired subset files for the VQA block: same images for baseline and combos,
        # so the official metric comparison is like-for-like.
        if V is not None:
            has_v = ~np.isnan(V).any(axis=1)
            Vf = V.copy(); Vf[~has_v] = 0.0
            LM, BG, BL = feats[..., 0], feats[..., 1], feats[..., 2]
            base3 = z(LM) + z(BG) + z(BL)
            combos = {
                "vqasub_zsum3": base3,
                "vqasub_blip_top1": BL,
                "vqasub_vqa_alone": Vf,
                "vqasub_zsum3_vqa_w0.5": base3 + 0.5 * z(Vf),
                "vqasub_zsum3_vqa_w1.0": base3 + 1.0 * z(Vf),
                "vqasub_zsum3_vqa_w2.0": base3 + 2.0 * z(Vf),
                "vqasub_zsum3_vqa_w1.0_ground": base3 + 1.0 * z(Vf) + 0.5 * z(G[..., 0]),
                "vqasub_blip_vqa": z(BL) + z(Vf),
                "vqasub_oracle20_approx": M,
            }
            for cname, s in combos.items():
                pick = s.argmax(1)
                r2 = [{"image_name": b["image_name"], "captions": b["captions"],
                       "prediction": cands[i][int(pick[i])]} for i, b in enumerate(beams) if has_v[i]]
                for sp in (["overall"] if bench == "coco" else ["overall", "out_domain"]):
                    sel = r2 if sp == "overall" else [r for r in r2 if next(b for b in beams if b["image_name"] == r["image_name"]).get("split") == sp]
                    json.dump(sel, open(os.path.join(HERE, "preds", bench, f"{cname}__{sp}.json"), "w"))
            print(f"wrote paired VQA-subset preds for {bench}: {len(combos)} rules x splits, n={int(has_v.sum())} images")
