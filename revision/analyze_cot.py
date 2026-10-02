"""System 1 versus System 2 use of the same Qwen2-VL-2B judge on 500 COCO images (every other image
of the 1,000-image subsample). Reports within-beam correlation with CIDEr, overlap with BLIP-ITM,
saturation of the scores, ties within a beam, time per image, and writes prediction files
preds/coco/cot500_<rule>__overall.json for official evaluation."""
import json, os, sys, glob
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from pilot_headroom import z
from pilot_c2_weighted_consensus import load
import analyze_grounding as AG
feats, M, _ = load("coco"); beams = json.load(open(AG.PATHS["coco"])); N, K = M.shape
G, _ = AG.grounding_features("coco", beams, K); V = AG.load_vqascore("coco", beams, K)
S = np.load(os.path.join(HERE, "mbr_self_coco.npy"))
rows = {}
for f in glob.glob(os.path.join(HERE, "revision", "cot_shard*.jsonl")):
    for l in open(f): r = json.loads(l); rows[r["image_name"]] = r
idx_of = {b["image_name"]: i for i, b in enumerate(beams)}
names = sorted(rows, key=lambda n: idx_of[n]); I = np.array([idx_of[n] for n in names])
S1 = np.array([rows[n]["s1"] for n in names]); S2 = np.array([rows[n]["s2"] for n in names])
LM, BG, BL, VQ, MB, GR, Ms = feats[I, :K, 0], feats[I, :K, 1], feats[I, :K, 2], V[I], S[I], G[I, :, 0], M[I]
def rho(A, B): return np.nanmean([spearmanr(A[i], B[i]).correlation for i in range(len(I)) if np.std(A[i]) > 0 and np.std(B[i]) > 0])
def approx(sc): return 100 * Ms[np.arange(len(I)), sc.argmax(1)].mean()
def ties(A): return np.mean([len(np.unique(np.round(a, 3))) < len(a) * 0.5 for a in A])
t1 = np.mean([rows[n]["t1"] for n in names]); t2 = np.mean([rows[n]["t2"] for n in names]); gt = np.mean([rows[n]["gen_tokens"] for n in names])
out = {"n_images": len(I), "time_s1_s_per_img": t1, "time_s2_s_per_img": t2, "gen_tokens_per_candidate": gt}
for tag, A in (("S1", S1), ("S2", S2)):
    out[tag] = {"rho_cider": rho(A, Ms), "rho_blip": rho(A, BL), "rho_bigG": rho(A, BG),
                "saturated_frac": float(np.mean((A > 0.99) | (A < 0.01))), "beams_mostly_tied": float(ties(A)),
                "alone": approx(A), "zsum3_plus": approx(z(LM) + z(BG) + z(BL) + z(A)),
                "six_swap": approx(z(LM) + z(BG) + z(BL) + z(A) + z(MB) + z(GR))}
out["rho_S1_S2"] = rho(S1, S2)
out["reference"] = {"zsum3": approx(z(LM) + z(BG) + z(BL)), "six_blip2": approx(z(LM) + z(BG) + z(BL) + z(VQ) + z(MB) + z(GR)),
                    "five_no_judge": approx(z(LM) + z(BG) + z(BL) + z(MB) + z(GR))}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "revision", "cot_summary.json"), "w"), indent=1)
rules = {"zsum3": z(LM) + z(BG) + z(BL), "six_blip2": z(LM) + z(BG) + z(BL) + z(VQ) + z(MB) + z(GR),
         "s1_alone": S1, "s2_alone": S2, "zsum3_s1": z(LM) + z(BG) + z(BL) + z(S1), "zsum3_s2": z(LM) + z(BG) + z(BL) + z(S2),
         "six_s1": z(LM) + z(BG) + z(BL) + z(S1) + z(MB) + z(GR), "six_s2": z(LM) + z(BG) + z(BL) + z(S2) + z(MB) + z(GR)}
for r, sc in rules.items():
    pick = sc.argmax(1)
    recs = [{"image_name": beams[i]["image_name"], "captions": beams[i]["captions"], "prediction": beams[i]["beam"][int(pick[k])]["caption"]} for k, i in enumerate(I)]
    json.dump(recs, open(os.path.join(HERE, "preds", "coco", f"cot500_{r}__overall.json"), "w"))
print("wrote", len(rules), "prediction files")
