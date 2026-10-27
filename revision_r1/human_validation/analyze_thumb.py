"""Analyses T1 to T3 of thumb_prereg.md. Inputs: thumb_cider.npz, thumb_chair.json (pulled from
GCP2). Writes thumb_results.json and prints a summary. Bootstrap: 10,000 paired resamples of
images, seed 0, 95% percentile intervals."""
import json, os
import numpy as np
from scipy.stats import kendalltau

HERE = os.path.dirname(os.path.abspath(__file__)); B = 10000
d = np.load(os.path.join(HERE, "thumb_cider.npz"), allow_pickle=True)
C5, C4, C1, H, P = d["C5"], d["C4"], d["C1"], d["H"], d["P"]
N, K = C5.shape; ar = np.arange(N)
I = np.random.default_rng(0).integers(0, N, size=(B, N))
ci = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
out = {"n_images": int(N)}

# T1
o = C5.argmax(1)
O_img = np.mean([C1[f, ar, o] - C1[f].mean(1) for f in range(5)], 0)
X_img = np.mean([C1[f, ar, C4[f].argmax(1)] - C1[f].mean(1) for f in range(5)], 0)
A_img = C5[ar, o] - C5.mean(1)
HO_img = H[ar, o] - H.mean(1)
HX_img = np.mean([H[ar, C4[f].argmax(1)] - H.mean(1) for f in range(5)], 0)
gap_img = HO_img - HX_img
bO, bX, bHO, bHX, bgap = (v[I].mean(1) for v in (O_img, X_img, HO_img, HX_img, gap_img))
out["T1"] = {"apparent_same_refs": float(A_img.mean()),
             "O_heldout": float(O_img.mean()), "ci_O": ci(bO), "X_heldout": float(X_img.mean()), "ci_X": ci(bX),
             "cider_surviving_share": float(X_img.mean() / O_img.mean()), "ci_cider_share": ci(bX / bO),
             "HO": float(HO_img.mean()), "ci_HO": ci(bHO), "HX": float(HX_img.mean()), "ci_HX": ci(bHX),
             "human_share": float(HX_img.mean() / HO_img.mean()), "ci_human_share": ci(bHX / bHO),
             "H_o_minus_H_of": float(gap_img.mean()), "ci_H_o_minus_H_of": ci(bgap),
             "cider_o_minus_of_heldout": float((O_img - X_img).mean()), "ci_cider_o_minus_of": ci(bO - bX),
             "human_score_scale": "total THumB score, roughly 1 to 5",
             "share_of_images_o_equals_of": float(np.mean([o == C4[f].argmax(1) for f in range(5)]))}

# T2
h = H.ravel()
out["T2"] = {"tau_C5": kendalltau(h, C5.ravel(), variant="b")[0],
             "tau_C4": kendalltau(h, C4.mean(0).ravel(), variant="b")[0],
             "tau_C1": kendalltau(h, C1.mean(0).ravel(), variant="b")[0],
             "tau_P_C5": kendalltau(P.ravel(), C5.ravel(), variant="b")[0]}

# T3
cp = os.path.join(HERE, "thumb_chair.json")
if os.path.exists(cp):
    ch = json.load(open(cp))["systems"]; systems = [str(s) for s in d["systems"]]; imgs = [str(x) for x in d["images"]]
    flag = np.array([[ch[s]["per_image"][im]["hall_sent"] for s in systems] for im in imgs], float).ravel()
    p = P.ravel(); res = {}
    for name, bad in (("P_le_3", p <= 3), ("P_lt_5", p < 5)):
        tp = float(((flag == 1) & bad).sum())
        res[name] = {"precision": tp / max(1, flag.sum()), "recall": tp / max(1, bad.sum()), "base_rate": float(bad.mean())}
    res["mean_P_flagged"] = float(p[flag == 1].mean()); res["mean_P_unflagged"] = float(p[flag == 0].mean())
    res["n_flagged"] = int(flag.sum()); res["n_captions"] = int(flag.size)
    res["CHAIRs_by_system"] = {s: 100 * ch[s]["CHAIR_s"] for s in ch}
    out["T3"] = res
json.dump(out, open(os.path.join(HERE, "thumb_results.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
