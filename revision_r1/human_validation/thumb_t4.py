"""Analysis T4 of thumb_prereg.md: FRAMA agents against human judgement on THumB 1.0.

Inputs (t4/, pulled from GCP2): thumb_align.npz (bigG and BLIP-ITM, csfn_extract_features.py), thumb_vqa.jsonl
(System One verifier), thumb_ground.jsonl (detector), thumb_mbr.npy (consensus); and thumb_cider.npz.
For every selector we report the mean human total score, Precision and Recall of the caption it picks, its
paired difference to random choice (the mean over the four captions) and to each single dual-encoder or
cross-encoder agent, and its pairwise agreement with the human ordering: over caption pairs of an image with
different human scores, the share ordered the same way by the selector, ties counted as one half.
Bootstrap: 10,000 paired resamples of images, seed 0, 95% percentile intervals. The LM agent does not apply,
because the captions were not produced by our captioner. Writes thumb_t4_results.json.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); T4 = os.path.join(HERE, "t4"); B = 10000
d = np.load(os.path.join(HERE, "thumb_cider.npz"), allow_pickle=True)
images = [str(x) for x in d["images"]]; systems = [str(s) for s in d["systems"]]
C5, C4, H, P, Rc = d["C5"], d["C4"], d["H"], d["P"], d["Rc"]
N, K = H.shape; ar = np.arange(N)
z = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def jsonl(path, key):
    recs = {}
    for line in open(path):
        r = json.loads(line); recs[r["image_name"]] = r
    return [recs[im] for im in images] if key is None else np.array([recs[im][key] for im in images], float)


al = np.load(os.path.join(T4, "thumb_align.npz"), allow_pickle=True)
assert [str(x) for x in al["image_names"]] == images
BG, BL = al["feats"][..., 1].astype(float), al["feats"][..., 2].astype(float)
S1 = jsonl(os.path.join(T4, "thumb_vqa.jsonl"), "vqascore")
MBR = np.load(os.path.join(T4, "thumb_mbr.npy"))
GR = np.full((N, K), np.nan)  # analyze_grounding.grounding_features, g_mean
for i, r in enumerate(jsonl(os.path.join(T4, "thumb_ground.jsonl"), None)):
    for j, c in enumerate(r["cands"][:K]):
        sc = [r["nouns"][n][0] for n in c["nouns"] if n in r["nouns"]]
        if sc:
            GR[i, j] = np.mean(sc)
    row = GR[i]; row[np.isnan(row)] = np.nanmean(row) if not np.isnan(row).all() else 0.0

agents = {"bigG": BG, "BLIP-ITM": BL, "System One verifier": S1, "consensus": MBR, "grounding": GR}
sel = {f"{a} alone": s for a, s in agents.items()}
sel["z-sum bigG + BLIP-ITM"] = z(BG) + z(BL)
sel["z-sum bigG + BLIP-ITM + System One"] = z(BG) + z(BL) + z(S1)
sel["five-agent FRAMA, equal weights"] = sum(z(s) for s in agents.values())
sel["CIDEr, five references (reference-selected)"] = C5
picks = {k: v.argmax(1) for k, v in sel.items()}
for j, s in enumerate(systems):
    picks[f"system {s}"] = np.full(N, j)

I = np.random.default_rng(0).integers(0, N, size=(B, N))
ci = lambda v: [float(x) for x in np.percentile(v[I].mean(1), [2.5, 97.5])]
rnd = H.mean(1)


def pair_agree(s):
    num = den = 0.0
    for j in range(K):
        for k in range(j + 1, K):
            m = H[:, j] != H[:, k]
            agree = np.sign(s[m, j] - s[m, k]) == np.sign(H[m, j] - H[m, k])
            tie = s[m, j] == s[m, k]
            num += agree.sum() + 0.5 * tie.sum(); den += m.sum()
    return num / den


out = {"n_images": N, "random_choice": {"H": float(rnd.mean()), "P": float(P.mean()), "R": float(Rc.mean())},
       "human_best_pick": {"H": float(H.max(1).mean())}, "selectors": {}}
# leakage-free CIDEr pick, averaged over the five folds
hx = np.mean([H[ar, C4[f].argmax(1)] for f in range(5)], 0)
out["selectors"]["CIDEr, four references (leakage-free), fold mean"] = {"H": float(hx.mean()), "ci_H": ci(hx),
                                                                        "H_minus_random": float((hx - rnd).mean()), "ci_H_minus_random": ci(hx - rnd)}
for k, p in picks.items():
    h = H[ar, p]
    r = {"H": float(h.mean()), "ci_H": ci(h), "P": float(P[ar, p].mean()), "R": float(Rc[ar, p].mean()),
         "H_minus_random": float((h - rnd).mean()), "ci_H_minus_random": ci(h - rnd)}
    if k in sel:
        r["pairwise_agreement"] = float(pair_agree(sel[k]))
    for a in ("bigG", "BLIP-ITM"):
        if k != f"{a} alone" and k in sel:
            ha = H[ar, picks[f"{a} alone"]]
            r[f"H_minus_{a}"] = float((h - ha).mean()); r[f"ci_H_minus_{a}"] = ci(h - ha)
    out["selectors"][k] = r
out["within_image_spearman_between_agents"] = {}
from scipy.stats import spearmanr
names = list(agents)
for x in range(len(names)):
    for y in range(x + 1, len(names)):
        a, b = agents[names[x]], agents[names[y]]
        v = [spearmanr(a[i], b[i]).correlation for i in range(N) if np.std(a[i]) > 0 and np.std(b[i]) > 0]
        out["within_image_spearman_between_agents"][f"{names[x]} vs {names[y]}"] = float(np.nanmean(v))
json.dump(out, open(os.path.join(HERE, "thumb_t4_results.json"), "w"), indent=1)
print(f"random H {rnd.mean():.3f}  human-best H {H.max(1).mean():.3f}")
for k, r in out["selectors"].items():
    extra = f"  agree {r['pairwise_agreement']:.3f}" if "pairwise_agreement" in r else ""
    vs = "".join(f"  vs {a} {r[f'H_minus_{a}']:+.3f} [{r[f'ci_H_minus_{a}'][0]:+.3f}, {r[f'ci_H_minus_{a}'][1]:+.3f}]"
                 for a in ("bigG", "BLIP-ITM") if f"H_minus_{a}" in r)
    print(f"{k:52s} H {r['H']:.3f}  vs random {r['H_minus_random']:+.3f} "
          f"[{r['ci_H_minus_random'][0]:+.3f}, {r['ci_H_minus_random'][1]:+.3f}]{extra}{vs}")
