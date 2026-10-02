"""Reviewer item M4: add the BLIP-2 caption as a 21st candidate and re-run the five-agent selector
(bigG, BLIP-ITM, System One verifier, beam consensus, grounding; no decoder likelihood, which is
undefined for a caption the decoder did not produce). Consensus is recomputed over 21 candidates.
Writes preds/<bench>/m4_<rule>__<split>.json for official evaluation."""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from pilot_headroom import z, CiderD, tok
from pilot_mbr_consensus import cider_pair
from pilot_c2_weighted_consensus import load
import analyze_grounding as AG
bench = sys.argv[1]; R = os.path.join(HERE, "revision")
feats, M, _ = load(bench); beams = json.load(open(AG.PATHS[bench])); N, K = M.shape
G, _ = AG.grounding_features(bench, beams, K); V = AG.load_vqascore(bench, beams, K)
names = [b["image_name"] for b in beams]
f = np.load(os.path.join(R, f"m4_{bench}_features.npz"), allow_pickle=True)
pos = {str(n): i for i, n in enumerate(f["image_names"])}; F = f["feats"][[pos[n] for n in names]][:, 0, :]
vq = {}; [vq.__setitem__(json.loads(l)["image_name"], json.loads(l)["vqascore"][0]) for l in open(os.path.join(R, f"m4_{bench}_vqa.jsonl"))]
gr = {}
for l in open(os.path.join(R, f"m4_{bench}_ground.jsonl")):
    r = json.loads(l); c = r["cands"][0]; sc = [r["nouns"][n][0] for n in c["nouns"] if n in r["nouns"]]
    gr[r["image_name"]] = float(np.mean(sc)) if sc else np.nan
new = [x["prediction"] for x in sorted(json.load(open(os.path.join(HERE, "preds", bench, "blip2cap__overall.json"))), key=lambda r: names.index(r["image_name"]))] if False else None
p2 = {r["image_name"]: r["prediction"] for r in json.load(open(os.path.join(HERE, "preds", bench, "blip2cap__overall.json")))}
cands21 = [[x["caption"] for x in b["beam"]][:K] + [p2[b["image_name"]]] for b in beams]
cider = CiderD([[tok(c) for c in b["captions"]] for b in beams])
ct = [[tok(c) for c in cs] for cs in cands21]
M21 = np.array([[cider.score_one(i, ct[i][j]) for j in range(K + 1)] for i in range(N)])
S21 = np.array([[np.mean([cider_pair(cider, ct[i][j], ct[i][l]) for l in range(K + 1) if l != j]) for j in range(K + 1)] for i in range(N)])
BG = np.c_[feats[:, :K, 1], F[:, 1]]; BL = np.c_[feats[:, :K, 2], F[:, 2]]
VQ = np.c_[V, [vq[n] for n in names]]
g_new = np.array([gr[n] for n in names]); GR = np.c_[G[:, :, 0], g_new]
GR[np.isnan(GR[:, K]), K] = GR[np.isnan(GR[:, K]), :K].mean(1)
five21 = z(BG) + z(BL) + z(VQ) + z(S21) + z(GR)
five20 = z(BG[:, :K]) + z(BL[:, :K]) + z(VQ[:, :K]) + z(S21[:, :K]) + z(GR[:, :K])
pick = five21.argmax(1); chosen_new = float(np.mean(pick == K))
ap = lambda sc, MM: 100 * MM[np.arange(N), sc.argmax(1)].mean()
out = {"bench": bench, "n": N, "five20_approx": ap(five20, M21[:, :K]), "five21_approx": ap(five21, M21), "blip2_alone_approx": 100 * M21[:, K].mean(),
       "oracle21_approx": 100 * M21.max(1).mean(), "oracle20_approx": 100 * M21[:, :K].max(1).mean(),
       "frac_blip2_picked": chosen_new, "frac_blip2_is_oracle": float(np.mean(M21.argmax(1) == K)),
       "blip2_better_than_five20": float(np.mean(M21[:, K] > M21[np.arange(N), five20.argmax(1)]))}
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(R, f"m4_{bench}_summary.json"), "w"), indent=1)
splits = {"overall": list(range(N))}
for i, b in enumerate(beams):
    if b.get("split"): splits.setdefault(b["split"], []).append(i)
simple = {"blip21": z(BL), "bigG21": z(BG), "s1_21": z(VQ), "align2_21": z(BG) + z(BL), "align2s1_21": z(BG) + z(BL) + z(VQ), "four21_noMBR": z(BG) + z(BL) + z(VQ) + z(GR)}
print("MBR z of the BLIP-2 caption, mean over images:", float(z(S21)[:, K].mean()), " grounding z:", float(z(GR)[:, K].mean()))
for r, sc in simple.items():
    out[r + "_approx"] = ap(sc, M21); out[r + "_frac_blip2"] = float(np.mean(sc.argmax(1) == K))
    print(f"{r:14s} approx {out[r + '_approx']:.2f}  picks BLIP-2 on {100*out[r + '_frac_blip2']:.1f}%")
json.dump(out, open(os.path.join(R, f"m4_{bench}_summary.json"), "w"), indent=1)
for rule, sc in [("five21", five21), ("five20", five20)] + [(k, v) for k, v in simple.items()]:
    pk = sc.argmax(1)
    for sp, ids in splits.items():
        recs = [{"image_name": names[i], "captions": beams[i]["captions"], "prediction": cands21[i][int(pk[i])]} for i in ids]
        json.dump(recs, open(os.path.join(HERE, "preds", bench, f"m4_{rule}__{sp}.json"), "w"))
print("wrote predictions for", list(splits))
