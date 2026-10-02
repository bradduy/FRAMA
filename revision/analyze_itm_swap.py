"""Recompute the COCO rules with the cross-encoder agent swapped from blip-itm-base-coco
(fine-tuned on COCO train pairs) to blip-itm-base-flickr (never fine-tuned on COCO captions).
Writes preds/coco/itmflickr_<rule>__overall.json for official evaluation."""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from pilot_headroom import z
from pilot_c2_weighted_consensus import load
import analyze_grounding as AG
feats, M, splits = load("coco"); beams = json.load(open(AG.PATHS["coco"])); N, K = M.shape
G, _ = AG.grounding_features("coco", beams, K); V = AG.load_vqascore("coco", beams, K)
S = np.load(os.path.join(HERE, "mbr_self_coco.npy"))
itm = {}
for l in open(os.path.join(HERE, "revision", "itm_flickr_coco.jsonl")):
    r = json.loads(l); itm[r["image_name"]] = r["itm"][:K]
F = np.array([itm[b["image_name"]] for b in beams], dtype=float)
LM, BG, BL, GR = feats[:, :K, 0], feats[:, :K, 1], feats[:, :K, 2], G[:, :, 0]
def rho(A): return np.nanmean([spearmanr(A[i], M[i]).correlation for i in range(N) if np.std(M[i]) > 0 and np.std(A[i]) > 0])
print(f"within-beam rho with CIDEr: BLIP-coco {rho(BL):.3f}  BLIP-flickr {rho(F):.3f}  rho(coco,flickr) "
      f"{np.nanmean([spearmanr(BL[i], F[i]).correlation for i in range(N)]):.3f}")
rules = {"blip_top1": z(F), "zsum3_lm_bigG_blip": z(LM) + z(BG) + z(F),
         "stack6_all_equal": z(LM) + z(BG) + z(F) + z(V) + z(S) + z(GR),
         "stack5_noLM_blip_bigG_vqa_mbr_ground": z(BG) + z(F) + z(V) + z(S) + z(GR)}
for name, sc in rules.items():
    pick = sc.argmax(1)
    print(f"{name:40s} approx CIDEr {100*M[np.arange(N), pick].mean():.1f}")
    recs = [{"image_name": b["image_name"], "captions": b["captions"], "prediction": b["beam"][int(pick[i])]["caption"]} for i, b in enumerate(beams)]
    os.makedirs(os.path.join(HERE, "preds", "coco"), exist_ok=True)
    json.dump(recs, open(os.path.join(HERE, "preds", "coco", f"itmflickr_{name}__overall.json"), "w"))
