"""C5 full set: the fixed six-agent configurations on the ViECap beam-20 of all 5,000 COCO
Karpathy-test images. Inputs (pulled from GCP2 c5/): viecap_coco_full_beam_dump.json,
viecap_full_features.npz, vqascore_viecap_coco_full.jsonl, ground_viecap_coco_full.jsonl.
Outputs: preds/coco_viecap_full/<rule>__overall.json; IFCap rows for the paired comparison
are the existing full-set files in preds/coco/."""
import json, os, sys, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import CiderD, tok, z, ROOT
from pilot_mbr_consensus import cider_pair
import analyze_grounding as AG
HERE = os.path.dirname(os.path.abspath(__file__)); C5 = os.path.join(HERE, "c5")
dump = os.path.join(C5, "viecap_coco_full_beam_dump.json")
beams = json.load(open(dump)); N = len(beams); K = min(len(b["beam"]) for b in beams)
f = np.load(os.path.join(C5, "viecap_full_features.npz"), allow_pickle=True)
pos = {str(n): i for i, n in enumerate(f["image_names"])}; order = [pos[b["image_name"]] for b in beams]
feats = f["feats"].astype(np.float64)[order][:, :K]
cands = [[x["caption"] for x in b["beam"]][:K] for b in beams]
fc = f["captions"][order]
assert all(str(fc[i][j]).strip() == cands[i][j].strip() for i in range(N) for j in range(K)), "feature/beam mismatch"
AG.PATHS["viecapfull"] = dump
shutil.copyfile(os.path.join(C5, "vqascore_viecap_coco_full.jsonl"), os.path.join(HERE, "vqascore_viecapfull.jsonl"))
shutil.copyfile(os.path.join(C5, "ground_viecap_coco_full.jsonl"), os.path.join(HERE, "ground_viecapfull.jsonl"))
Gf, miss = AG.grounding_features("viecapfull", beams, K)
V = AG.load_vqascore("viecapfull", beams, K)
assert miss == 0 and V is not None and not np.isnan(V).any(), "agent files incomplete"
G = Gf[..., 0]; LM, BG, BL = feats[..., 0], feats[..., 1], feats[..., 2]
cands_tok = [[tok(c) for c in row] for row in cands]
cider = CiderD([[tok(c) for c in b["captions"]] for b in beams])
M = np.array([[cider.score_one(i, cands_tok[i][j]) for j in range(K)] for i in range(N)])
S = np.array([[np.mean([cider_pair(cider, cands_tok[i][j], cands_tok[i][l]) for l in range(K) if l != j]) for j in range(K)] for i in range(N)])
np.save(os.path.join(C5, "mbr_self_viecapfull.npy"), S)
base3 = z(LM) + z(BG) + z(BL)
rules = {"lm_top1": LM, "blip_top1": BL, "zsum3_lm_bigG_blip": base3, "stack4_zsum3_vqa": base3 + z(V), "stack4_zsum3_mbr": base3 + z(S),
         "stack6_all_equal": base3 + z(V) + z(S) + z(G), "stack5_noLM_blip_bigG_vqa_mbr_ground": z(BG) + z(BL) + z(V) + z(S) + z(G), "oracle20_approx": M}
out = os.path.join(HERE, "preds", "coco_viecap_full"); os.makedirs(out, exist_ok=True)
idx = np.arange(N); print(f"ViECap full: N={N} K={K}")
for name, s in rules.items():
    pick = s.argmax(1)
    json.dump([{"image_name": b["image_name"], "captions": b["captions"], "prediction": cands[i][int(pick[i])]} for i, b in enumerate(beams)], open(os.path.join(out, f"{name}__overall.json"), "w"))
    print(f"viecap-full  {name:40s} approxCIDEr={M[idx, pick].mean():.4f}")
from scipy.stats import spearmanr
print("within-beam rho: " + "  ".join(f"{nm} {np.nanmean([spearmanr(s[i], M[i]).correlation for i in range(N) if np.std(M[i])>0 and np.std(s[i])>0]):.3f}" for nm, s in (("LM", LM), ("bigG", BG), ("BLIP", BL), ("VQA", V), ("MBR", S), ("ground", G))))
