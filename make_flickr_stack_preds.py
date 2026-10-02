"""Flickr30k Karpathy test: six-agent stack prediction files from the stored beam-20
dump plus the agent outputs produced on GCP2 (features.npz with LM/bigG/BLIP,
vqascore_flickr.jsonl, ground_flickr.jsonl). Beam-MBR is computed here.
Writes preds/flickr/<rule>__overall.json and prints approximate CIDEr per rule."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import CiderD, tok, z, ROOT
from pilot_mbr_consensus import cider_pair
import analyze_grounding as AG

HERE = os.path.dirname(os.path.abspath(__file__))
dump = f"{ROOT}/inference_result/flickr-indomain-bigg-blip/flickr30k_beam_dump.json"
feat = f"{ROOT}/inference_result/flickr-indomain-bigg-blip/features.npz"
beams = json.load(open(dump)); N = len(beams); K = min(len(b["beam"]) for b in beams)
f = np.load(feat, allow_pickle=True)
pos = {str(n): i for i, n in enumerate(f["image_names"])}
order = [pos[b["image_name"]] for b in beams]
feats = f["feats"].astype(np.float64)[order][:, :K]
fc = f["captions"][order]
cands = [[x["caption"] for x in b["beam"]][:K] for b in beams]
assert all(str(fc[i][j]).strip() == cands[i][j].strip() for i in range(N) for j in range(K)), "feature/beam mismatch"
LM, BG, BL = feats[..., 0], feats[..., 1], feats[..., 2]
AG.PATHS["flickr"] = dump
G, miss = AG.grounding_features("flickr", beams, K)
V = AG.load_vqascore("flickr", beams, K)
assert miss == 0 and V is not None and not np.isnan(V).any(), "flickr agent files incomplete"
cands_tok = [[tok(c) for c in row] for row in cands]
cider = CiderD([[tok(c) for c in b["captions"]] for b in beams])
M = np.array([[cider.score_one(i, cands_tok[i][j]) for j in range(K)] for i in range(N)])
S = np.array([[np.mean([cider_pair(cider, cands_tok[i][j], cands_tok[i][l]) for l in range(K) if l != j]) for j in range(K)] for i in range(N)])
np.save(os.path.join(HERE, "mbr_self_flickr.npy"), S)
base3 = z(LM) + z(BG) + z(BL)
rules = {
    "lm_top1": LM, "blip_top1": BL, "zsum3_lm_bigG_blip": base3,
    "stack4_zsum3_vqa": base3 + z(V), "stack4_zsum3_mbr": base3 + z(S),
    "stack5_zsum3_vqa_mbr": base3 + z(V) + z(S),
    "stack6_all_equal": base3 + z(V) + z(S) + z(G[..., 0]),
    "stack5_noLM_blip_bigG_vqa_mbr_ground": z(BG) + z(BL) + z(V) + z(S) + z(G[..., 0]),
    "stack4_noLM_blip_bigG_vqa_mbr": z(BG) + z(BL) + z(V) + z(S),
    "oracle20_approx": M,
}
os.makedirs(os.path.join(HERE, "preds", "flickr"), exist_ok=True)
idx = np.arange(N)
for name, s in rules.items():
    pick = s.argmax(1)
    recs = [{"image_name": b["image_name"], "captions": b["captions"], "prediction": cands[i][int(pick[i])]} for i, b in enumerate(beams)]
    json.dump(recs, open(os.path.join(HERE, "preds", "flickr", f"{name}__overall.json"), "w"))
    print(f"flickr  {name:40s} approxCIDEr={M[idx, pick].mean():.4f}")
print(f"within-beam rho: " + "  ".join(f"{nm} {np.nanmean([__import__('scipy.stats').stats.spearmanr(s[i], M[i]).correlation for i in range(N) if np.std(M[i])>0 and np.std(s[i])>0]):.3f}" for nm, s in (("LM", LM), ("bigG", BG), ("BLIP", BL), ("VQA", V), ("MBR", S), ("ground", G[..., 0]))))
