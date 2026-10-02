"""C4: six-agent stack on a wider beam. Needs, for beam width BW:
  IFCap/inference_result/c4_expand/coco_beam<BW>_dump.json and features_beam<BW>.npz (LM, bigG, BLIP),
  vqascore_coco_beam<BW>.jsonl and ground_coco_beam<BW>.jsonl in this folder.
Reports approximate 5-ref CIDEr for each rule at N=BW and at N=20 (the first 20
candidates of the same beam, which reproduce the original beam-20 order), Oracle@N,
and writes cocoeval-ready prediction files for the stack picks at N=BW.
Usage: python analyze_c4_stack.py 40
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import CiderD, tok, z, ROOT
from pilot_mbr_consensus import cider_pair
import analyze_grounding as AG

HERE = os.path.dirname(os.path.abspath(__file__))
bw = sys.argv[1] if len(sys.argv) > 1 else "40"  # beam width tag, e.g. 40, 40s (1-in-5 subsample), 80s
tag = f"coco_beam{bw}"
dump = f"{ROOT}/inference_result/c4_expand/{tag}_dump.json"
feat = f"{ROOT}/inference_result/c4_expand/features_beam{bw}.npz"
beams = json.load(open(dump)); N = len(beams); K = min(len(b["beam"]) for b in beams)
f = np.load(feat, allow_pickle=True)
pos = {str(n): i for i, n in enumerate(f["image_names"])}
order = [pos[b["image_name"]] for b in beams]
feats = f["feats"].astype(np.float64)[order][:, :K]
fc = f["captions"][order]
cands = [[x["caption"] for x in b["beam"]][:K] for b in beams]
assert all(str(fc[i][j]).strip() == cands[i][j].strip() for i in range(N) for j in range(K)), "feature/beam mismatch"
LM, BG, BL = feats[..., 0], feats[..., 1], feats[..., 2]
G, missing_g = AG.grounding_features(tag, beams, K)
V = AG.load_vqascore(tag, beams, K)
assert missing_g == 0 and V is not None and not np.isnan(V).any(), "grounding or VQA incomplete"
cands_tok = [[tok(c) for c in row] for row in cands]
cider = CiderD([[tok(c) for c in b["captions"]] for b in beams])
M = np.array([[cider.score_one(i, cands_tok[i][j]) for j in range(K)] for i in range(N)])
S = np.array([[np.mean([cider_pair(cider, cands_tok[i][j], cands_tok[i][l]) for l in range(K) if l != j]) for j in range(K)] for i in range(N)])
np.save(os.path.join(HERE, f"mbr_self_{tag}.npy"), S)

def zn(x):  # per-image z over the first n columns is handled by slicing before z
    return z(x)

def rules(n):
    sl = slice(0, n)
    lm, bg, bl, v, s, g = LM[:, sl], BG[:, sl], BL[:, sl], V[:, sl], S[:, sl], G[:, sl, 0]
    base3 = zn(lm) + zn(bg) + zn(bl)
    return {
        "LM top-1": lm, "BLIP alone": bl, "zsum3": base3,
        "stack4 zsum3+VQA": base3 + zn(v),
        "stack5 zsum3+VQA+MBR": base3 + zn(v) + zn(s),
        "stack6 all equal": base3 + zn(v) + zn(s) + zn(g),
        "stack5 noLM": zn(bg) + zn(bl) + zn(v) + zn(s) + zn(g),
    }

print(f"===== {tag}: N={N} K={K} =====")
print(f"{'rule':28s} {'N=20':>10s} {'N=' + str(K):>10s}")
idx = np.arange(N)
for name in rules(K):
    r20 = rules(20)[name].argmax(1); rK = rules(K)[name].argmax(1)
    print(f"{name:28s} {M[idx, r20].mean():10.4f} {M[idx, rK].mean():10.4f}")
print(f"{'Oracle@N':28s} {M[:, :20].max(1).mean():10.4f} {M.max(1).mean():10.4f}")
# prediction files for the stack picks at N=K
os.makedirs(os.path.join(HERE, "preds", "coco"), exist_ok=True)
for name, key in (("stack6 all equal", f"c4_beam{bw}_stack6"), ("stack5 noLM", f"c4_beam{bw}_stack5_noLM"), ("zsum3", f"c4_beam{bw}_zsum3")):
    pick = rules(K)[name].argmax(1)
    recs = [{"image_name": b["image_name"], "captions": b["captions"], "prediction": cands[i][int(pick[i])]} for i, b in enumerate(beams)]
    json.dump(recs, open(os.path.join(HERE, "preds", "coco", f"{key}__overall.json"), "w"))
    print(f"wrote preds/coco/{key}__overall.json")
