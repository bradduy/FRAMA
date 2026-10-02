"""Write cocoeval-ready prediction files for the multi-agent stacks on the FULL
sets (requires vqascore_<bench>.jsonl to cover every image). Weights are fixed
a priori (equal weights), never tuned on references."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import z
from pilot_c2_weighted_consensus import load
import analyze_grounding as AG

HERE = os.path.dirname(os.path.abspath(__file__))
for bench in ("coco", "nocaps"):
    feats, M, splits = load(bench)
    beams = json.load(open(AG.PATHS[bench]))
    N, K = M.shape
    G, _ = AG.grounding_features(bench, beams, K)
    V = AG.load_vqascore(bench, beams, K)
    has = ~np.isnan(V).any(axis=1)
    if has.sum() < N:
        print(f"{bench}: VQA covers {has.sum()}/{N} images; full-set stack preds NOT written"); continue
    S = np.load(os.path.join(HERE, f"mbr_self_{bench}.npy"))
    LM, BG, BL = feats[..., 0], feats[..., 1], feats[..., 2]
    gm = G[..., 0]
    base3 = z(LM) + z(BG) + z(BL)
    rules = {
        "stack4_zsum3_vqa": base3 + z(V),
        "stack4_zsum3_mbr": base3 + z(S),
        "stack5_zsum3_vqa_mbr": base3 + z(V) + z(S),
        "stack6_all_equal": base3 + z(V) + z(S) + z(gm),
        "stack5_noLM_blip_bigG_vqa_mbr_ground": z(BG) + z(BL) + z(V) + z(S) + z(gm),
        "stack4_noLM_blip_bigG_vqa_mbr": z(BG) + z(BL) + z(V) + z(S),
        "blip_vqa_mbr_ground": z(BL) + z(V) + z(S) + z(gm),
    }
    cands = [[x["caption"] for x in b["beam"]][:K] for b in beams]
    for name, s in rules.items():
        pick = s.argmax(1)
        recs = [{"image_name": b["image_name"], "captions": b["captions"], "prediction": cands[i][int(pick[i])]} for i, b in enumerate(beams)]
        for sp in (["overall"] if bench == "coco" else ["overall", "out_domain"]):
            sel = recs if sp == "overall" else [r for r, b in zip(recs, beams) if b.get("split") == sp]
            os.makedirs(os.path.join(HERE, "preds", bench), exist_ok=True)
            json.dump(sel, open(os.path.join(HERE, "preds", bench, f"{name}__{sp}.json"), "w"))
        print(f"{bench:7s} {name:40s} approxCIDEr={M[np.arange(N), pick].mean():.4f}")
