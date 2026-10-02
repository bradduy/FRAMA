"""C3: cocoeval-ready prediction files for verifier-ladder variants on the FULL
sets, once scores_<scorer>_<bench>.jsonl cover every image. Configurations are
fixed a priori (equal weights); nothing is tuned on references."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import z
from pilot_c2_weighted_consensus import load
import analyze_grounding as AG
from analyze_ladder import load_scores

HERE = os.path.dirname(os.path.abspath(__file__))
for bench in ("coco", "nocaps"):
    feats, M, splits = load(bench)
    beams = json.load(open(AG.PATHS[bench]))
    N, K = M.shape
    LM, BG, BL = feats[..., 0], feats[..., 1], feats[..., 2]
    G, _ = AG.grounding_features(bench, beams, K)
    V = np.nan_to_num(AG.load_vqascore(bench, beams, K))
    S = np.load(os.path.join(HERE, f"mbr_self_{bench}.npy"))
    sc = {}
    for name in ("siglip2", "blip2itm"):
        p = os.path.join(HERE, f"scores_{name}_{bench}.jsonl")
        if os.path.exists(p):
            X = load_scores(p, beams, K)
            if not np.isnan(X).any():
                sc[name] = X
            else:
                print(f"{bench}: {name} incomplete ({(~np.isnan(X).any(1)).sum()}/{N}); skipped")
    if not sc:
        print(f"{bench}: no complete ladder scores yet"); continue
    base3 = z(LM) + z(BG) + z(BL)
    stack6 = base3 + z(V) + z(S) + z(G[..., 0])
    rules = {}
    if "siglip2" in sc:
        rules["ladder_zsum3_bigG2siglip2"] = z(LM) + z(sc["siglip2"]) + z(BL)
        rules["ladder_zsum3_plus_siglip2"] = base3 + z(sc["siglip2"])
        rules["ladder_stack6_bigG2siglip2"] = z(LM) + z(sc["siglip2"]) + z(BL) + z(V) + z(S) + z(G[..., 0])
    if "blip2itm" in sc:
        rules["ladder_zsum3_plus_blip2itm"] = base3 + z(sc["blip2itm"])
        rules["ladder_stack6_plus_blip2itm"] = stack6 + z(sc["blip2itm"])
    if len(sc) == 2:
        rules["ladder_stack8_all"] = stack6 + z(sc["siglip2"]) + z(sc["blip2itm"])
    cands = [[x["caption"] for x in b["beam"]][:K] for b in beams]
    for name, s in rules.items():
        pick = s.argmax(1)
        recs = [{"image_name": b["image_name"], "captions": b["captions"], "prediction": cands[i][int(pick[i])]} for i, b in enumerate(beams)]
        for sp in (["overall"] if bench == "coco" else ["overall", "out_domain"]):
            sel = recs if sp == "overall" else [r for r, b in zip(recs, beams) if b.get("split") == sp]
            json.dump(sel, open(os.path.join(HERE, "preds", bench, f"{name}__{sp}.json"), "w"))
        print(f"{bench:7s} {name:32s} approxCIDEr={M[np.arange(N), pick].mean():.4f}")
