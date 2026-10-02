"""Combine every training-free agent on the VQA-scored subset and measure the
gain against the cross-reference (attainable) oracle on the same images."""
import json, os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import CiderD, tok, ngrams, z, ROOT
from pilot_c2_weighted_consensus import load
import analyze_grounding as AG
from pilot_crossref_oracle import score_vs

HERE = os.path.dirname(os.path.abspath(__file__))
for bench in ("coco", "nocaps"):
    feats, M, splits = load(bench)
    beams = json.load(open(AG.PATHS[bench]))
    N, K = M.shape
    G, _ = AG.grounding_features(bench, beams, K)
    V = AG.load_vqascore(bench, beams, K)
    S = np.load(os.path.join(HERE, f"mbr_self_{bench}.npy"))
    has = ~np.isnan(V).any(axis=1)
    idx = np.where(has)[0]
    LM, BG, BL = feats[..., 0], feats[..., 1], feats[..., 2]
    Vf = V.copy(); Vf[~has] = 0.0
    gm = G[..., 0]
    base3 = z(LM) + z(BG) + z(BL)
    rules = {
        "LM top-1": LM, "BLIP alone": BL, "zsum3": base3,
        "zsum3 + 0.5 VQA": base3 + 0.5 * z(Vf),
        "zsum3 + 1.0 VQA": base3 + 1.0 * z(Vf),
        "zsum3 + 0.5 MBR": base3 + 0.5 * z(S),
        "zsum3 + 0.5 VQA + 0.5 MBR": base3 + 0.5 * z(Vf) + 0.5 * z(S),
        "zsum3 + 1.0 VQA + 0.5 MBR": base3 + 1.0 * z(Vf) + 0.5 * z(S),
        "zsum3 + 1.0 VQA + 0.5 MBR + 0.5 ground": base3 + 1.0 * z(Vf) + 0.5 * z(S) + 0.5 * z(gm),
        "zsum3 + 0.5 VQA + 0.5 MBR + 0.5 ground": base3 + 0.5 * z(Vf) + 0.5 * z(S) + 0.5 * z(gm),
        "BLIP + bigG + VQA + MBR (no LM)": z(BL) + z(BG) + z(Vf) + z(S),
        "BLIP + VQA + MBR + ground (no LM, no bigG)": z(BL) + z(Vf) + z(S) + 0.5 * z(gm),
        "all six equal": z(LM) + z(BG) + z(BL) + z(Vf) + z(S) + z(gm),
    }
    cands = [[tok(x["caption"]) for x in b["beam"]][:K] for b in beams]
    refs = [[tok(c) for c in b["captions"][:5]] for b in beams]
    cider = CiderD([[tok(c) for c in b["captions"]] for b in beams])
    # cross-reference oracle and per-rule held-out scores on the subset
    def heldout_mean(pick):
        vals = []
        for i in idx:
            if len(refs[i]) < 5: continue
            for h in range(5):
                vals.append(score_vs(cider, cands[i][int(pick[i])], [refs[i][h]]))
        return np.mean(vals)
    o4 = []
    for i in idx:
        if len(refs[i]) < 5: continue
        for h in range(5):
            four = [r for j, r in enumerate(refs[i]) if j != h]
            sc4 = [score_vs(cider, c, four) for c in cands[i]]
            o4.append(score_vs(cider, cands[i][int(np.argmax(sc4))], [refs[i][h]]))
    o4 = np.mean(o4)
    print(f"\n===== {bench}: {len(idx)} VQA-scored images =====")
    print(f"{'rule':44s} {'5-ref CIDEr~':>13s} {'held-out-1 CIDEr~':>18s} {'share of attainable':>20s}")
    base_h = None
    for nm, s in rules.items():
        pick = s.argmax(1)
        c5 = M[idx, pick[idx]].mean(); h1 = heldout_mean(pick)
        if nm == "zsum3": base_h = h1
        line = f"{nm:44s} {c5:13.4f} {h1:18.4f}"
        if base_h is not None and nm != "zsum3":
            line += f" {100*(h1-base_h)/(o4-base_h):19.0f}%"
        print(line)
    print(f"{'cross-reference oracle (attainable ceiling)':44s} {'':13s} {o4:18.4f}")
    print(f"{'Oracle@20 on all 5 refs (optimistic)':44s} {M[idx].max(1).mean():13.4f}")
