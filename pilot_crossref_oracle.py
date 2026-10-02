"""C1 pilot: how much of the Oracle@20 headroom generalises to unseen references?

For each image with 5 refs, hold one ref out (5 folds). Select a candidate on
the 4 remaining refs (oracle-on-4), evaluate against the held-out ref alone.
Compare with the reference-blind z-sum pick and with a human caption (the
held-out ref's siblings scored against it) on exactly the same held-out ref.
Scale: CIDEr-D against ONE reference (not comparable to the 5-ref numbers).
"""
import json, os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import CiderD, tok, ngrams, z, ROOT
from pilot_c2_weighted_consensus import load

PATHS = {"coco": f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_biggFull/coco_beam_dump.json",
         "nocaps": f"{ROOT}/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json"}

def score_vs(cider, hyp_tok, refs_tok):
    """CIDEr-D of hyp against an arbitrary ref list, using cider's corpus IDF."""
    hv, hn = cider.vec(ngrams(hyp_tok, cider.n), len(hyp_tok))
    tot = np.zeros(cider.n)
    for r in refs_tok:
        rv, rn = cider.vec(ngrams(r, cider.n), len(r))
        delta = float(len(hyp_tok) - len(r))
        for k in range(cider.n):
            s = sum(min(v, rv[k].get(g, 0.0)) * rv[k].get(g, 0.0) for g, v in hv[k].items())
            if hn[k] and rn[k]: s /= (hn[k] * rn[k])
            tot[k] += s * math.e ** (-(delta ** 2) / (2 * cider.sigma ** 2))
    return float(tot.mean() / len(refs_tok) * 10.0)

for bench in ("coco", "nocaps"):
    feats, M, splits = load(bench)
    beams = json.load(open(PATHS[bench]))
    keep = [i for i, b in enumerate(beams) if len(b["captions"]) >= 5]
    N, K = M.shape
    cands = [[tok(x["caption"]) for x in b["beam"]][:K] for b in beams]
    refs = [[tok(c) for c in b["captions"][:5]] for b in beams]
    cider = CiderD(refs)
    LM, BG, BL = feats[..., 0], feats[..., 1], feats[..., 2]
    pick_z = (z(LM) + z(BG) + z(BL)).argmax(1)
    pick_lm = np.zeros(N, dtype=int)
    res = {"zsum3 pick": [], "LM top-1": [], "oracle-on-4 pick": [], "oracle-on-5 pick (optimistic)": [], "human caption": [], "random candidate": []}
    rng = np.random.default_rng(0)
    for i in keep:
        for h in range(5):
            held = [refs[i][h]]; four = [r for j, r in enumerate(refs[i]) if j != h]
            sc4 = [score_vs(cider, c, four) for c in cands[i]]
            o4 = int(np.argmax(sc4))
            o5 = int(M[i].argmax())
            res["zsum3 pick"].append(score_vs(cider, cands[i][pick_z[i]], held))
            res["LM top-1"].append(score_vs(cider, cands[i][0], held))
            res["oracle-on-4 pick"].append(score_vs(cider, cands[i][o4], held))
            res["oracle-on-5 pick (optimistic)"].append(score_vs(cider, cands[i][o5], held))
            # human: another reference scored against the held-out one
            res["human caption"].append(np.mean([score_vs(cider, r, held) for r in four]))
            res["random candidate"].append(score_vs(cider, cands[i][rng.integers(K)], held))
    print(f"\n===== {bench}: CIDEr-D vs ONE held-out reference (mean over {len(keep)} images x 5 folds) =====")
    for k, v in res.items():
        print(f"{k:32s} {np.mean(v):.4f}")
    g_o4 = np.mean(res["oracle-on-4 pick"]) - np.mean(res["zsum3 pick"])
    g_o5 = np.mean(res["oracle-on-5 pick (optimistic)"]) - np.mean(res["zsum3 pick"])
    print(f"generalising share of oracle gain: {100*g_o4/g_o5:.0f}%  (oracle-on-4 gain {g_o4:.4f} / oracle-on-5 gain {g_o5:.4f})")
