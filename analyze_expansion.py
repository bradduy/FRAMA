"""C4 analysis on the wider-beam dumps (CPU only, no rescoring of new candidates):
Oracle@N and the cross-reference (attainable) oracle as N grows from 20 to 80,
plus the reference-blind picks that need no GPU scorer: LM top-1 and beam-MBR.
Usage: python analyze_expansion.py coco_beam40_dump.json coco_beam80_dump.json"""
import json, os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import CiderD, tok, ngrams, ROOT
from pilot_crossref_oracle import score_vs
from pilot_mbr_consensus import cider_pair

HERE = os.path.dirname(os.path.abspath(__file__))
BASE20 = f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_biggFull/coco_beam_dump.json"

def analyse(path, label, folds=5, mbr=True):
    beams = json.load(open(path))
    N = len(beams); K = min(len(b["beam"]) for b in beams)
    cands = [[tok(x["caption"]) for x in b["beam"]][:K] for b in beams]
    refs = [[tok(c) for c in b["captions"][:5]] for b in beams]
    cider = CiderD(refs)
    M = np.array([[cider.score_one(i, c) for c in cands[i]] for i in range(N)])
    print(f"\n===== {label}: N={N} K={K} =====")
    for n in (1, 5, 10, 20, 40, 80):
        if n <= K: print(f"Oracle@{n} (5-ref) ~{M[:, :n].max(1).mean():.4f}")
    # cross-reference oracle at K and at 20, held-out-1 scale
    o4K, o420, lm1 = [], [], []
    for i in range(N):
        if len(refs[i]) < 5: continue
        for h in range(folds):
            held = [refs[i][h]]; four = [r for j, r in enumerate(refs[i]) if j != h]
            sc4 = [score_vs(cider, c, four) for c in cands[i]]
            o4K.append(score_vs(cider, cands[i][int(np.argmax(sc4))], held))
            o420.append(score_vs(cider, cands[i][int(np.argmax(sc4[:20]))], held))
            lm1.append(score_vs(cider, cands[i][0], held))
    print(f"cross-ref oracle@20 ~{np.mean(o420):.4f}  cross-ref oracle@{K} ~{np.mean(o4K):.4f}  LM top-1 ~{np.mean(lm1):.4f}  (held-out-1 scale)")
    if mbr:
        S = np.array([[np.mean([cider_pair(cider, cands[i][j], cands[i][l]) for l in range(K) if l != j]) for j in range(K)] for i in range(N)])
        print(f"beam-MBR pick over {K} ~{M[np.arange(N), S.argmax(1)].mean():.4f}   LM top-1 ~{M[:, 0].mean():.4f}   (5-ref)")
        np.save(os.path.join(HERE, f"mbr_self_{label}.npy"), S)
    return M

if __name__ == "__main__":
    analyse(BASE20, "coco_beam20", mbr=False)
    for p in sys.argv[1:]:
        analyse(p, os.path.basename(p).replace("_dump.json", ""))
