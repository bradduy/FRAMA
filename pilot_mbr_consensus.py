"""F9 pilot: training-free consensus agents that target reference-style phrasing.

mbr_self : for each candidate, mean CIDEr-D against the other 19 beam candidates
           (minimum-Bayes-risk over the captioner's own beam; no image, no refs).
mbr_retr : mean CIDEr-D against the retrieved captions of the image, if a
           retrieval file is available (pseudo-references from the text corpus).
Both are regime-clean (text only at inference) and cost nothing on GPU.
Metric for evaluation: approximate CIDEr-D vs references (pilot_headroom.CiderD).
"""
import json, os, sys, math
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import CiderD, tok, ngrams, z, ROOT
from pilot_c2_weighted_consensus import load

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = {"coco": f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_biggFull/coco_beam_dump.json",
         "nocaps": f"{ROOT}/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json"}

def cider_pair(cider, a_tok, b_tok):
    """CIDEr-D of hypothesis a against single pseudo-reference b, corpus IDF from cider."""
    ha, na = cider.vec(ngrams(a_tok, cider.n), len(a_tok))
    hb, nb = cider.vec(ngrams(b_tok, cider.n), len(b_tok))
    delta = float(len(a_tok) - len(b_tok)); tot = 0.0
    for k in range(cider.n):
        s = sum(min(v, hb[k].get(g, 0.0)) * hb[k].get(g, 0.0) for g, v in ha[k].items())
        if na[k] and nb[k]: s /= (na[k] * nb[k])
        tot += s * math.e ** (-(delta ** 2) / (2 * cider.sigma ** 2))
    return tot / cider.n * 10.0

for bench in ("coco", "nocaps"):
    feats, M, splits = load(bench)
    beams = json.load(open(PATHS[bench]))
    N, K = M.shape
    cands_tok = [[tok(x["caption"]) for x in b["beam"]][:K] for b in beams]
    cider = CiderD([[tok(c) for c in b["captions"]] for b in beams])
    S = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            S[i, j] = np.mean([cider_pair(cider, cands_tok[i][j], cands_tok[i][l]) for l in range(K) if l != j])
    LM, BG, BL = feats[..., 0], feats[..., 1], feats[..., 2]
    idx = np.arange(N)
    def cid(pick, mask=None):
        return M[idx, pick].mean() if mask is None else M[mask][np.arange(mask.sum()), pick[mask]].mean()
    rho = lambda s: np.nanmean([spearmanr(s[i], M[i]).correlation for i in range(N) if np.std(M[i]) > 0 and np.std(s[i]) > 0])
    print(f"\n===== {bench} =====")
    print(f"within-beam Spearman with CIDEr: mbr_self {rho(S):.3f} | LM {rho(LM):.3f} bigG {rho(BG):.3f} BLIP {rho(BL):.3f}")
    print(f"mbr_self vs BLIP {np.nanmean([spearmanr(S[i], BL[i]).correlation for i in range(N)]):.3f}  vs LM {np.nanmean([spearmanr(S[i], LM[i]).correlation for i in range(N)]):.3f}")
    base3 = z(LM) + z(BG) + z(BL)
    out_mask = None if bench == "coco" else (splits == "out_domain")
    rows = [("LM top-1", LM), ("BLIP alone", BL), ("zsum3", base3), ("mbr_self alone", S)]
    for w in (0.5, 1.0, 2.0, 3.0):
        rows.append((f"zsum3 + {w}*z(mbr_self)", base3 + w * z(S)))
    for w in (0.5, 1.0, 2.0):
        rows.append((f"z(BLIP) + {w}*z(mbr_self)", z(BL) + w * z(S)))
    rows.append(("z(BLIP)+z(bigG)+z(mbr_self)", z(BL) + z(BG) + z(S)))
    for nm, s in rows:
        pick = s.argmax(1); line = f"{nm:32s} CIDEr~{cid(pick):.4f}"
        if out_mask is not None: line += f"  out~{cid(pick, out_mask):.4f}"
        print(line)
    print(f"Oracle@20 ~{M.max(1).mean():.4f}")
    np.save(os.path.join(HERE, f"mbr_self_{bench}.npy"), S)
