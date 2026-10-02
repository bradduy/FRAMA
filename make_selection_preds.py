"""Write cocoeval-ready prediction files for each training-free selection rule.

Reads the local beam-20 dumps and Stage-4 feature files, applies each rule,
and writes {image_name, captions, prediction} records under preds/<bench>/.
Oracle@20 is selected with the approximate CIDEr from pilot_headroom.py, so its
official score is a lower bound on the true in-beam oracle.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pilot_headroom import CiderD, tok, borda, z, ROOT  # noqa: E402

OUT = os.path.join(HERE, "preds")


def rules(feats, M):
    LM, BG, BL, LEN = feats[..., 0], feats[..., 1], feats[..., 2], feats[..., 4]
    N = LM.shape[0]
    return {
        "lm_top1": np.zeros(N, dtype=int),
        "bigG_top1": BG.argmax(1),
        "blip_top1": BL.argmax(1),
        "zmix_lm_bigG_a0.48": (0.48 * z(LM) + 0.52 * z(BG)).argmax(1),
        "zsum3_lm_bigG_blip": (z(LM) + z(BG) + z(BL)).argmax(1),
        "borda3_lm_bigG_blip": borda([LM, BG, BL]).argmax(1),
        "borda4_with_length": borda([LM, BG, BL, LEN]).argmax(1),
        "oracle20_approx": M.argmax(1),
    }


def build(bench, beam_path, feat_path, splits):
    beams = json.load(open(beam_path))
    f = np.load(feat_path, allow_pickle=True)
    feats = f["feats"].astype(np.float64)
    fnames = list(f["image_names"])
    pos = {str(n): i for i, n in enumerate(fnames)}
    order = np.array([pos[b["image_name"]] for b in beams])
    feats = feats[order]
    fc = f["captions"][order]
    N = len(beams)
    K = feats.shape[1]
    cands = [[x["caption"] for x in b["beam"]][:K] for b in beams]
    mism = sum(str(fc[i][j]).strip() != cands[i][j].strip() for i in range(N) for j in range(K))
    assert mism == 0, f"{bench}: {mism} caption mismatches between features and beam dump"
    refs_tok = [[tok(c) for c in b["captions"]] for b in beams]
    cider = CiderD(refs_tok)
    M = np.array([[cider.score_one(i, tok(cands[i][j])) for j in range(K)] for i in range(N)])
    picks = rules(feats, M)
    os.makedirs(os.path.join(OUT, bench), exist_ok=True)
    for name, idx in picks.items():
        recs = [{"image_name": b["image_name"], "captions": b["captions"],
                 "prediction": cands[i][int(idx[i])]} for i, b in enumerate(beams)]
        for sp in splits:
            sel = recs if sp == "overall" else [r for r, b in zip(recs, beams) if b.get("split") == sp]
            path = os.path.join(OUT, bench, f"{name}__{sp}.json")
            json.dump(sel, open(path, "w"))
            print(f"{bench:7s} {name:22s} {sp:11s} n={len(sel):5d} approxCIDEr={M[[i for i, b in enumerate(beams) if sp == 'overall' or b.get('split') == sp], :].max(1).mean() if name == 'oracle20_approx' else np.mean([M[i, int(idx[i])] for i, b in enumerate(beams) if sp == 'overall' or b.get('split') == sp]):.3f}")


if __name__ == "__main__":
    build("coco",
          f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_biggFull/coco_beam_dump.json",
          f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_csfn/features.npz",
          ["overall"])
    build("nocaps",
          f"{ROOT}/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json",
          f"{ROOT}/inference_result/nocaps-bigg-blip/features.npz",
          ["overall", "out_domain"])
