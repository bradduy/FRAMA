"""Official per-image CIDEr-D for several COCO prediction files (same references, corpus IDF
from those references). Run on GCP2 from IFCap/evaluation with rap_venv's python.
Usage: python per_image_cider_remote.py --preds_dir <preds/coco> --rules lm_top1 borda3_lm_bigG_blip zsum3_lm_bigG_blip stack6_all_equal stack5_noLM_blip_bigG_vqa_mbr_ground --out per_image_cider_coco.json"""
import argparse, json, os, sys
sys.path.insert(0, os.path.expanduser("~/ZeroShotImageCap/IFCap/evaluation"))
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
ap = argparse.ArgumentParser(); ap.add_argument("--preds_dir", required=True); ap.add_argument("--rules", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = PTBTokenizer(); out = {}
for r in a.rules:
    recs = json.load(open(os.path.join(a.preds_dir, f"{r}__overall.json")))
    gts = {i: [{"caption": c} for c in x["captions"]] for i, x in enumerate(recs)}
    res = {i: [{"caption": x["prediction"]}] for i, x in enumerate(recs)}
    gts_t = tok.tokenize(gts); res_t = tok.tokenize(res)
    score, scores = Cider().compute_score(gts_t, res_t)
    out[r] = {x["image_name"]: {"cider": float(s), "caption": x["prediction"]} for x, s in zip(recs, scores)}
    print(r, "corpus", round(100*score, 2), flush=True)
refs = {x["image_name"]: x["captions"] for x in recs}
out["_refs"] = refs
json.dump(out, open(a.out, "w"))
print("wrote", a.out)
