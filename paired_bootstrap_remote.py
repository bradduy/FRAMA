"""Paired bootstrap over images for CIDEr-D differences between prediction files,
using the OFFICIAL tokenizer and scorer (IFCap evaluation/pycocoevalcap, Java 11).

Run on GCP2 from IFCap/evaluation with rap_venv:
  python ~/ZeroShotImageCap/macst_v1/v2_upgrade/paired_bootstrap_remote.py \
      --base preds/coco/zsum3_lm_bigG_blip__overall.json \
      --cand preds/coco/stack6_all_equal__overall.json [more cand files] \
      --out bootstrap_coco.json
Per-image CIDEr-D is computed once per file (corpus IDF from the references of
that file), then 10,000 paired resamples of images give the CI of the mean
difference and a one-sided p-value for cand > base.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/ZeroShotImageCap/IFCap/evaluation"))
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider


def per_image_cider(path):
    recs = json.load(open(path))
    tok = PTBTokenizer()
    gts = {i: [{"caption": c} for c in r["captions"]] for i, r in enumerate(recs)}
    res = {i: [{"caption": r["prediction"]}] for i, r in enumerate(recs)}
    gts_t = tok.tokenize(gts); res_t = tok.tokenize(res)
    score, scores = Cider().compute_score(gts_t, res_t)
    names = [r["image_name"] for r in recs]
    return float(score), dict(zip(names, map(float, scores)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    base_corpus, base_img = per_image_cider(args.base)
    names = list(base_img)
    b = np.array([base_img[n] for n in names])
    out = {"base": args.base, "base_corpus_cider": base_corpus, "n_images": len(names), "cands": {}}
    print(f"base {os.path.basename(args.base)} corpus CIDEr {100*base_corpus:.2f} (mean per-image {100*b.mean():.2f}), n={len(names)}", flush=True)
    for c in args.cand:
        cc, cimg = per_image_cider(c)
        x = np.array([cimg[n] for n in names])
        d = x - b
        idx = rng.integers(0, len(names), size=(args.n_boot, len(names)))
        boots = d[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = float((boots <= 0).mean())
        out["cands"][c] = {"corpus_cider": cc, "mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "p_one_sided": p,
                           "frac_images_improved": float((d > 0).mean()), "frac_changed": float((d != 0).mean())}
        print(f"  {os.path.basename(c):48s} corpus {100*cc:.2f}  mean diff {100*d.mean():+.2f}  95% CI [{100*lo:+.2f}, {100*hi:+.2f}]  p={p:.4f}  improved {100*(d>0).mean():.1f}% changed {100*(d!=0).mean():.1f}%", flush=True)
    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
