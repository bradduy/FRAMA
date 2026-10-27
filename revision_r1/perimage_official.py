"""Per-image official CIDEr-D and SPICE for prediction files of the R1 revision.

Each prediction file is a list of {image_name, captions, prediction}. Writes one
JSON per file to <out_dir>/<bench>__<name>.json with the image names in file
order, per-image CIDEr-D (x100), per-image SPICE F-score (x100, the 'All' entry
that cocoeval averages) and the corpus values. Tokenization and scorers are the
official ones used by IFCap/evaluation/cocoeval.py. Files already written are
skipped, so the job can be restarted after a pre-emption.

  python perimage_official.py --out_dir perimage --bench coco --preds a.json b.json ...
"""
import argparse, json, os, sys, time
import numpy as np

EVAL = os.environ.get("COCOEVAL_DIR", os.path.expanduser("~/ZeroShotImageCap/IFCap/evaluation"))
sys.path.insert(0, EVAL)
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--no_spice", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cwd = os.getcwd()
    for p in a.preds:
        name = os.path.basename(p).replace("__overall.json", "").replace(".json", "")
        out = os.path.join(a.out_dir, f"{a.bench}__{name}.json")
        if os.path.exists(out):
            print("skip", out); continue
        t0 = time.time()
        recs = json.load(open(p))
        tok = PTBTokenizer()
        gts = tok.tokenize({i: [{"caption": c} for c in r["captions"]] for i, r in enumerate(recs)})
        res = tok.tokenize({i: [{"caption": r["prediction"]}] for i, r in enumerate(recs)})
        c_corpus, c_img = Cider().compute_score(gts, res)
        rec = {"pred_file": p, "image_names": [r["image_name"] for r in recs],
               "cider": (100 * np.asarray(c_img)).tolist(), "cider_corpus": 100 * float(c_corpus)}
        if not a.no_spice:
            os.chdir(os.path.join(EVAL, "pycocoevalcap", "spice"))
            s_corpus, s_img = Spice().compute_score(gts, res)
            os.chdir(cwd)
            rec["spice"] = [100 * float(s["All"]["f"]) for s in s_img]
            rec["spice_corpus"] = 100 * float(s_corpus)
        json.dump(rec, open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
        print(f"{a.bench} {name}: CIDEr {rec['cider_corpus']:.2f}"
              + (f" SPICE {rec['spice_corpus']:.2f}" if "spice" in rec else "")
              + f" ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
