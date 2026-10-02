"""Official-metric version of the cross-reference oracle (C1 protocol).

For each benchmark and each held-out reference index f in 0..4:
  select, per image, the beam candidate with the highest official CIDEr-D against
  the other four references (corpus IDF from those four-reference gts), then score
  the selected caption against the held-out reference f alone (corpus IDF from the
  single-reference gts). The same single-held-out-reference scoring is applied to
  the plain Oracle@20 pick (selected on all five references, so it has seen the
  held-out one), to LM top-1, the three-agent z-sum pick and the six-agent stack
  pick, read from the prediction files. Mean over the five folds is reported.
Run on GCP2 from IFCap/evaluation with rap_venv's python (Java 11 on PATH):
  python crossref_oracle_official.py --bench coco --beam_dump <dump> --preds_dir <preds/coco> --out <json>
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/ZeroShotImageCap/IFCap/evaluation"))
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider


def tokenize_all(beams, K, extra_preds):
    tok = PTBTokenizer()
    cand = {f"{i}_{j}": [{"caption": beams[i]["beam"][j]["caption"]}] for i in range(len(beams)) for j in range(K)}
    refs = {f"{i}_{r}": [{"caption": c}] for i in range(len(beams)) for r, c in enumerate(beams[i]["captions"][:5])}
    preds = {f"{name}_{i}": [{"caption": p}] for name, lst in extra_preds.items() for i, p in enumerate(lst)}
    ct, rt, pt = tok.tokenize(cand), tok.tokenize(refs), tok.tokenize(preds)
    return ct, rt, pt


def cider_per_image(gts_tok, res_tok):
    _, scores = Cider().compute_score(gts_tok, res_tok)
    return np.asarray(scores, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--beam_dump", required=True)
    ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--rules", nargs="+", default=["lm_top1", "zsum3_lm_bigG_blip", "stack6_all_equal", "stack5_noLM_blip_bigG_vqa_mbr_ground", "oracle20_approx"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    beams = json.load(open(args.beam_dump))
    beams = [b for b in beams if len(b["captions"]) >= 5]
    N = len(beams); K = min(len(b["beam"]) for b in beams)
    names = [b["image_name"] for b in beams]
    extra = {}
    for r in args.rules:
        p = os.path.join(args.preds_dir, f"{r}__overall.json")
        if os.path.exists(p):
            d = {x["image_name"]: x["prediction"] for x in json.load(open(p))}
            extra[r] = [d[n] for n in names]
    print(f"{args.bench}: N={N} K={K} rules={list(extra)}", flush=True)
    ct, rt, pt = tokenize_all(beams, K, extra)
    out = {"bench": args.bench, "n_images": N, "K": K, "folds": []}
    # full five-reference Oracle@K, official
    gts5 = {f"{i}_{j}": [rt[f"{i}_{r}"][0] for r in range(5)] for i in range(N) for j in range(K)}
    res_all = {f"{i}_{j}": ct[f"{i}_{j}"] for i in range(N) for j in range(K)}
    s5 = cider_per_image(gts5, res_all).reshape(N, K)
    pick5 = s5.argmax(1)
    out["oracle20_fiveref_official"] = float(100 * s5[np.arange(N), pick5].mean())
    print(f"  Oracle@{K} five-ref official CIDEr {out['oracle20_fiveref_official']:.2f}", flush=True)
    per_rule = {r: [] for r in list(extra) + ["crossref_oracle", "oracle20_fiveref_pick"]}
    for f in range(5):
        others = [r for r in range(5) if r != f]
        gts4 = {f"{i}_{j}": [rt[f"{i}_{r}"][0] for r in others] for i in range(N) for j in range(K)}
        s4 = cider_per_image(gts4, res_all).reshape(N, K)
        pick = s4.argmax(1)
        # score selections against the held-out reference alone
        gts1 = {str(i): [rt[f"{i}_{f}"][0]] for i in range(N)}
        def score_sel(captions_tok):
            res = {str(i): captions_tok[i] for i in range(N)}
            return float(100 * cider_per_image(gts1, res).mean())
        fold = {"heldout": f}
        fold["crossref_oracle"] = score_sel([ct[f"{i}_{pick[i]}"] for i in range(N)])
        fold["oracle20_fiveref_pick"] = score_sel([ct[f"{i}_{pick5[i]}"] for i in range(N)])
        for r in extra:
            fold[r] = score_sel([pt[f"{r}_{i}"] for i in range(N)])
        for k_, v in fold.items():
            if k_ != "heldout": per_rule[k_].append(v)
        out["folds"].append(fold)
        print("  fold", f, {k_: round(v, 2) for k_, v in fold.items() if k_ != "heldout"}, flush=True)
    out["mean_over_folds"] = {k_: float(np.mean(v)) for k_, v in per_rule.items()}
    m = out["mean_over_folds"]
    if "lm_top1" in m:
        out["attainable_gain_over_lm"] = m["crossref_oracle"] - m["lm_top1"]
        out["fiveref_pick_gain_over_lm_singleref"] = m["oracle20_fiveref_pick"] - m["lm_top1"]
    print("MEAN over folds (single held-out ref scoring):", {k_: round(v, 2) for k_, v in m.items()}, flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
