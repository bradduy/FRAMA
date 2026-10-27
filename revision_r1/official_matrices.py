"""Official per-candidate CIDEr-D matrices for the R1 revision (Access-2026-48860).

For a beam dump with N images and K candidates, writes <out>.npz with
  names [N]          image names in beam-dump order
  cands [N, K]       candidate strings
  M_all [N, K]       CIDEr-D of every candidate against all references of its image
  M_sel[k] [5, N, K] CIDEr-D against k of the four non-held-out references (k = 1..4,
                     the same rotated subsets as revision/crossref_sensitivity.py)
  M_one [5, N, K]    CIDEr-D against the single held-out reference r_f
  H_one [5, N]       CIDEr-D of the next human reference r_{(f+1) mod 5} against r_f
All scores use the official PTB tokenizer and pycocoevalcap CIDEr-D (x100). CIDEr-D
scores each caption against its own references with document frequencies taken from
the references only, so the score of a candidate does not depend on which candidates
of other images are scored with it. One Cider call is therefore made per candidate
slot j with exactly N entries, which reproduces the official corpus score of any
selection as the mean of the selected entries.

The cross-reference blocks use the first five references, as in the paper; images
with fewer than five references are dropped from those blocks only (mask in ok5).

Run on GCP2 with IFCap/rap_venv and Java 11:
  python official_matrices.py --beam_dump <dump.json> --out <prefix> [--no_xref]
"""
import argparse, json, os, sys, time
import numpy as np

EVAL = os.environ.get("COCOEVAL_DIR", os.path.expanduser("~/ZeroShotImageCap/IFCap/evaluation"))
sys.path.insert(0, EVAL)
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider


def cider_vec(refs, hyps):
    """refs: list of list of tokenized strings, hyps: list of tokenized strings."""
    gts = {i: r for i, r in enumerate(refs)}
    res = {i: [h] for i, h in enumerate(hyps)}
    return 100.0 * np.asarray(Cider().compute_score(gts, res)[1], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam_dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_xref", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out + ".npz"):
        print("exists, skip:", a.out + ".npz"); return
    t0 = time.time()
    beams = json.load(open(a.beam_dump))
    N = len(beams); K = min(len(b["beam"]) for b in beams)
    names = [b["image_name"] for b in beams]
    cands = [[b["beam"][j]["caption"] for j in range(K)] for b in beams]
    tok = PTBTokenizer()
    ct = tok.tokenize({f"{i}_{j}": [{"caption": cands[i][j]}] for i in range(N) for j in range(K)})
    rt = tok.tokenize({f"{i}_{r}": [{"caption": c}] for i, b in enumerate(beams) for r, c in enumerate(b["captions"])})
    refs_all = [[rt[f"{i}_{r}"][0] for r in range(len(b["captions"]))] for i, b in enumerate(beams)]
    hyp = lambda j: [ct[f"{i}_{j}"][0] for i in range(N)]
    M_all = np.stack([cider_vec(refs_all, hyp(j)) for j in range(K)], axis=1)
    out = {"names": np.array(names), "cands": np.array(cands, dtype=object), "M_all": M_all}
    print(f"M_all done {time.time()-t0:.0f}s, mean top-1 {M_all[:, 0].mean():.2f}", flush=True)
    if not a.no_xref:
        ok5 = np.array([len(b["captions"]) >= 5 for b in beams])
        idx = np.where(ok5)[0]
        r5 = [[rt[f"{i}_{r}"][0] for r in range(5)] for i in idx]
        M_sel = {k: np.full((5, N, K), np.nan) for k in (1, 2, 3, 4)}
        M_one = np.full((5, N, K), np.nan); H_one = np.full((5, N), np.nan)
        for f in range(5):
            others = [r for r in range(5) if r != f]
            one = [[x[f]] for x in r5]
            H_one[f, idx] = cider_vec(one, [x[(f + 1) % 5] for x in r5])
            for j in range(K):
                hj = [ct[f"{i}_{j}"][0] for i in idx]
                M_one[f, idx, j] = cider_vec(one, hj)
                for k in (1, 2, 3, 4):
                    sel = [others[(f + t) % 4] for t in range(k)]
                    M_sel[k][f, idx, j] = cider_vec([[x[r] for r in sel] for x in r5], hj)
            print(f"fold {f} done {time.time()-t0:.0f}s", flush=True)
        out.update({"ok5": ok5, "M_one": M_one, "H_one": H_one})
        out.update({f"M_sel{k}": v for k, v in M_sel.items()})
    np.savez_compressed(a.out + ".tmp.npz", **out)
    os.replace(a.out + ".tmp.npz", a.out + ".npz")
    print(f"wrote {a.out}.npz in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
