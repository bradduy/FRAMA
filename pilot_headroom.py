import os
"""CPU pilot on the local COCO / NoCaps beam-20 dumps.

Approximate CIDEr-D (pycocoevalcap algorithm, simple tokenizer instead of the
Java PTBTokenizer). Absolute values shift by a fraction of a point versus the
official suite; relative comparisons on the same refs/tokenizer are valid.
"""
import json, re, sys, math
from collections import defaultdict
import numpy as np

ROOT = os.environ.get("IFCAP_ROOT", "IFCap")
_punct = re.compile(r"[^a-z0-9 ]+")

def tok(s):
    return _punct.sub(" ", s.lower()).split()

def ngrams(tokens, n_max=4):
    out = [defaultdict(int) for _ in range(n_max)]
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            out[n - 1][tuple(tokens[i:i + n])] += 1
    return out

class CiderD:
    """CIDEr-D with corpus-level IDF from the reference set (as pycocoevalcap)."""
    def __init__(self, refs_tok, n=4, sigma=6.0):
        self.n, self.sigma = n, sigma
        self.refs = [[ngrams(r, n) for r in rs] for rs in refs_tok]
        self.ref_len = [[len(r) for r in rs] for rs in refs_tok]
        df = defaultdict(float)
        for rs in self.refs:
            seen = set()
            for r in rs:
                for k in range(n):
                    seen.update(r[k].keys())
            for g in seen:
                df[g] += 1
        self.df = df
        self.log_ref = math.log(float(len(refs_tok)))

    def vec(self, ng, length):
        vecs = [defaultdict(float) for _ in range(self.n)]
        norms = [0.0] * self.n
        for k in range(self.n):
            for g, c in ng[k].items():
                idf = self.log_ref - math.log(max(1.0, self.df.get(g, 0.0)))
                v = float(c) * idf
                vecs[k][g] = v
                norms[k] += v * v
        return vecs, [math.sqrt(x) for x in norms]

    def score_one(self, i, hyp_tok):
        hng = ngrams(hyp_tok, self.n)
        hv, hn = self.vec(hng, len(hyp_tok))
        total = np.zeros(self.n)
        for r, rl in zip(self.refs[i], self.ref_len[i]):
            rv, rn = self.vec(r, rl)
            delta = float(len(hyp_tok) - rl)
            for k in range(self.n):
                s = 0.0
                for g, v in hv[k].items():
                    s += min(v, rv[k].get(g, 0.0)) * rv[k].get(g, 0.0)
                if hn[k] != 0 and rn[k] != 0:
                    s /= (hn[k] * rn[k])
                s *= math.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
                total[k] += s
        total /= len(self.refs[i])
        return float(total.mean() * 10.0)

def borda(sig_list):
    """sig_list: list of (N,K) arrays, higher=better. Returns (N,K) Borda score."""
    N, K = sig_list[0].shape
    out = np.zeros((N, K), dtype=np.float64)
    for s in sig_list:
        order = np.argsort(-s, axis=1)
        ranks = np.empty_like(order)
        rows = np.arange(N)[:, None]
        ranks[rows, order] = np.arange(K)[None, :]
        out += (K - 1 - ranks)
    return out / len(sig_list)

def z(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)

def run(name, beam_path, feat_path, pred_jsons=None, split_filter=None):
    print(f"\n==================== {name} ====================")
    beams = json.load(open(beam_path))
    if split_filter:
        beams = [b for b in beams if b.get("split") == split_filter]
    N = len(beams)
    refs_tok = [[tok(c) for c in b["captions"]] for b in beams]
    cands = [[x["caption"] for x in b["beam"]] for b in beams]
    K = min(len(c) for c in cands)
    cands = [c[:K] for c in cands]
    cider = CiderD(refs_tok)

    # per-candidate CIDEr matrix (N,K)
    M = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            M[i, j] = cider.score_one(i, tok(cands[i][j]))
    print(f"images={N} beam={K}  per-candidate CIDEr matrix computed")

    def report(label, idx):
        v = M[np.arange(N), idx].mean()
        print(f"{label:<46s} CIDEr~ {v:6.2f}")
        return v

    lm_idx = np.zeros(N, dtype=int)  # beam order = LM order
    base = report("LM top-1 (beam[0])", lm_idx)
    for k in (1, 2, 3, 5, 10, K):
        report(f"Oracle@{k} (best of first {k} by LM)", M[:, :k].argmax(axis=1))
    report("Random candidate (expected)", None) if False else print(f"{'Random candidate (expected)':<46s} CIDEr~ {M.mean():6.2f}")
    print(f"{'Worst candidate':<46s} CIDEr~ {M.min(axis=1).mean():6.2f}")

    if feat_path:
        f = np.load(feat_path, allow_pickle=True)
        feats = f["feats"].astype(np.float64)
        fc = f["captions"]
        if split_filter:
            names = [b["image_name"] for b in beams]
            keep = {n: i for i, n in enumerate(f["image_names"])}
            sel = np.array([keep[n] for n in names])
            feats, fc = feats[sel], fc[sel]
        # alignment check
        mism = sum(1 for i in range(N) for j in range(K) if str(fc[i][j]).strip() != cands[i][j].strip())
        print(f"feature/beam caption alignment mismatches: {mism}")
        LM, BG, BL, LEN = feats[..., 0], feats[..., 1], feats[..., 2], feats[..., 4]
        report("bigG CLIPScore top-1", BG.argmax(axis=1))
        report("BLIP-ITM top-1", BL.argmax(axis=1))
        for a in (0.48,):
            report(f"fixed z-mix LM/bigG alpha={a}", (a * z(LM) + (1 - a) * z(BG)).argmax(axis=1))
        report("equal z-sum LM+bigG+BLIP (3-signal linear)", (z(LM) + z(BG) + z(BL)).argmax(axis=1))
        b3 = borda([LM, BG, BL])
        report("Direct Borda (LM,bigG,BLIP) argmax", b3.argmax(axis=1))
        b4 = borda([LM, BG, BL, LEN])
        report("Direct Borda 4-way (+length, as in code)", b4.argmax(axis=1))
        # top-1 agreement
        t = np.stack([LM.argmax(1), BG.argmax(1), BL.argmax(1)], 1)
        all3 = (t[:, 0] == t[:, 1]) & (t[:, 1] == t[:, 2])
        print(f"top-1 agreement: all three {all3.mean()*100:.1f}%  LM=bigG {(t[:,0]==t[:,1]).mean()*100:.1f}%  LM=BLIP {(t[:,0]==t[:,2]).mean()*100:.1f}%  bigG=BLIP {(t[:,1]==t[:,2]).mean()*100:.1f}%")
        # oracle membership
        orc = M.argmax(axis=1)
        print(f"oracle-best is LM top-1: {(orc==0).mean()*100:.1f}%   in LM top-5: {(orc<5).mean()*100:.1f}%   is Borda pick: {(orc==b3.argmax(1)).mean()*100:.1f}%")
        # consensus-margin buckets -> where is the headroom?
        srt = np.sort(b3, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        pick = b3.argmax(axis=1)
        qs = np.quantile(margin, [0.25, 0.5, 0.75])
        edges = [-1, qs[0], qs[1], qs[2], 1e9]
        print("Borda margin quartile | n | CIDEr(pick) | CIDEr(oracle) | gap")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (margin > lo) & (margin <= hi)
            print(f"  ({lo:5.2f},{hi:5.2f}] | {m.sum():4d} | {M[m, pick[m]].mean():6.2f} | {M[m].max(1).mean():6.2f} | {M[m].max(1).mean()-M[m, pick[m]].mean():5.2f}")
        # per-scorer reliability proxy: Spearman of each scorer with per-candidate CIDEr
        from scipy.stats import spearmanr
        for nm, s in (("LM", LM), ("bigG", BG), ("BLIP", BL), ("Borda3", b3)):
            rs = [spearmanr(s[i], M[i]).correlation for i in range(N) if np.std(M[i]) > 0]
            print(f"mean per-image Spearman({nm}, candidate CIDEr) = {np.nanmean(rs):.3f}")

    if pred_jsons:
        name2i = {b["image_name"]: i for i, b in enumerate(beams)}
        for label, p in pred_jsons:
            try:
                d = json.load(open(p))
            except Exception as e:
                print(label, "ERR", e); continue
            rows = d if isinstance(d, list) else d.get("annotations", d)
            tot, n = 0.0, 0
            for r in rows:
                nm = r.get("image_name") or r.get("image_id") or r.get("filename")
                cap = r.get("prediction") or r.get("caption")
                if nm in name2i and cap:
                    tot += cider.score_one(name2i[nm], tok(cap)); n += 1
            print(f"{label:<46s} CIDEr~ {tot/max(n,1):6.2f}  (n={n})")
    return M

if __name__ == "__main__":
    run("COCO Karpathy test (bigG+BLIP pipeline beam-20)",
        f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_biggFull/coco_beam_dump.json",
        f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_csfn/features.npz",
        pred_jsons=[
            ("CSFN v3_all preds (transductive, official~117.5)", f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_csfn/coco_generated_captions_v3_all.json"),
            ("CSFN v2 LM+bigG+BLIP preds", f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_csfn/coco_generated_captions_v2_LMbigGBLIP.json"),
        ])
    run("NoCaps val overall (bigG+BLIP pipeline beam-20)",
        f"{ROOT}/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json",
        f"{ROOT}/inference_result/nocaps-bigg-blip/features.npz")
    run("NoCaps out-domain",
        f"{ROOT}/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json",
        f"{ROOT}/inference_result/nocaps-bigg-blip/features.npz", split_filter="out_domain")
    run("Flickr30k Karpathy test (beam-20, no features locally)",
        f"{ROOT}/inference_result/flickr-indomain-bigg-blip/flickr30k_beam_dump.json", None)
