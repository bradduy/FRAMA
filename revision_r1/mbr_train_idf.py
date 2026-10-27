"""Beam-consensus (MBR) agent with document frequencies from the captioner's TRAINING corpus.

The submitted manuscript computed the MBR agent with CIDEr-D whose document frequencies were
taken from the human references of the test set (pilot_mbr_consensus.CiderD built on
b["captions"]). That uses test references at inference, which the text-only regime forbids.
This script recomputes the agent with document frequencies from the text corpus the captioner
was trained on: the 566,747 COCO Karpathy-train captions for every COCO-trained beam (COCO,
NoCaps, ViECap, beam-width study) and the 145,000 Flickr30k Karpathy-train captions for
Flickr30k. Each training caption is one document. Everything else is unchanged: CIDEr-D
n = 1..4, sigma = 6, clipping, length penalty, x10, and the agent score of candidate j is its
mean CIDEr-D against the other K - 1 candidates of the same beam.

  python3 mbr_train_idf.py coco nocaps flickr viecapfull coco_beam40 coco_beam80
writes ../mbr_trainidf_<bench>.npy, shape (N, K), aligned with the beam dump order.
"""
import json, math, os, pickle, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
sys.path.insert(0, V2)
from pilot_headroom import tok, ngrams, ROOT  # same tokenizer and n-gram code as the agent

DUMPS = {
    "coco": f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_biggFull/coco_beam_dump.json",
    "nocaps": f"{ROOT}/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json",
    "flickr": f"{ROOT}/inference_result/flickr-indomain-bigg-blip/flickr30k_beam_dump.json",
    "viecapfull": f"{V2}/c5/viecap_coco_full_beam_dump.json",
    "coco_beam40": f"{ROOT}/inference_result/c4_expand/coco_beam40_dump.json",
    "coco_beam80": f"{ROOT}/inference_result/c4_expand/coco_beam80_dump.json",
}
CORPUS = {"coco": f"{ROOT}/annotations/coco/coco_with_entities.pickle",
          "flickr": f"{ROOT}/annotations/flickr30k/flickr30k_with_entities.pickle"}
N_GRAM, SIGMA = 4, 6.0
_DF, _LOGN = None, None


def build_df(corpus_key):
    caps = [c for _, c in pickle.load(open(CORPUS[corpus_key], "rb"))]
    df = defaultdict(float)
    for c in caps:
        ng = ngrams(tok(c), N_GRAM)
        seen = set()
        for k in range(N_GRAM):
            seen.update(ng[k].keys())
        for g in seen:
            df[g] += 1
    return dict(df), math.log(float(len(caps)))


def vec(ng, length):
    vecs = [dict() for _ in range(N_GRAM)]; norms = [0.0] * N_GRAM
    for k in range(N_GRAM):
        for g, c in ng[k].items():
            v = float(c) * (_LOGN - math.log(max(1.0, _DF.get(g, 0.0))))
            vecs[k][g] = v; norms[k] += v * v
    return vecs, [math.sqrt(x) for x in norms], length


def pair(a, b):
    (ha, na, la), (hb, nb, lb) = a, b
    delta = float(la - lb); tot = 0.0
    for k in range(N_GRAM):
        s = sum(min(v, hb[k].get(g, 0.0)) * hb[k].get(g, 0.0) for g, v in ha[k].items())
        if na[k] and nb[k]:
            s /= (na[k] * nb[k])
        tot += s * math.e ** (-(delta ** 2) / (2 * SIGMA ** 2))
    return tot / N_GRAM * 10.0


def beam_scores(caps):
    vs = [vec(ngrams(t, N_GRAM), len(t)) for t in (tok(c) for c in caps)]
    K = len(vs)
    return [float(np.mean([pair(vs[j], vs[l]) for l in range(K) if l != j])) for j in range(K)]


def init(df, logn):
    global _DF, _LOGN
    _DF, _LOGN = df, logn


def main():
    for bench in sys.argv[1:]:
        out = os.path.join(V2, f"mbr_trainidf_{bench}.npy")
        if os.path.exists(out):
            print("skip", out); continue
        df, logn = build_df("flickr" if bench == "flickr" else "coco")
        beams = json.load(open(DUMPS[bench]))
        K = min(len(b["beam"]) for b in beams)
        jobs = [[x["caption"] for x in b["beam"][:K]] for b in beams]
        with Pool(processes=max(1, os.cpu_count() - 2), initializer=init, initargs=(df, logn)) as pool:
            S = np.array(pool.map(beam_scores, jobs, chunksize=16))
        np.save(out, S)
        print(f"{bench}: {S.shape} written to {out}", flush=True)


if __name__ == "__main__":
    main()
