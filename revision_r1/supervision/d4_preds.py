"""D4 / E3: selection files for the COCO pipeline without any COCO-fine-tuned matcher (run_d4.sh outputs in d4/).

The retrieval rerank and the output cross-encoder both use BLIP-ITM fine-tuned on Flickr30k ("nococo_*").
The "retrieval_only_*" rows keep the COCO cross-encoder at the output, so the retrieval and output swaps can be
separated. Agents, z-normalisation and equal weights are those of rebuild_preds_r1.py. Also reports how many of the
500 control beams, decoded on the T4 with the original retrieval, are identical to the paper's beams.
Writes ../preds_r1/coco_d4/<rule>__overall.json.

  python3 d4_preds.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE); D = os.path.join(HERE, "d4")
sys.path.insert(0, V2)
from pilot_headroom import z, ROOT

beams = json.load(open(os.path.join(D, "full", "coco_beam_dump.json"))); N = len(beams)
K = min(len(b["beam"]) for b in beams); names = [b["image_name"] for b in beams]
f = np.load(os.path.join(D, "features.npz"), allow_pickle=True)
pos = {str(n): i for i, n in enumerate(f["image_names"])}
F = f["feats"].astype(np.float64)[[pos[n] for n in names]][:, :K]


def jl(path, key):
    d = {}
    for line in open(path):
        r = json.loads(line); d[r["image_name"]] = r[key][:K]
    return np.array([d[n] for n in names], float)


G = np.full((N, K), np.nan)  # analyze_grounding.grounding_features, g_mean
gr = {}
for line in open(os.path.join(D, "ground.jsonl")):
    r = json.loads(line); gr[r["image_name"]] = r
for i, n in enumerate(names):
    r = gr[n]
    for j, c in enumerate(r["cands"][:K]):
        sc = [r["nouns"][w][0] for w in c["nouns"] if w in r["nouns"]]
        if sc:
            G[i, j] = np.mean(sc)
    row = G[i]; row[np.isnan(row)] = np.nanmean(row) if not np.isnan(row).all() else 0.0
A = {"LM": F[..., 0], "bigG": F[..., 1], "BLIPcoco": F[..., 2], "BLIP": jl(os.path.join(D, "itm_flickr.jsonl"), "itm"),
     "VQA": jl(os.path.join(D, "vqa.jsonl"), "vqascore"), "MBR": np.load(os.path.join(D, "mbr.npy")), "ground": G}
assert all(v.shape == (N, K) and not np.isnan(v).any() for v in A.values())

zs = lambda names_: sum(z(A[a]) for a in names_)
SIX = ["LM", "bigG", "BLIP", "VQA", "MBR", "ground"]; FIVE = SIX[1:]
SIXc = ["LM", "bigG", "BLIPcoco", "VQA", "MBR", "ground"]; FIVEc = SIXc[1:]
rules = {"nococo_lm_top1": A["LM"], "nococo_blip_top1": A["BLIP"],
         "nococo_zsum3_lm_bigG_blip": zs(["LM", "bigG", "BLIP"]),
         "nococo_stack6_all_equal": zs(SIX), "nococo_stack5_noLM_blip_bigG_vqa_mbr_ground": zs(FIVE),
         "nococo_stack5_noBLIP": zs(["LM", "bigG", "VQA", "MBR", "ground"]),
         "retrieval_only_blip_top1": A["BLIPcoco"], "retrieval_only_zsum3_lm_bigG_blip": zs(["LM", "bigG", "BLIPcoco"]),
         "retrieval_only_stack6_all_equal": zs(SIXc), "retrieval_only_stack5_noLM_blip_bigG_vqa_mbr_ground": zs(FIVEc)}
out = os.path.join(V2, "preds_r1", "coco_d4"); os.makedirs(out, exist_ok=True)
for name, sc in rules.items():
    pick = sc.argmax(1)
    recs = [{"image_name": b["image_name"], "captions": b["captions"], "prediction": b["beam"][int(p)]["caption"]}
            for b, p in zip(beams, pick)]
    json.dump(recs, open(os.path.join(out, f"{name}__overall.json"), "w"))

ctrl = json.load(open(os.path.join(D, "control", "coco_beam_dump.json")))
paper = {b["image_name"]: b for b in json.load(open(f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_biggFull/coco_beam_dump.json"))}
same_beam = np.mean([[x["caption"] for x in c["beam"]] == [x["caption"] for x in paper[c["image_name"]]["beam"]] for c in ctrl])
same_top1 = np.mean([c["prediction"] == paper[c["image_name"]]["prediction"] for c in ctrl])
paper_ent = np.mean([set(b["retrieved_entities"]) == set(paper[b["image_name"]]["retrieved_entities"]) for b in beams])
summary = {"n": N, "K": K, "control_n": len(ctrl), "control_identical_beam": float(same_beam),
           "control_identical_top1": float(same_top1), "nococo_entities_same_as_paper": float(paper_ent),
           "nococo_top1_same_as_paper": float(np.mean([b["prediction"] == paper[b["image_name"]]["prediction"] for b in beams]))}
json.dump(summary, open(os.path.join(HERE, "d4_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1)); print("rules:", len(rules), "->", out)
