"""Check frama_infer.py stage by stage against the artifacts behind the paper, on the first
COCO Karpathy-test images in retrieval order.

  python verify_infer.py --ifcap_root <IFCap checkout with annotations and val2014> \
      --ref <folder with coco_beam_dump.json, features.npz, vqascore_coco.jsonl, ground_coco.jsonl,
             mbr_trainidf_coco.npy, stack6_all_equal__overall.json, stack5_noLM_blip_bigG_vqa_mbr_ground__overall.json> \
      --n 96
  python verify_infer.py --ifcap_root ... --retrieval_only --n 5000

Retrieval is compared with caption_coco_image_coco_bigG_9.json, the BLIP-ITM step with
..._bigG_blip5.json and its entity file, the beam with the stored dump, every agent with its
released score file, and the final pick with the paper's prediction files. Since the R1 revision the MBR
reference is mbr_trainidf_coco.npy and the prediction files are those of predictions_r1/coco/.
"""
import argparse, json, os
import numpy as np
import frama_infer as F

ap = argparse.ArgumentParser()
ap.add_argument("--ifcap_root", required=True)
ap.add_argument("--ref", default="")
ap.add_argument("--n", type=int, default=96)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--retrieval_only", action="store_true")
ap.add_argument("--out", default="verify_infer.json")
a = ap.parse_args()
IF = a.ifcap_root
ann = lambda p: json.load(open(os.path.join(IF, "annotations", p)))
bigG9 = ann("retrieved_sentences/caption_coco_image_coco_bigG_9.json")
names = list(bigG9)[: a.n]
recs = [{"path": os.path.join(IF, "annotations", "coco", "val2014", x), "name": x} for x in names]
report = {"n": len(recs)}


def frac(k, total):
    return f"{k}/{total}"


if a.retrieval_only:
    F.stage_retrieve(recs, a.device)
    exact = sum(r["retrieved"] == bigG9[r["name"]] for r in recs)
    same_set = sum(set(r["retrieved"]) == set(bigG9[r["name"]]) for r in recs)
    report.update(retrieval_exact=frac(exact, len(recs)), retrieval_same_set=frac(same_set, len(recs)),
                  mismatches=[{"image": r["name"], "ours": r["retrieved"], "paper": bigG9[r["name"]]}
                              for r in recs if r["retrieved"] != bigG9[r["name"]]][:20])
else:
    blip5 = ann("retrieved_sentences/caption_coco_image_coco_bigG_blip5.json")
    ent = ann("retrieved_entity/caption_coco_image_coco_bigG_blip5.json")
    R = lambda f: os.path.join(a.ref, f)
    dump = {b["image_name"]: (i, b) for i, b in enumerate(json.load(open(R("coco_beam_dump.json"))))}
    npz = np.load(R("features.npz"), allow_pickle=True)
    fpos = {str(x): i for i, x in enumerate(npz["image_names"])}
    vqa = {json.loads(l)["image_name"]: json.loads(l)["vqascore"] for l in open(R("vqascore_coco.jsonl"))}
    gro = {json.loads(l)["image_name"]: json.loads(l) for l in open(R("ground_coco.jsonl"))}
    mbr = np.load(R("mbr_trainidf_coco.npy"))  # R1: MBR document frequencies from the training captions
    picks = {c: {p["image_name"]: p["prediction"] for p in json.load(open(R(f)))}
             for c, f in (("six", "stack6_all_equal__overall.json"), ("five", "stack5_noLM_blip_bigG_vqa_mbr_ground__overall.json"))}

    def paper_ground(g):
        ns = g["nouns"]
        v = np.array([np.mean([ns[n][0] for n in c["nouns"] if n in ns]) if any(n in ns for n in c["nouns"]) else np.nan for c in g["cands"]])
        if np.isnan(v).any():
            v[np.isnan(v)] = np.nanmean(v) if not np.isnan(v).all() else 0.0
        return v

    out = F.run([r["path"] for r in recs], a.device, "six", ifcap_root=IF,
                ckpt=os.path.join(IF, "checkpoints", "train_coco", "coco-indomain-004.pt"))
    cnt = {k: 0 for k in ("retrieval", "kept", "entities", "beam_text", "pick_six", "pick_five")}
    diff = {k: 0.0 for k in ("LM", "bigG", "BLIP", "VQA", "MBR", "ground")}
    beam_mismatch = []
    for r, o in zip(recs, out):
        nm = r["name"]
        cnt["retrieval"] += o["retrieved"] == bigG9[nm]
        cnt["kept"] += o["kept"] == blip5[nm]
        cnt["entities"] += o["entities"] == [w for f, w in ent[nm] if f >= 3]
        di, d = dump[nm]
        same_beam = [b["caption"] for b in o["beam"]] == [b["caption"] for b in d["beam"]]
        cnt["beam_text"] += same_beam
        if not same_beam:
            beam_mismatch.append({"image": nm, "ours_top3": [b["caption"] for b in o["beam"][:3]], "paper_top3": [b["caption"] for b in d["beam"][:3]],
                                  "same_kept": o["kept"] == blip5[nm], "same_entities": o["entities"] == [w for f, w in ent[nm] if f >= 3]})
            continue
        fe = npz["feats"][fpos[nm]]
        ref = {"LM": fe[:, 0], "bigG": fe[:, 1], "BLIP": fe[:, 2], "VQA": np.array(vqa[nm]), "MBR": mbr[di], "ground": paper_ground(gro[nm])}
        for k in diff:
            diff[k] = max(diff[k], float(np.abs(np.array([b[k] for b in o["beam"]]) - ref[k]).max()))
        for c in ("six", "five"):
            cnt[f"pick_{c}"] += F.adjudicate(o, c)[0] == picks[c][nm]
    report.update({k: frac(v, len(recs)) for k, v in cnt.items()})
    report["max_abs_diff_on_identical_beams"] = diff
    report["beam_mismatches"] = beam_mismatch[:20]
print(json.dumps(report, indent=1))
json.dump(report, open(a.out, "w"), indent=1)
