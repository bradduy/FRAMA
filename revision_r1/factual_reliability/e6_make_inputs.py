"""E6 inputs (e6_prereg.md), built on the laptop and pushed to GCP2.

  spice_in/<unit>__<rule>.json   SPICE input [{image_id, test, refs}] for E6a and the fact extraction of E6c,
                                 with spice_in/<unit>__<rule>.names.json giving the image name of each image_id.
                                 Units: coco, nocaps, flickr (the paper's selections), thumb and thumb2 (the THumB
                                 machine systems and the human caption, for the validation of the E6c verifier against human Precision).
  e6b_items.json                 SugarCrepe and VALSE items on COCO Karpathy-test images: correct caption and foil.
  e6b_beams.json                 one beam-dump record per image holding every correct caption and foil of that image,
                                 for the unchanged FRAMA agent scorers.

  python3 e6_make_inputs.py
"""
import ast, glob, json, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); R1 = os.path.dirname(HERE); V2 = os.path.dirname(R1)
PRED = os.path.join(V2, "preds_r1")
RULES = ["lm_top1", "borda3_lm_bigG_blip", "zsum3_lm_bigG_blip", "stack6_all_equal", "stack5_noLM_blip_bigG_vqa_mbr_ground"]
THUMB = ["Up-Down", "Unified-VLP", "VinVL-base", "Human"]
out = os.path.join(HERE, "spice_in"); os.makedirs(out, exist_ok=True)


def spice_file(unit, rule, recs):
    json.dump([{"image_id": i, "test": r["prediction"], "refs": r["captions"]} for i, r in enumerate(recs)],
              open(os.path.join(out, f"{unit}__{rule}.json"), "w"))
    json.dump([r["image_name"] for r in recs], open(os.path.join(out, f"{unit}__{rule}.names.json"), "w"))


n = 0
for unit in ("coco", "nocaps", "flickr"):
    for rule in RULES:
        p = os.path.join(PRED, unit, f"{rule}__overall.json")
        if os.path.exists(p):
            spice_file(unit, rule, json.load(open(p))); n += 1
for s in THUMB:
    spice_file("thumb", s, json.load(open(os.path.join(R1, "human_bench", "thumb_preds", f"{s}__overall.json")))); n += 1
# VinVL-large was added after the verifier had started on the unit "thumb"; it is its own unit so that the chunking
# of "thumb" does not change.
spice_file("thumb2", "VinVL-large", json.load(open(os.path.join(R1, "human_bench", "thumb_preds", "VinVL-large__overall.json")))); n += 1
print("spice inputs:", n)

# E6b
beams = json.load(open(os.path.join(os.path.dirname(os.path.dirname(V2)), "IFCap", "inference_result", "coco-indomain",
                                    "coco-indomain-004_biggFull", "coco_beam_dump.json")))
kt = {int(b["image_name"].split("_")[-1].split(".")[0]): b for b in beams}
vg = json.load(open(os.path.join(HERE, "data", "vg_to_coco.json")))
coco_name = lambda cid: f"COCO_val2014_{cid:012d}.jpg"
items = []
for f in sorted(glob.glob(os.path.join(HERE, "data", "sugarcrepe", "*.json"))):
    sub = os.path.basename(f)[:-5]
    for k, v in json.load(open(f)).items():
        cid = int(v["filename"].split(".")[0])
        if cid in kt:
            items.append({"id": f"sugarcrepe:{sub}:{k}", "source": "SugarCrepe", "phenomenon": sub,
                          "image_name": coco_name(cid), "pos": v["caption"].strip(), "foil": v["negative_caption"].strip()})
for f in sorted(glob.glob(os.path.join(HERE, "data", "valse", "*.json"))):
    sub = os.path.basename(f)[:-5]
    for k, v in json.load(open(f)).items():
        fn = v["image_file"]
        cid = vg.get(fn[4:].split(".")[0]) if fn.startswith("v7w_") else int(fn.split("_")[-1].split(".")[0])
        m = v.get("mturk"); m = ast.literal_eval(m) if isinstance(m, str) else m
        if cid in kt and (m is None or m.get("caption", 0) >= 2):
            items.append({"id": f"valse:{sub}:{k}", "source": "VALSE", "phenomenon": sub,
                          "image_name": coco_name(cid), "pos": v["caption"].strip(), "foil": v["foil"].strip()})
items = [it for it in items if it["pos"] != it["foil"]]
json.dump(items, open(os.path.join(HERE, "e6b_items.json"), "w"), indent=0)
caps = defaultdict(list)
for it in items:
    for c in (it["pos"], it["foil"]):
        if c not in caps[it["image_name"]]:
            caps[it["image_name"]].append(c)
names = {b["image_name"]: b for b in beams}
recs = [{"image_name": im, "split": "e6b", "captions": names[im]["captions"],
         "beam": [{"caption": c, "score": 0.0} for c in cs]} for im, cs in sorted(caps.items())]
json.dump(recs, open(os.path.join(HERE, "e6b_beams.json"), "w"))
by = defaultdict(int)
for it in items:
    by[f"{it['source']}:{it['phenomenon']}"] += 1
print("e6b items:", len(items), "images:", len(recs), "max captions per image:", max(len(r["beam"]) for r in recs))
for k, v in sorted(by.items()):
    print(f"  {k:40s} {v}")
