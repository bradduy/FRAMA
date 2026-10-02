"""Write the 1-in-5 (1,000-image) subsample of a COCO wider-beam dump as its own dump
file so the agent scorers run on the subsample only. Usage: make_sub_dump.py <bw>"""
import json, os, sys
bw = sys.argv[1]
C4 = os.path.expanduser("~/ZeroShotImageCap/IFCap/inference_result/c4_expand")
lst = [l.strip() for l in open(os.path.expanduser("~/ZeroShotImageCap/macst_v1/v2_upgrade/c5/coco_sub1000_images.txt")) if l.strip()]
d = json.load(open(f"{C4}/coco_beam{bw}_dump.json"))
sub = d[::5]
assert [b["image_name"] for b in sub] == lst, "subsample order differs from c5 list"
json.dump(sub, open(f"{C4}/coco_beam{bw}s_dump.json", "w"))
print(f"wrote coco_beam{bw}s_dump.json: {len(sub)} images, K={min(len(b['beam']) for b in sub)}")
