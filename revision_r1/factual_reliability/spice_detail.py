"""E6a / E6c on GCP2: run the official SPICE 1.0 jar with per-category scores and the scene-graph tuples of every caption.

Same jar, Stanford models, Java 11 and parse cache as the paper's cocoeval runs; only -detailed and -subset are added.
The output is written to <out>.partial by the jar and renamed only after Python has parsed it.

  python spice_detail.py spice_in/coco__lm_top1.json spice_out/coco__lm_top1.json
"""
import json, os, subprocess, sys

SP = os.path.expanduser("~/ZeroShotImageCap/IFCap/evaluation/pycocoevalcap/spice")
src, dst = sys.argv[1], sys.argv[2]
part = dst + ".partial"
cmd = ["java", "-jar", "-Xmx8G", os.path.join(SP, "spice-1.0.jar"), src, "-cache", os.path.join(SP, "cache"),
       "-out", part, "-subset", "-detailed", "-silent"]
subprocess.check_call(cmd, cwd=SP)
res = json.load(open(part))
assert len(res) == len(json.load(open(src))), "SPICE output incomplete"
os.replace(part, dst)
print("wrote", dst, len(res))
