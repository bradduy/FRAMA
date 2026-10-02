"""Write the in-domain and near-domain NoCaps prediction files from each overall file.

The selection scripts write preds/nocaps/<rule>__overall.json and __out_domain.json; the
paper also reports the in-domain and near-domain splits, which are the overall records
filtered by the split field of the beam dump. Existing files are left untouched."""
import glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("IFCAP_ROOT", "IFCap")
dump = json.load(open(f"{ROOT}/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json"))
split = {b["image_name"]: b.get("split") for b in dump}
n = 0
for path in sorted(glob.glob(os.path.join(HERE, "preds", "nocaps", "*__overall.json"))):
    rule = os.path.basename(path)[:-len("__overall.json")]
    recs = None
    for sp in ("in_domain", "near_domain", "out_domain"):
        out = os.path.join(HERE, "preds", "nocaps", f"{rule}__{sp}.json")
        if os.path.exists(out):
            continue
        recs = recs if recs is not None else json.load(open(path))
        json.dump([r for r in recs if split[r["image_name"]] == sp], open(out, "w")); n += 1
print(f"wrote {n} NoCaps split files")
