#!/bin/bash
# Rebuild every prediction file of the revised manuscript from the released agent scores, on CPU,
# and check each one against the SHA-256 of the files the paper was scored from.
#
#   ./verify_r1.sh                         about one hour on one CPU core
#   IFCAP_ANNOTATIONS=<IFCap>/annotations ./verify_r1.sh
#
# The rules that add a BLIP-2 caption as a 21st candidate recompute the consensus agent over the
# COCO training captions; they need IFCap's annotations/coco/coco_with_entities.pickle, given by
# IFCAP_ANNOTATIONS. Without it those files are reported as not rebuilt.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
./reproduce.sh all
if [ -n "${IFCAP_ANNOTATIONS:-}" ] && [ ! -e data/annotations ]; then ln -s "$IFCAP_ANNOTATIONS" data/annotations; fi
IFCAP_ROOT=data $PY revision_r1/rebuild_preds_r1.py || echo "rebuild stopped early; the check below lists what is missing"
$PY - <<'PYEOF'
import hashlib, os
ok, bad, missing = 0, [], []
for line in open("reference_results/r1/predictions_r1.sha256"):
    h, f = line.split(None, 1); f = f.strip(); p = os.path.join("preds_r1", f)
    if not os.path.exists(p):
        missing.append(f); continue
    if hashlib.sha256(open(p, "rb").read()).hexdigest() == h:
        ok += 1
    else:
        bad.append(f)
print(f"identical: {ok}   different: {len(bad)}   not rebuilt: {len(missing)}")
for f in bad: print("  different:", f)
groups = sorted({f.split("/")[0] + "/" + f.split("/")[1].split("__")[0] for f in missing})
for g in groups: print("  not rebuilt:", g)
if missing and all(f.startswith("coco_d4/") or "/m4_" in f for f in missing):
    print("coco_d4/ comes from revision_r1/supervision/d4_preds.py; m4_* needs IFCAP_ANNOTATIONS.")
PYEOF
