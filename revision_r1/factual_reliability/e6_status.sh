#!/bin/bash
# Prints "spice e6b chunks" counts for E6 on GCP2; complete at "19 3 120".
cd ~/ZeroShotImageCap/macst_v1/v2_upgrade/revision_r1/e6
NB=$(python3 -c "import json; print(len(json.load(open('e6b_beams.json'))))")
l() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
b=0; [ -f e6b_align.npz ] && b=$((b+1)); [ "$(l e6b_vqa.jsonl)" -ge "$NB" ] && b=$((b+1)); [ "$(l e6b_ground.jsonl)" -ge "$NB" ] && b=$((b+1))
echo "$(ls spice_out/*.json 2>/dev/null | grep -vc partial) $b $(ls e6c_out/*/chunk_*.jsonl 2>/dev/null | wc -l)"
