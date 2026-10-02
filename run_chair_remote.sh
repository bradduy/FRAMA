#!/usr/bin/env bash
# Official CHAIR_s / CHAIR_i (LisaAnne/Hallucination @ 6e4d33c, hash-pinned adapter in
# aic_repro/chair) on the COCO Karpathy-test selection files of the v2 stack.
# CPU only. The adapter scores two prediction files per call, so runs are paired.
# Idempotent: a pair is skipped when its report already has a "status" field.
#
# Usage (GCP2, inside tmux):  bash ~/ZeroShotImageCap/macst_v1/v2_upgrade/run_chair_remote.sh
set -uo pipefail
HERE=~/ZeroShotImageCap/macst_v1/v2_upgrade
PRED=$HERE/preds/coco
OUT=$HERE/chair_official
REPO=~/ZeroShotImageCap/aic_repro
A=$REPO/assets/chair_official
PY=$A/venv/bin/python
mkdir -p "$OUT"
cd "$REPO"

PAIRS=(
  "lm_top1 stack6_all_equal"
  "borda3_lm_bigG_blip stack5_noLM_blip_bigG_vqa_mbr_ground"
  "blip_top1 zsum3_lm_bigG_blip"
  "stack4_zsum3_vqa zsum3_plus_ground_best"
  "stack4_zsum3_mbr oracle20_approx"
)
for pair in "${PAIRS[@]}"; do
  set -- $pair; a=$1; b=$2
  rep="$OUT/coco__${a}__vs__${b}.json"
  if [ -f "$rep" ] && grep -q '"status"' "$rep"; then echo "skip $a vs $b (done)"; continue; fi
  echo "=== $a vs $b  $(date -u +%H:%M:%S)"
  common=(--pcvr-predictions "$PRED/${a}__overall.json" --matcher-predictions "$PRED/${b}__overall.json"
          --annotation-root "$A/annotations" --chair-source-root "$A/sources/Hallucination-6e4d33c4bedf"
          --pattern-source-root "$A/sources/pattern-5b85d998c30d" --nltk-data-root "$A/nltk_data"
          --manifest manifests/chair_official_sources.json --output "$rep" --force)
  if ! "$PY" scripts/run_official_chair.py "${common[@]}" 2>&1 | tee "$rep.log"; then
    echo "  canonical run failed; retrying as noncanonical diagnostic"
    "$PY" scripts/run_official_chair.py "${common[@]}" --expected-count 5000 --noncanonical-smoke 2>&1 | tee -a "$rep.log"
  fi
done
echo "CHAIR_ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/_driver.log"
