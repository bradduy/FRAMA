#!/usr/bin/env bash
# Official COCO-suite evaluation of the selection-rule prediction files on GCP2.
# Idempotent: a prediction file is skipped when its result .txt already ends with
# a SPICE line, so the script can be relaunched after a pre-emption.
#
# Usage (on the VM, inside tmux):
#   bash ~/ZeroShotImageCap/macst_v1/v2_upgrade/run_official_eval_remote.sh
set -uo pipefail
IFCAP=~/ZeroShotImageCap/IFCap
HERE=~/ZeroShotImageCap/macst_v1/v2_upgrade
PRED=$HERE/preds
RES=$HERE/results_official
mkdir -p "$RES"

source "$IFCAP/rap_venv/bin/activate"
if [ -d /usr/lib/jvm/java-11-openjdk-amd64 ]; then
  export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
  export PATH=$JAVA_HOME/bin:$PATH
fi
cd "$IFCAP/evaluation"

# Order: COCO first (the calibration anchor), then Flickr30k, then NoCaps overall and its three splits.
files=$(ls "$PRED"/coco/*.json "$PRED"/flickr/*.json "$PRED"/coco_viecap/*.json "$PRED"/coco_ifcapsub/*.json "$PRED"/coco_viecap_full/*.json "$PRED"/nocaps/*__overall.json "$PRED"/nocaps/*__out_domain.json "$PRED"/nocaps/*__in_domain.json "$PRED"/nocaps/*__near_domain.json 2>/dev/null)
n=0; total=$(echo "$files" | wc -w)
for f in $files; do
  n=$((n+1))
  bench=$(basename "$(dirname "$f")")
  name=$(basename "$f" .json)
  out="$RES/${bench}__${name}.txt"
  if [ -f "$out" ] && grep -q "^SPICE:" "$out"; then
    echo "[$n/$total] skip $bench/$name (done)"; continue
  fi
  echo "[$n/$total] eval $bench/$name  $(date -u +%H:%M:%S)"
  python cocoeval.py --result_file_path "$f" > "$out.partial" 2>&1
  if grep -q "^SPICE:" "$out.partial"; then
    mv "$out.partial" "$out"
    grep -E "^(Bleu_4|METEOR|ROUGE_L|CIDEr|SPICE):" "$out" | tr '\n' ' '; echo
  else
    echo "  FAILED (no SPICE line); tail:"; tail -5 "$out.partial"
  fi
done
echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RES/_driver.log"
