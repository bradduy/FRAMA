#!/usr/bin/env bash
# Small-VLM yes-probability judges on the 1,000-image COCO subsample (same list as C4/C5).
# Args: device. Sequential: SmolVLM-256M, SmolVLM-500M, Qwen2-VL-2B (skipped if disk < 7 GB).
set -uo pipefail
dev=${1:-cuda:1}
IF=~/ZeroShotImageCap/IFCap; V2=~/ZeroShotImageCap/macst_v1/v2_upgrade; C4=$IF/inference_result/c4_expand
cd "$V2"; source "$IF/rap_venv/bin/activate"; export HF_HOME=~/ZeroShotImageCap/hf_cache
DUMP=$C4/coco_beam20s_dump.json
if [ ! -f "$DUMP" ]; then
  python - <<'PY'
import json, os
C4 = os.path.expanduser("~/ZeroShotImageCap/IFCap/inference_result/c4_expand")
lst = [l.strip() for l in open(os.path.expanduser("~/ZeroShotImageCap/macst_v1/v2_upgrade/c5/coco_sub1000_images.txt")) if l.strip()]
d = json.load(open(os.path.expanduser("~/ZeroShotImageCap/IFCap/inference_result/coco-indomain-004_biggFull/coco_beam_dump.json")))
sub = d[::5]; assert [b["image_name"] for b in sub] == lst
json.dump(sub, open(f"{C4}/coco_beam20s_dump.json", "w")); print("wrote coco_beam20s_dump.json", len(sub))
PY
fi
IMG=$IF/annotations/coco/val2014
run() {  # family model tag dtype batch
  local fam=$1 model=$2 tag=$3 dtype=$4 batch=$5
  [ -f "smalljudge_${tag}.done" ] && { echo "skip $tag"; return; }
  echo "=== $tag  $(date -u +%T)"
  python smallvlm_yesprob_scorer.py --family "$fam" --model "$model" --beam_dump "$DUMP" --image_dir "$IMG" \
    --out_jsonl "smalljudge_${tag}_coco20s.jsonl" --device "$dev" --dtype "$dtype" --batch "$batch" 2>&1 \
    | grep -vE "Warning|warn|Fetching|Loading checkpoint|it/s\]" | tee "smalljudge_${tag}.log"
  n=$(wc -l < "smalljudge_${tag}_coco20s.jsonl" 2>/dev/null || echo 0)
  [ "$n" -ge 1000 ] && echo "SMALLJUDGE_DONE $tag $(date -u +%FT%TZ)" >> "smalljudge_${tag}.done"
}
run smolvlm HuggingFaceTB/SmolVLM-256M-Instruct smolvlm256m float32 20
run smolvlm HuggingFaceTB/SmolVLM-500M-Instruct smolvlm500m float32 20
free_gb=$(df -BG ~ | tail -1 | awk '{gsub("G","",$4); print $4}')
if [ "$free_gb" -ge 7 ]; then run qwen2vl Qwen/Qwen2-VL-2B-Instruct qwen2vl2b float16 10; else echo "skip qwen2vl2b: only ${free_gb} GB free"; fi
echo "SMALLJUDGE_ALL_DONE $(date -u +%FT%TZ)" >> smalljudge_all.done
