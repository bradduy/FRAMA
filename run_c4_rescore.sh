#!/usr/bin/env bash
# C4: score every candidate of a wider-beam dump with the six agents.
# Args: beam_width device   (expects IFCap/inference_result/c4_expand/coco_beam<bw>_dump.json)
# Stage 1 (LM/bigG/BLIP features) is not resumable; stages 2-3 are.
set -uo pipefail
bw=$1; dev=$2; stride=${3:-1}
IF=~/ZeroShotImageCap/IFCap; V2=~/ZeroShotImageCap/macst_v1/v2_upgrade
cd "$IF"; source rap_venv/bin/activate
export HF_HOME=~/ZeroShotImageCap/hf_cache
DUMP=$IF/inference_result/c4_expand/coco_beam${bw}_dump.json
[ -f "$DUMP" ] || { echo "missing $DUMP"; exit 2; }
OUT=$IF/inference_result/c4_expand
if [ ! -f "$OUT/features_beam${bw}.npz" ]; then
  python src/csfn_extract_features.py --beam_dump "$DUMP" --image_folder annotations/coco/val2014 \
    --out_npz "$OUT/features_beam${bw}.npz" --device "$dev" 2>&1 | grep -vE "it/s\]|Warning|warn"
fi
cd "$V2"
python blip2vqa_scorer.py --beam_dump "$DUMP" --image_dir $IF/annotations/coco/val2014 \
  --out_jsonl vqascore_coco_beam${bw}.jsonl --stride "$stride" --device "$dev" 2>&1 | grep -vE "torch_dtype|Fetching|Loading checkpoint"
python detector_ground_scorer.py --beam_dump "$DUMP" --image_dir $IF/annotations/coco/val2014 \
  --out_jsonl ground_coco_beam${bw}.jsonl --entity_vocab $IF/annotations/vocabulary/vgcocooiobjects_v1_class2ind.json \
  --device "$dev" --max_queries 60 --stride "$stride" 2>&1 | grep -vE "slow image processor|use_fast|NLTK will not authorize|incr_download"
echo "C4_RESCORE_DONE beam=$bw $(date -u +%FT%TZ)" >> "$OUT/coco_beam${bw}_rescore.done"
