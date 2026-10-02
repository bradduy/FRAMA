#!/usr/bin/env bash
# C5: second captioner (ViECap COCO checkpoint) on the 1,000-image COCO subsample.
# Stage 1 beam-20 dump (ViECap code, rap_venv); stages 2-4 the same six-agent
# scorers used for IFCap (features, BLIP-2 VQA, OWLv2). Args: device
set -uo pipefail
dev=${1:-cuda:0}
IF=~/ZeroShotImageCap/IFCap; V2=~/ZeroShotImageCap/macst_v1/v2_upgrade; VE=~/ZeroShotImageCap/ViECap
C5=$V2/c5; mkdir -p "$C5"
source "$IF/rap_venv/bin/activate"
export HF_HOME=~/ZeroShotImageCap/hf_cache
DUMP=$C5/viecap_coco_sub1000_beam_dump.json
IMG=$IF/annotations/coco/val2014
if [ ! -f "$DUMP" ]; then
  cd "$VE"
  python "$C5/validation_dumpbeam_viecap.py" --weight_path checkpoints/train_coco/coco_prefix-0014.pt \
    --annotations_root "$IF/annotations" --image_list "$C5/coco_sub1000_images.txt" \
    --out_json "$DUMP" --device "$dev" 2>&1 | grep -vE "it/s\]|Warning|warn"
fi
[ -f "$DUMP" ] || { echo "dump failed"; exit 2; }
cd "$IF"
if [ ! -f "$C5/viecap_features.npz" ]; then
  python src/csfn_extract_features.py --beam_dump "$DUMP" --image_folder "$IMG" \
    --out_npz "$C5/viecap_features.npz" --device "$dev" 2>&1 | grep -vE "it/s\]|Warning|warn"
fi
cd "$V2"
python blip2vqa_scorer.py --beam_dump "$DUMP" --image_dir "$IMG" --out_jsonl "$C5/vqascore_viecap_coco.jsonl" --device "$dev" 2>&1 | grep -vE "torch_dtype|Fetching|Loading checkpoint"
python detector_ground_scorer.py --beam_dump "$DUMP" --image_dir "$IMG" --out_jsonl "$C5/ground_viecap_coco.jsonl" \
  --entity_vocab $IF/annotations/vocabulary/vgcocooiobjects_v1_class2ind.json --device "$dev" --max_queries 60 2>&1 | grep -vE "slow image processor|use_fast|NLTK will not authorize|incr_download"
echo "C5_VIECAP_AGENTS_DONE $(date -u +%FT%TZ)" >> "$C5/viecap_agents.done"
