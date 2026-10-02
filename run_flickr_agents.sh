#!/usr/bin/env bash
# Flickr30k Karpathy test: score the stored beam-20 dump with the GPU agents
# (LM/bigG/BLIP features, BLIP-2 VQA, OWLv2 grounding). MBR and the stack are
# computed locally afterwards (make_stack_preds-style). Args: device
# Stage 1 is not resumable; stages 2-3 are.
set -uo pipefail
dev=${1:-cuda:0}
IF=~/ZeroShotImageCap/IFCap; V2=~/ZeroShotImageCap/macst_v1/v2_upgrade
cd "$IF"; source rap_venv/bin/activate
export HF_HOME=~/ZeroShotImageCap/hf_cache
DUMP=$IF/inference_result/flickr-indomain-bigg-blip/flickr30k_beam_dump.json
IMG=$IF/annotations/flickr30k/flickr30k-images
[ -f "$DUMP" ] || { echo "missing $DUMP"; exit 2; }
OUT=$IF/inference_result/flickr-indomain-bigg-blip
if [ ! -f "$OUT/features.npz" ]; then
  python src/csfn_extract_features.py --beam_dump "$DUMP" --image_folder "$IMG" \
    --out_npz "$OUT/features.npz" --device "$dev" 2>&1 | grep -vE "it/s\]|Warning|warn"
fi
cd "$V2"
python blip2vqa_scorer.py --beam_dump "$DUMP" --image_dir "$IMG" --out_jsonl vqascore_flickr.jsonl --device "$dev" 2>&1 | grep -vE "torch_dtype|Fetching|Loading checkpoint"
python detector_ground_scorer.py --beam_dump "$DUMP" --image_dir "$IMG" --out_jsonl ground_flickr.jsonl \
  --entity_vocab $IF/annotations/vocabulary/vgcocooiobjects_v1_class2ind.json --device "$dev" 2>&1 | grep -vE "slow image processor|use_fast|NLTK will not authorize|incr_download"
echo "FLICKR_AGENTS_DONE $(date -u +%FT%TZ)" >> flickr_agents.done
