#!/usr/bin/env bash
# C5 full set: ViECap beam-20 on all 5,000 COCO Karpathy-test images, then the six agents.
# Two GPUs: dev_a runs the dump then the bigG/BLIP features; dev_b waits for the dump and runs
# BLIP-2 VQA then OWLv2 grounding. Args: dev_a dev_b. VQA and grounding are resumable.
set -uo pipefail
deva=${1:-cuda:1}; devb=${2:-cuda:0}
IF=~/ZeroShotImageCap/IFCap; V2=~/ZeroShotImageCap/macst_v1/v2_upgrade; VE=~/ZeroShotImageCap/ViECap; C5=$V2/c5
source "$IF/rap_venv/bin/activate"; export HF_HOME=~/ZeroShotImageCap/hf_cache
DUMP=$C5/viecap_coco_full_beam_dump.json; IMG=$IF/annotations/coco/val2014
log() { echo "$(date -u +%FT%TZ) $*" >> "$C5/c5_full.log"; }
if [ ! -f "$DUMP" ]; then
  log "stage1 dump start on $deva"
  ( cd "$VE" && python "$C5/validation_dumpbeam_viecap.py" --weight_path checkpoints/train_coco/coco_prefix-0014.pt \
    --annotations_root "$IF/annotations" --out_json "$DUMP" --device "$deva" > "$C5/dump_full.log" 2>&1 )
  log "stage1 dump end"
fi
[ -f "$DUMP" ] || { log "DUMP_FAILED"; exit 2; }
# branch b: VQA then grounding on dev_b
(
  cd "$V2"; log "branch b: vqa start on $devb"
  python blip2vqa_scorer.py --beam_dump "$DUMP" --image_dir "$IMG" --out_jsonl "$C5/vqascore_viecap_coco_full.jsonl" --device "$devb" > "$C5/vqa_full.log" 2>&1
  log "branch b: vqa end, ground start"
  python detector_ground_scorer.py --beam_dump "$DUMP" --image_dir "$IMG" --out_jsonl "$C5/ground_viecap_coco_full.jsonl" \
    --entity_vocab $IF/annotations/vocabulary/vgcocooiobjects_v1_class2ind.json --device "$devb" --max_queries 60 > "$C5/ground_full.log" 2>&1
  log "branch b: ground end"
) &
pb=$!
# branch a: features on dev_a
if [ ! -f "$C5/viecap_full_features.npz" ]; then
  log "branch a: features start on $deva"
  ( cd "$IF" && python src/csfn_extract_features.py --beam_dump "$DUMP" --image_folder "$IMG" --out_npz "$C5/viecap_full_features.npz" --device "$deva" > "$C5/features_full.log" 2>&1 )
  log "branch a: features end"
fi
wait $pb
nv=$(wc -l < "$C5/vqascore_viecap_coco_full.jsonl" 2>/dev/null || echo 0); ng=$(wc -l < "$C5/ground_viecap_coco_full.jsonl" 2>/dev/null || echo 0)
if [ "$nv" -ge 5000 ] && [ "$ng" -ge 5000 ] && [ -f "$C5/viecap_full_features.npz" ]; then echo "C5_FULL_AGENTS_DONE $(date -u +%FT%TZ)" >> "$C5/viecap_full_agents.done"; else log "INCOMPLETE vqa=$nv ground=$ng"; fi
