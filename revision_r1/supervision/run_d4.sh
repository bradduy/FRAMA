#!/bin/bash
# D4 / E3 control on GCP2: the full COCO pipeline without any COCO-fine-tuned matcher.
#   retrieval: bigG top-9 (unchanged) -> BLIP-ITM *flickr* rerank to top-5 -> entities (K=3)
#   decode:    IFCap coco-indomain-004, beam 20, k=5, K=3 (unchanged)
#   agents:    LM, bigG, BLIP-ITM coco (kept for the retrieval-only row), BLIP-ITM flickr, System One, detector, MBR
# A 500-image control decode with the original blip5 retrieval checks that this T4 reproduces the paper's beams.
# Lane 1 (GPU1): rerank, decodes, MBR, alignment features. Lane 0 (GPU0, after the E5 timing run has ended):
# System One, detector, BLIP-ITM flickr. Every step skips finished or resumable output, so the @reboot cron line
# can relaunch it after a pre-emption. Done when every check of d4_status.sh passes; no marker is written.
set -u
IF=~/ZeroShotImageCap/IFCap; V=~/ZeroShotImageCap/macst_v1/v2_upgrade; R=$V/revision_r1; D=$R/d4
PY=$IF/rap_venv/bin/python; IMG=$IF/annotations/coco/val2014
export HF_HOME=~/ZeroShotImageCap/hf_cache PYTHONPATH=$IF/src
mkdir -p $D/control $D/full $D/logs
BIGG9=caption_coco_image_coco_bigG_9.json; RF=caption_coco_image_coco_bigG_blip5flickr.json
DUMP=$D/full/coco_beam_dump.json; CTRL=$D/control/coco_beam_dump.json
log() { echo "$(date -u +%FT%TZ) $*" >> $D/logs/d4.log; }
okjson() { [ -f "$1" ] && $PY -c "import json,sys; json.load(open(sys.argv[1]))" "$1" 2>/dev/null; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
idle() {  # true when GPU $1 has no compute process and under 512 MiB in use
  local u m; u=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader); m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$m" -lt 512 ] && ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"
}
wait_idle() { until idle $1; do sleep 30; done; }

lane1() {
  cd $IF
  $PY -c "from huggingface_hub import snapshot_download as s; s('Salesforce/blip-itm-base-flickr')" >> $D/logs/dl.log 2>&1
  wait_idle 1
  if ! okjson annotations/retrieved_sentences/$RF; then
    log "rerank start"
    CUDA_VISIBLE_DEVICES=1 $PY src/blip_rerank_split.py --in_json annotations/retrieved_sentences/$BIGG9 --out_pos annotations/retrieved_sentences/$RF \
      --out_neg annotations/retrieved_sentences/caption_coco_image_coco_bigG_blipneg4flickr.json --image_folder $IMG \
      --top_k 5 --bot_n 4 --model Salesforce/blip-itm-base-flickr > $D/logs/rerank.log 2>&1 \
      && $PY -c "import json,sys; json.load(open(sys.argv[1]))" annotations/retrieved_sentences/$RF \
      && log "rerank end"
  fi
  okjson annotations/retrieved_entity/$RF || $PY src/entity_filter_one.py --in_json annotations/retrieved_sentences/$RF \
      --out_json annotations/retrieved_entity/$RF > $D/logs/entity.log 2>&1
  okjson $CTRL || { log "control decode start"; $PY src/validation_dumpbeam.py --device cuda:1 --weight_path checkpoints/train_coco/coco-indomain-004.pt \
      --out_path $D/control --retrieved_info caption_coco_image_coco_bigG_blip5.json --beam_width 20 --k 5 --K 3 --limit 500 > $D/logs/control.log 2>&1; log "control decode end"; }
  okjson $DUMP || { log "decode start"; $PY src/validation_dumpbeam.py --device cuda:1 --weight_path checkpoints/train_coco/coco-indomain-004.pt \
      --out_path $D/full --retrieved_info $RF --beam_width 20 --k 5 --K 3 > $D/logs/decode.log 2>&1; log "decode end"; }
  okjson $DUMP || { log "DECODE_FAILED"; return 1; }
  [ -f $D/mbr.npy ] || $PY $R/human_bench/thumb_mbr.py $DUMP $D/mbr.npy > $D/logs/mbr.log 2>&1
  $PY -c "import numpy,sys; numpy.load(sys.argv[1])['feats']" $D/features.npz 2>/dev/null || { log "features start";
    $PY src/csfn_extract_features.py --beam_dump $DUMP --image_folder $IMG --out_npz $D/features.npz --device cuda:1 > $D/logs/features.log 2>&1; log "features end"; }
}

lane0() {
  until okjson $DUMP; do sleep 120; done
  until [ -f $R/timing/summary.json ] || grep -q Traceback $R/logs/time_configs.log 2>/dev/null; do sleep 120; done
  wait_idle 0
  cd $V
  [ "$(lines $D/vqa.jsonl)" -ge 5000 ] || { log "vqa start"; $PY blip2vqa_scorer.py --beam_dump $DUMP --image_dir $IMG --out_jsonl $D/vqa.jsonl --device cuda:0 > $D/logs/vqa.log 2>&1; log "vqa end"; }
  [ "$(lines $D/ground.jsonl)" -ge 5000 ] || { log "ground start"; $PY detector_ground_scorer.py --beam_dump $DUMP --image_dir $IMG --out_jsonl $D/ground.jsonl \
      --entity_vocab $IF/annotations/vocabulary/vgcocooiobjects_v1_class2ind.json --device cuda:0 --max_queries 60 > $D/logs/ground.log 2>&1; log "ground end"; }
  [ "$(lines $D/itm_flickr.jsonl)" -ge 5000 ] || { log "itm start"; $PY $V/revision/itm_swap_scorer.py --beam_dump $DUMP --image_dir $IMG --out_jsonl $D/itm_flickr.jsonl --device cuda:0 > $D/logs/itm.log 2>&1; log "itm end"; }
}

log "launch"
lane1 & lane0 & wait
log "lanes returned"
