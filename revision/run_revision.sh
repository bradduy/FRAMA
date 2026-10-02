#!/bin/bash
# Reviewer-response GPU chain on GCP2. GPU0: BLIP-2 direct captioning (COCO, NoCaps).
# GPU1: BLIP-ITM swap (COCO), then CLIPScore on the main rows once the BLIP-2 captions exist.
# Every step is resumable; completion is read from the files the python steps write.
set -u
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; RV=$V/revision
PY=$IF/rap_venv/bin/python; export HF_HOME=~/ZeroShotImageCap/hf_cache
COCO_DUMP=$IF/inference_result/coco-indomain-004_biggFull/coco_beam_dump.json
NC_DUMP=$IF/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json
COCO_IMG=$IF/annotations/coco/val2014; NC_IMG=$IF/annotations/nocaps/images
cd $RV
gpu0() {
  [ -f $V/preds/coco/blip2cap__overall.json ] || $PY blip2_caption.py --beam_dump $COCO_DUMP --image_dir $COCO_IMG --cache blip2cap_coco.jsonl --out_prefix $V/preds/coco/blip2cap --device cuda:0 > blip2cap_coco.log 2>&1
  [ -f $V/preds/nocaps/blip2cap__overall.json ] || $PY blip2_caption.py --beam_dump $NC_DUMP --image_dir $NC_IMG --cache blip2cap_nocaps.jsonl --out_prefix $V/preds/nocaps/blip2cap --device cuda:0 > blip2cap_nocaps.log 2>&1
}
gpu1() {
  $PY itm_swap_scorer.py --beam_dump $COCO_DUMP --image_dir $COCO_IMG --out_jsonl itm_flickr_coco.jsonl --device cuda:1 > itm_swap.log 2>&1
  until [ -f $V/preds/coco/blip2cap__overall.json ] && [ -f $V/preds/nocaps/blip2cap__overall.json ]; do sleep 60; done
  R="lm_top1 blip_top1 borda3_lm_bigG_blip zsum3_lm_bigG_blip stack4_zsum3_vqa stack4_zsum3_mbr zsum3_plus_ground_best stack6_all_equal stack5_noLM_blip_bigG_vqa_mbr_ground oracle20_approx blip2cap"
  for B in coco nocaps; do
    IMG=$COCO_IMG; [ $B = nocaps ] && IMG=$NC_IMG
    [ -f clipscore_$B.json ] || $PY clipscore_eval.py --image_dir $IMG --preds $(for r in $R; do echo $V/preds/$B/${r}__overall.json; done) --out clipscore_$B.json --device cuda:1 > clipscore_$B.log 2>&1
  done
}
gpu0 & gpu1 & wait
