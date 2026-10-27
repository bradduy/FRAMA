#!/bin/bash
# T4 of thumb_prereg.md on GCP2: score the four THumB machine captions of each image with the unchanged FRAMA
# agents. Every step skips finished outputs and the scorers resume from their jsonl, so the @reboot cron line
# can relaunch this after a pre-emption. The GPUs are shared with another project's job, so the System One
# verifier is sharded over the free memory of both cards, and the bigG/BLIP-ITM pass runs in fp32 on the CPU,
# the precision used in the paper.
set -u
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; H=$V/revision_r1/human_bench
PY=$IF/rap_venv/bin/python; IMG=$IF/annotations/coco/val2014
export HF_HOME=~/ZeroShotImageCap/hf_cache
cd $H; mkdir -p t4 logs
[ -f t4/thumb_beams.json ] || $PY thumb_make_dump.py thumb_cider.npz t4/thumb_beams.json > logs/t4_dump.log 2>&1
N=$($PY -c "import json; print(len(json.load(open('t4/thumb_beams.json'))))")
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

lane_cpu() {
  [ -f t4/thumb_mbr.npy ] || $PY thumb_mbr.py t4/thumb_beams.json t4/thumb_mbr.npy > logs/t4_mbr.log 2>&1
  [ -f t4/thumb_align.npz ] || (cd $IF && $PY src/csfn_extract_features.py --beam_dump $H/t4/thumb_beams.json \
      --image_folder $IMG --out_npz $H/t4/thumb_align.npz --device cpu) > logs/t4_align.log 2>&1
}
lane_gpu() {
  until [ -f $V/revision_r1/logs/dl_r1_models.done ]; do sleep 60; done
  [ "$(lines t4/thumb_ground.jsonl)" -ge "$N" ] || (cd $V && $PY detector_ground_scorer.py --beam_dump $H/t4/thumb_beams.json \
      --image_dir $IMG --out_jsonl $H/t4/thumb_ground.jsonl --device cuda:1 \
      --entity_vocab $IF/annotations/vocabulary/vgcocooiobjects_v1_class2ind.json) > logs/t4_ground.log 2>&1
  [ "$(lines t4/thumb_vqa.jsonl)" -ge "$N" ] || (cd $V && $PY blip2vqa_scorer.py --beam_dump $H/t4/thumb_beams.json \
      --image_dir $IMG --out_jsonl $H/t4/thumb_vqa.jsonl --device cuda:0 --batch 4 \
      --max_memory "${T4_MAXMEM:-0:4GiB,1:4600MiB}") > logs/t4_vqa.log 2>&1
}
lane_cpu & lane_gpu & wait
