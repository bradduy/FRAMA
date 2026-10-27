#!/bin/bash
# E6 on GCP2 (e6_prereg.md). CPU lane: SPICE -detailed -subset for every selection file (E6a, and the facts of E6c).
# GPU0: the unchanged FRAMA agent scorers on the E6b foil items, then the E6c verifier. GPU1: the E6c verifier.
# Every step skips finished output and the verifier resumes from its partial chunks, so the @reboot cron line can
# relaunch this after a pre-emption. Done when every check of e6_status.sh passes; no marker is written.
set -u
IF=~/ZeroShotImageCap/IFCap; V=~/ZeroShotImageCap/macst_v1/v2_upgrade; E=$V/revision_r1/e6
PY=$IF/rap_venv/bin/python; IMG=$IF/annotations/coco/val2014
export HF_HOME=~/ZeroShotImageCap/hf_cache PYTHONPATH=$IF/src
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
cd $E; mkdir -p spice_out e6c_out logs
rm -rf e6c_out/*/chunk_*.lock
log() { echo "$(date -u +%FT%TZ) $*" >> logs/e6.log; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
NB=$($PY -c "import json; print(len(json.load(open('e6b_beams.json'))))")

lane_spice() {
  for u in thumb coco nocaps flickr; do
    for f in spice_in/${u}__*.json; do
      case $f in *.names.json) continue;; esac
      o=spice_out/$(basename $f)
      [ -f $o ] || { log "spice $o"; $PY spice_detail.py $E/$f $E/$o >> logs/spice.log 2>&1; }
    done
  done
  log "spice lane returned"
}
lane_gpu0() {
  $PY -c "import numpy,sys; numpy.load(sys.argv[1])['feats']" e6b_align.npz 2>/dev/null || { log "e6b align start"
    (cd $IF && $PY src/csfn_extract_features.py --beam_dump $E/e6b_beams.json --image_folder $IMG --out_npz $E/e6b_align.npz \
      --device cuda:0) > logs/e6b_align.log 2>&1; log "e6b align end"; }
  [ "$(lines e6b_vqa.jsonl)" -ge "$NB" ] || { log "e6b vqa start"; (cd $V && $PY blip2vqa_scorer.py --beam_dump $E/e6b_beams.json \
      --image_dir $IMG --out_jsonl $E/e6b_vqa.jsonl --device cuda:0) > logs/e6b_vqa.log 2>&1; log "e6b vqa end"; }
  [ "$(lines e6b_ground.jsonl)" -ge "$NB" ] || { log "e6b ground start"; (cd $V && $PY detector_ground_scorer.py --beam_dump $E/e6b_beams.json \
      --image_dir $IMG --out_jsonl $E/e6b_ground.jsonl --device cuda:0 --max_queries 60 \
      --entity_vocab $IF/annotations/vocabulary/vgcocooiobjects_v1_class2ind.json) > logs/e6b_ground.log 2>&1; log "e6b ground end"; }
  log "verify gpu0 start"; $PY e6c_verify.py --device cuda:0 --batch 32 >> logs/verify_gpu0.log 2>&1; log "verify gpu0 returned"
}
lane_gpu1() { log "verify gpu1 start"; $PY e6c_verify.py --device cuda:1 --batch 32 >> logs/verify_gpu1.log 2>&1; log "verify gpu1 returned"; }

log "launch"
lane_spice & lane_gpu0 & lane_gpu1 & wait
log "lanes returned"
