#!/bin/bash
# D4 / E3: official COCO evaluation of preds_r1/coco_d4 (built locally by d4_preds.py and pushed), then paired
# bootstrap over images. Skips finished outputs, so it can be relaunched after a pre-emption.
set -u
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; R=$V/revision_r1; P=$V/preds_r1/coco_d4
PY=$IF/rap_venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
mkdir -p $R/results_r1 $R/logs
cd $IF/evaluation
for f in $P/*__overall.json; do
  out=$R/results_r1/coco_d4__$(basename $f .json).txt
  if [ -f $out ] && grep -q "^SPICE:" $out; then continue; fi
  $PY cocoeval.py --result_file_path $f > $out.partial 2>&1 && grep -q "^SPICE:" $out.partial && mv $out.partial $out
done
[ -f $R/d4_bootstrap.json ] || $PY $V/paired_bootstrap_remote.py --base $P/nococo_zsum3_lm_bigG_blip__overall.json \
  --cand $P/nococo_stack6_all_equal__overall.json $P/nococo_stack5_noLM_blip_bigG_vqa_mbr_ground__overall.json \
  --out $R/d4_bootstrap.json > $R/logs/d4_bootstrap.log 2>&1
[ -f $R/d4_bootstrap_lmbase.json ] || $PY $V/paired_bootstrap_remote.py --base $P/nococo_lm_top1__overall.json \
  --cand $P/nococo_stack6_all_equal__overall.json $P/nococo_stack5_noLM_blip_bigG_vqa_mbr_ground__overall.json \
  --out $R/d4_bootstrap_lmbase.json > $R/logs/d4_bootstrap_lmbase.log 2>&1
