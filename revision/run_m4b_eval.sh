#!/bin/bash
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; PY=$IF/rap_venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
cd $V && bash run_official_eval_remote.sh > revision/eval_m4b.log 2>&1
cd $IF/evaluation
for B in nocaps coco; do P=$V/preds/$B
  $PY $V/paired_bootstrap_remote.py --base $P/m4_five21__overall.json --cand $P/m4_blip21__overall.json $P/m4_bigG21__overall.json $P/m4_s1_21__overall.json $P/m4_align2_21__overall.json $P/m4_align2s1_21__overall.json $P/m4_four21_noMBR__overall.json --out $V/revision/bootstrap_m4b_$B.json > $V/revision/bootstrap_m4b_$B.log 2>&1
done
