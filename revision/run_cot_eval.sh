#!/bin/bash
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; PY=$IF/rap_venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
cd $V && bash run_official_eval_remote.sh > revision/eval_cot.log 2>&1
cd $IF/evaluation
P=$V/preds/coco
[ -f $V/revision/bootstrap_cot_six.json ] || $PY $V/paired_bootstrap_remote.py --base $P/cot500_six_s1__overall.json --cand $P/cot500_six_s2__overall.json $P/cot500_six_blip2__overall.json --out $V/revision/bootstrap_cot_six.json > $V/revision/bootstrap_cot_six.log 2>&1
[ -f $V/revision/bootstrap_cot_zsum.json ] || $PY $V/paired_bootstrap_remote.py --base $P/cot500_zsum3__overall.json --cand $P/cot500_zsum3_s1__overall.json $P/cot500_zsum3_s2__overall.json --out $V/revision/bootstrap_cot_zsum.json > $V/revision/bootstrap_cot_zsum.log 2>&1
