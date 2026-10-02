#!/bin/bash
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; PY=$IF/rap_venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
cd $V && bash run_official_eval_remote.sh > revision/eval_m4.log 2>&1
cd $IF/evaluation
for B in nocaps coco; do
  P=$V/preds/$B
  [ -f $V/revision/bootstrap_m4_$B.json ] || $PY $V/paired_bootstrap_remote.py --base $P/m4_five20__overall.json --cand $P/m4_five21__overall.json $P/blip2cap__overall.json --out $V/revision/bootstrap_m4_$B.json > $V/revision/bootstrap_m4_$B.log 2>&1
  [ -f $V/revision/bootstrap_m4_${B}_vsblip2.json ] || $PY $V/paired_bootstrap_remote.py --base $P/blip2cap__overall.json --cand $P/m4_five21__overall.json --out $V/revision/bootstrap_m4_${B}_vsblip2.json > $V/revision/bootstrap_m4_${B}_vsblip2.log 2>&1
done
