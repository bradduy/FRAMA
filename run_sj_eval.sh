#!/usr/bin/env bash
# Official eval of the small-judge prediction files plus paired bootstrap against the
# six-agent stack on the same 1,000 images. Idempotent.
set -uo pipefail
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; PY=$IF/rap_venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
cd "$V"; bash run_official_eval_remote.sh > eval_sj.log 2>&1
cd "$IF/evaluation"
cands=$(ls $V/preds/coco/sj_*_stack6swap__overall.json $V/preds/coco/sj_*_zsum3plus__overall.json 2>/dev/null | tr "\n" " ")
$PY $V/paired_bootstrap_remote.py --base $V/preds/coco_ifcapsub/stack6_all_equal__overall.json --cand $cands $V/preds/coco_ifcapsub/zsum3_lm_bigG_blip__overall.json $V/preds/coco_ifcapsub/stack4_zsum3_vqa__overall.json --out $V/bootstrap_sj_vs_stack6.json > $V/bootstrap_sj_vs_stack6.log 2>&1
echo "SJ_EVAL_DONE $(date -u +%FT%TZ)" >> $V/sj_eval.done
