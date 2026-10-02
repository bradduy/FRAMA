#!/usr/bin/env bash
# C5 official eval (cocoeval via the shared driver) and paired bootstrap for the ViECap subsample.
set -uo pipefail
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; PY=$IF/rap_venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
cd "$V"; bash run_official_eval_remote.sh > eval_c5.log 2>&1
cd "$IF/evaluation"
P=$V/preds/coco_viecap
$PY $V/paired_bootstrap_remote.py --base $P/zsum3_lm_bigG_blip__overall.json --cand $P/stack6_all_equal__overall.json $P/stack5_noLM_blip_bigG_vqa_mbr_ground__overall.json $P/stack4_zsum3_vqa__overall.json $P/stack4_zsum3_mbr__overall.json $P/blip_top1__overall.json $P/lm_top1__overall.json --out $V/c5/bootstrap_viecap.json > $V/c5/bootstrap_viecap.log 2>&1
$PY $V/paired_bootstrap_remote.py --base $P/lm_top1__overall.json --cand $P/zsum3_lm_bigG_blip__overall.json $P/stack6_all_equal__overall.json $P/stack5_noLM_blip_bigG_vqa_mbr_ground__overall.json --out $V/c5/bootstrap_viecap_lmbase.json > $V/c5/bootstrap_viecap_lmbase.log 2>&1
Q=$V/preds/coco_ifcapsub
$PY $V/paired_bootstrap_remote.py --base $Q/zsum3_lm_bigG_blip__overall.json --cand $Q/stack6_all_equal__overall.json $Q/stack5_noLM_blip_bigG_vqa_mbr_ground__overall.json $Q/lm_top1__overall.json --out $V/c5/bootstrap_ifcapsub.json > $V/c5/bootstrap_ifcapsub.log 2>&1
echo "C5_EVAL_DONE $(date -u +%FT%TZ)" >> $V/c5/c5_eval.done
