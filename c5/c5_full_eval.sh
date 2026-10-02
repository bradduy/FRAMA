#!/usr/bin/env bash
# Official eval of the full-set ViECap prediction files plus paired bootstrap (bases: z-sum and LM top-1).
set -uo pipefail
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; PY=$IF/rap_venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
cd "$V"; bash run_official_eval_remote.sh > eval_c5_full.log 2>&1
cd "$IF/evaluation"; P=$V/preds/coco_viecap_full
$PY $V/paired_bootstrap_remote.py --base $P/zsum3_lm_bigG_blip__overall.json --cand $P/stack6_all_equal__overall.json $P/stack5_noLM_blip_bigG_vqa_mbr_ground__overall.json $P/stack4_zsum3_vqa__overall.json $P/stack4_zsum3_mbr__overall.json $P/blip_top1__overall.json $P/lm_top1__overall.json --out $V/c5/bootstrap_viecap_full.json > $V/c5/bootstrap_viecap_full.log 2>&1
$PY $V/paired_bootstrap_remote.py --base $P/lm_top1__overall.json --cand $P/zsum3_lm_bigG_blip__overall.json $P/stack6_all_equal__overall.json $P/stack5_noLM_blip_bigG_vqa_mbr_ground__overall.json --out $V/c5/bootstrap_viecap_full_lmbase.json > $V/c5/bootstrap_viecap_full_lmbase.log 2>&1
echo "C5_FULL_EVAL_DONE $(date -u +%FT%TZ)" >> $V/c5/c5_full_eval.done
