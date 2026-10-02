#!/usr/bin/env bash
# Paired bootstrap CIs for every leave-one-out cell (base: six-agent stack) on COCO, NoCaps
# (overall and out-domain) and Flickr30k, official per-image CIDEr. CPU only.
set -uo pipefail
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; PY=$IF/rap_venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
cd "$IF/evaluation"
for spec in "coco overall" "nocaps overall" "nocaps out_domain" "flickr overall"; do
  set -- $spec; b=$1; sp=$2; P=$V/preds/$b
  cands=""; for a in bigG BLIP VQA MBR ground; do cands="$cands $P/loo_minus_${a}__${sp}.json"; done
  $PY $V/paired_bootstrap_remote.py --base $P/stack6_all_equal__${sp}.json --cand $cands $P/stack5_noLM_blip_bigG_vqa_mbr_ground__${sp}.json --out $V/bootstrap_loo_${b}_${sp}.json > $V/bootstrap_loo_${b}_${sp}.log 2>&1
  echo "done $b $sp"
done
echo "LOO_BOOT_DONE $(date -u +%FT%TZ)" >> $V/loo_bootstrap.done
