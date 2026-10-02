#!/usr/bin/env bash
# GPU1 chain: wait for the beam-80 dump, then rescore it on the 1,000-image subsample (same list as beam 40s and C5).
cd ~/ZeroShotImageCap/macst_v1/v2_upgrade
source ~/ZeroShotImageCap/IFCap/rap_venv/bin/activate
until [ -f ~/ZeroShotImageCap/IFCap/inference_result/c4_expand/coco_beam80.done ] || ! tmux has-session -t c4_beam80 2>/dev/null; do sleep 60; done
[ -f ~/ZeroShotImageCap/IFCap/inference_result/c4_expand/coco_beam80_dump.json ] || { echo "beam80 dump missing" >> chain_gpu1.log; exit 2; }
echo "chain1: beam80 dump over $(date -u +%T)"
python make_sub_dump.py 80 && bash run_c4_rescore.sh 80s cuda:1 1
echo "CHAIN1_DONE $(date -u +%FT%TZ)" >> chain_gpu1.done
