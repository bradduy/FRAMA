#!/usr/bin/env bash
# GPU0 chain b: after the C5 agents finish, rescore the beam-40 dump on the 1,000-image subsample.
cd ~/ZeroShotImageCap/macst_v1/v2_upgrade
source ~/ZeroShotImageCap/IFCap/rap_venv/bin/activate
until [ -f c5/viecap_agents.done ] || ! tmux has-session -t c5_viecap 2>/dev/null; do sleep 90; done
echo "chain0b: C5 agents over $(date -u +%T)"
python make_sub_dump.py 40 && bash run_c4_rescore.sh 40s cuda:0 1
echo "CHAIN0B_DONE $(date -u +%FT%TZ)" >> chain_gpu0b.done
