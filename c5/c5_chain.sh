#!/usr/bin/env bash
# C5 chain for tmux (bash, not sh): ViECap beam-20 dump now on a lightly loaded GPU,
# then the heavy agents once GPU 1's current jobs end. Args: dump_device [agent_device]
set -uo pipefail
ddev=${1:-cuda:1}; adev=${2:-cuda:1}
IF=~/ZeroShotImageCap/IFCap; V2=~/ZeroShotImageCap/macst_v1/v2_upgrade; VE=~/ZeroShotImageCap/ViECap; C5=$V2/c5
source "$IF/rap_venv/bin/activate"
export HF_HOME=~/ZeroShotImageCap/hf_cache
DUMP=$C5/viecap_coco_sub1000_beam_dump.json
if [ ! -f "$DUMP" ]; then
  cd "$VE"
  python "$C5/validation_dumpbeam_viecap.py" --weight_path checkpoints/train_coco/coco_prefix-0014.pt \
    --annotations_root "$IF/annotations" --image_list "$C5/coco_sub1000_images.txt" \
    --out_json "$DUMP" --device "$ddev" > "$C5/dump.log" 2>&1
fi
[ -f "$DUMP" ] || { echo "DUMP_FAILED" >> "$C5/dump.log"; exit 2; }
# take the first GPU that frees: GPU 0 when chain_gpu0 ends, GPU 1 when chain_gpu1 and the beam-80 dump end
adev=""
while [ -z "$adev" ]; do
  if ! tmux has-session -t chain_gpu0 2>/dev/null; then adev=cuda:0
  elif ! tmux has-session -t chain_gpu1 2>/dev/null && ! tmux has-session -t c4_beam80 2>/dev/null; then adev=cuda:1
  else sleep 120; fi
done
echo "C5 agents on $adev $(date -u +%FT%TZ)" >> "$C5/agents.log"
bash "$C5/run_c5_viecap.sh" "$adev" >> "$C5/agents.log" 2>&1
