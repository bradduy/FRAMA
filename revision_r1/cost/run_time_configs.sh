#!/bin/bash
# E5 timing on GCP2. Waits until the THumB agent pass (human_bench/run_thumb_t4.sh) has finished and one T4 is
# idle (no compute process, under 512 MiB used, on two checks 20 s apart), then runs time_configs.py on that
# card. Cost numbers are only valid on an unshared card, so this never starts on a busy one. Finished stages are
# skipped, so the @reboot cron line can relaunch it after a pre-emption; done when timing/summary.json exists.
set -u
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; R=$V/revision_r1; H=$R/human_bench; IF=~/ZeroShotImageCap/IFCap
PY=$IF/rap_venv/bin/python
export HF_HOME=~/ZeroShotImageCap/hf_cache JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
cd $R; mkdir -p timing logs
[ -f timing/summary.json ] && exit 0
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
until [ "$(lines $H/t4/thumb_vqa.jsonl)" -ge 500 ] && [ "$(lines $H/t4/thumb_ground.jsonl)" -ge 500 ] \
      && [ -f $H/t4/thumb_align.npz ] && [ -f $H/t4/thumb_mbr.npy ]; do sleep 60; done
idle() {  # prints the index of an idle GPU, if any
  local busy; busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | while IFS=', ' read -r i u m; do
    [ "$m" -lt 512 ] && ! grep -q "$u" <<< "$busy" && echo "$i"
  done | head -1
}
while true; do
  g=$(idle); [ -n "$g" ] && { sleep 20; [ "$(idle)" = "$g" ] && break; }
  sleep 20
done
echo "start on GPU $g $(date -u +%FT%TZ)" >> logs/time_configs.log
$PY time_configs.py --device cuda:$g --out_dir timing >> logs/time_configs.log 2>&1
