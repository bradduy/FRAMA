#!/usr/bin/env bash
# Rebuild the prediction files behind every table of the paper from the released
# artifacts (beam dumps and per-candidate agent scores). CPU only, no model download.
#
#   ./reproduce.sh          main tables: baselines, FRAMA stacks on COCO / NoCaps / Flickr30k,
#                           grounding, leave-one-out ablation, reliability weighting, ViECap transfer
#   ./reproduce.sh all      main tables plus verifier ladder, small judges, wider beams and the
#                           revision controls; about 50 minutes in total, mostly pure-Python MBR on beams of 40 and 80
#
# Prediction files land in preds/<benchmark>/<rule>__<split>.json. Score them with
# ./evaluate_official.sh after ./setup_eval.sh, then compare against reference_results/.
set -euo pipefail
cd "$(dirname "$0")"
export IFCAP_ROOT="${IFCAP_ROOT:-$PWD/data}"
PY="${PYTHON:-python3}"
MODE="${1:-main}"
step() { echo; echo "[$(date +%H:%M:%S)] $*"; }

step "1/7 beam-internal MBR agent, COCO and NoCaps"
$PY pilot_mbr_consensus.py
step "2/7 single-agent baselines, Borda and z-sum of LM + bigG + BLIP, Oracle@20"
$PY make_selection_preds.py
step "3/7 FRAMA six-agent and five-agent stacks, COCO and NoCaps"
$PY make_stack_preds.py
step "3/7 same rules on Flickr30k"
$PY make_flickr_stack_preds.py
step "4/7 grounding agent and VQA-subset rules"
$PY analyze_grounding.py
step "5/7 leave-one-out ablation and Dirichlet weight sweep"
$PY analyze_ablation.py
step "6/7 label-free reliability weighting at six agents"
$PY analyze_reliability6.py
step "7/7 transfer to ViECap, full 5,000-image set and 1,000-image subsample"
$PY analyze_c5_full.py
$PY analyze_c5.py

if [ "$MODE" = "all" ]; then
  step "extra: verifier ladder (SigLIP2, BLIP-2 ITM)"
  $PY analyze_ladder.py
  $PY make_ladder_preds.py
  step "extra: small System One judges"
  $PY analyze_smalljudge.py
  step "extra: wider beams, Oracle@N on the full set and stacks on the 1,000-image subsample"
  for bw in 40 80; do
    f="$IFCAP_ROOT/inference_result/c4_expand/coco_beam${bw}_dump.json"
    [ -f "$f" ] || gunzip -k "$f.gz"
  done
  $PY analyze_expansion.py "$IFCAP_ROOT/inference_result/c4_expand/coco_beam40_dump.json" \
                           "$IFCAP_ROOT/inference_result/c4_expand/coco_beam80_dump.json"
  $PY analyze_c4_stack.py 40s
  $PY analyze_c4_stack.py 80s
  step "extra: revision controls (BLIP-ITM checkpoint swap, System One vs System Two, BLIP-2 caption as candidate)"
  for b in coco nocaps flickr; do mkdir -p "preds/$b"; cp revision/blip2cap/$b/*.json "preds/$b/"; done
  $PY revision/analyze_itm_swap.py
  $PY revision/analyze_cot.py
  $PY revision/analyze_m4.py coco
  $PY revision/analyze_m4.py nocaps
  $PY revision/pairwise_corr.py
fi

step "NoCaps in-domain and near-domain splits"
$PY split_nocaps_preds.py

step "done: $(find preds -name '*.json' | wc -l | tr -d ' ') prediction files under preds/"
