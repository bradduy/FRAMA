#!/usr/bin/env bash
# Official COCO caption evaluation (Java PTB tokenizer, BLEU-4, METEOR, ROUGE-L, CIDEr, SPICE)
# of every prediction file under preds/, with the toolkit shipped in the IFCap v1.0 release.
# Idempotent: a prediction file is skipped when its result .txt already ends with a SPICE line,
# so the script can be relaunched after an interruption.
#
# Usage: ./setup_eval.sh once, then ./evaluate_official.sh [preds_subdir ...]
#   e.g. ./evaluate_official.sh coco nocaps     (default: every subdirectory of preds/)
#   PREDS_DIR=preds_r1 RESULTS_DIR=results_r1 ./evaluate_official.sh   for the files of the revised paper
# Results: results/<bench>__<rule>__<split>.txt, same names as reference_results/official_metrics/.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL="${COCOEVAL_DIR:-$HERE/third_party/evaluation}"
PRED="${PREDS_DIR:-$HERE/preds}"
RES="${RESULTS_DIR:-$HERE/results}"
PY="${PYTHON:-python3}"
[ -f "$EVAL/cocoeval.py" ] || { echo "IFCap evaluation toolkit not found at $EVAL; run ./setup_eval.sh or set COCOEVAL_DIR"; exit 2; }
if [ -d /usr/lib/jvm/java-11-openjdk-amd64 ]; then
  export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
  export PATH=$JAVA_HOME/bin:$PATH
fi
mkdir -p "$RES"
cd "$EVAL"

if [ $# -gt 0 ]; then dirs=("$@"); else dirs=($(ls "$PRED")); fi
files=$(for d in "${dirs[@]}"; do ls "$PRED/$d"/*.json 2>/dev/null; done)
n=0; total=$(echo "$files" | wc -w)
for f in $files; do
  n=$((n+1))
  bench=$(basename "$(dirname "$f")")
  name=$(basename "$f" .json)
  out="$RES/${bench}__${name}.txt"
  if [ -f "$out" ] && grep -q "^SPICE:" "$out"; then
    echo "[$n/$total] skip $bench/$name (done)"; continue
  fi
  echo "[$n/$total] eval $bench/$name  $(date -u +%H:%M:%S)"
  $PY cocoeval.py --result_file_path "$f" > "$out.partial" 2>&1
  if grep -q "^SPICE:" "$out.partial"; then
    mv "$out.partial" "$out"
    grep -E "^(Bleu_4|METEOR|ROUGE_L|CIDEr|SPICE):" "$out" | tr '\n' ' '; echo
  else
    echo "  FAILED (no SPICE line); tail:"; tail -5 "$out.partial"
  fi
done
echo "Compare with the paper: python3 compare_results.py"
