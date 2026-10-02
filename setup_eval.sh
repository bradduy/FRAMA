#!/usr/bin/env bash
# Fetch the COCO caption evaluation toolkit shipped with the IFCap v1.0 release (550 MB, Stanford
# CoreNLP models for SPICE included) into third_party/evaluation, the copy behind every official
# number in the paper. SPICE needs Java 11 on PATH; Java 21 breaks the fst-2.47 jar.
set -euo pipefail
cd "$(dirname "$0")"
URL=https://github.com/boreng0817/IFCap/releases/download/v1.0/evaluation.zip
SHA=3fd2d5090f1637be45de0870679792e2273b6acc5f3f7fc4a9bfdf3380afeebd
mkdir -p third_party
if [ ! -f third_party/evaluation/cocoeval.py ]; then
  [ -f third_party/evaluation.zip ] || curl -L --fail -o third_party/evaluation.zip "$URL"
  got=$( (sha256sum third_party/evaluation.zip 2>/dev/null || shasum -a 256 third_party/evaluation.zip) | cut -d' ' -f1)
  [ "$got" = "$SHA" ] || { echo "checksum mismatch for evaluation.zip: $got"; exit 1; }
  unzip -q third_party/evaluation.zip -d third_party
  rm third_party/evaluation.zip
fi
python3 -c "import matplotlib, skimage, ipdb" 2>/dev/null || echo "note: the toolkit imports matplotlib, scikit-image and ipdb; pip install -r requirements.txt"
java -version 2>&1 | head -1 | grep -q '"11\.' || echo "note: SPICE expects Java 11, found: $(java -version 2>&1 | head -1)"
echo "toolkit ready at third_party/evaluation"
