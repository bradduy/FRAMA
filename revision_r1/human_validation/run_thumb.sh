#!/bin/bash
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; H=$V/revision_r1/human_bench; PY=$IF/rap_venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
cd $H
[ -f thumb_cider.npz ] || $PY thumb_official.py --thumb_dir thumb_data --out thumb_cider > thumb_cider.log 2>&1
if [ ! -f thumb_chair.json ]; then
  A=~/ZeroShotImageCap/aic_repro/assets/chair_official
  S=""; for n in Up-Down Unified-VLP VinVL-base VinVL-large Human; do S="$S --system $n=$H/thumb_preds/${n}__overall.json"; done
  (cd ~/ZeroShotImageCap/aic_repro && $A/venv/bin/python $V/revision_r1/chair_coverage.py $S --annotation-root $A/annotations \
     --chair-source-root $A/sources/Hallucination-6e4d33c4bedf --pattern-source-root $A/sources/pattern-5b85d998c30d \
     --nltk-data-root $A/nltk_data --manifest manifests/chair_official_sources.json --output $H/thumb_chair.json) > thumb_chair.log 2>&1
fi
