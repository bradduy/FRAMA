#!/bin/bash
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; PY=$IF/rap_venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
cd $V/revision
C=$IF/inference_result/coco-indomain-004_biggFull/coco_beam_dump.json; NC=$IF/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json; FL=$IF/inference_result/flickr-indomain-bigg-blip/flickr30k_beam_dump.json
[ -f xh_coco.json ] || $PY xref_human_ci.py --beam_dump $C --preds_dir $V/preds/coco --out xh_coco.json > xh_coco.log 2>&1 &
[ -f xh_nocaps.json ] || $PY xref_human_ci.py --beam_dump $NC --preds_dir $V/preds/nocaps --out xh_nocaps.json > xh_nocaps.log 2>&1 &
[ -f xh_flickr.json ] || $PY xref_human_ci.py --beam_dump $FL --preds_dir $V/preds/flickr --out xh_flickr.json > xh_flickr.log 2>&1 &
[ -f xh_coco_spice.json ] || $PY xref_human_ci.py --beam_dump $C --preds_dir $V/preds/coco --out xh_coco_spice.json --spice --stride 5 > xh_coco_spice.log 2>&1 &
wait
