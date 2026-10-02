#!/bin/bash
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; PY=$IF/rap_venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
cd $V/revision
[ -f xref_sens_coco.json ] || $PY crossref_sensitivity.py --beam_dump $IF/inference_result/coco-indomain-004_biggFull/coco_beam_dump.json --out xref_sens_coco.json > xref_sens_coco.log 2>&1
[ -f xref_sens_nocaps.json ] || $PY crossref_sensitivity.py --beam_dump $IF/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json --out xref_sens_nocaps.json > xref_sens_nocaps.log 2>&1
F=$IF/inference_result/flickr-indomain-bigg-blip/flickr30k_beam_dump.json
[ -f xref_sens_flickr.json ] || $PY crossref_sensitivity.py --beam_dump $F --out xref_sens_flickr.json > xref_sens_flickr.log 2>&1
