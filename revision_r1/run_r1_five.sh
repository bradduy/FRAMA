#!/bin/bash
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; R=$V/revision_r1; PY=$IF/rap_venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
cd $R
$PY five_ref_matrix.py --beam_dump $IF/inference_result/coco-indomain-004_biggFull/coco_beam_dump.json --out matrices/coco > logs/five_coco.log 2>&1
$PY five_ref_matrix.py --beam_dump $IF/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json --out matrices/nocaps > logs/five_nocaps.log 2>&1
$PY five_ref_matrix.py --beam_dump $IF/inference_result/flickr-indomain-bigg-blip/flickr30k_beam_dump.json --out matrices/flickr > logs/five_flickr.log 2>&1
$PY five_ref_matrix.py --beam_dump $V/c5/viecap_coco_full_beam_dump.json --out matrices/viecap_full > logs/five_viecap.log 2>&1
$PY five_ref_matrix.py --beam_dump $IF/inference_result/c4_expand/coco_beam40_dump.json --out matrices/coco_beam40 > logs/five_b40.log 2>&1
$PY five_ref_matrix.py --beam_dump $IF/inference_result/c4_expand/coco_beam80_dump.json --out matrices/coco_beam80 > logs/five_b80.log 2>&1
