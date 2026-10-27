#!/bin/bash
# R1 revision CPU queue on GCP2 (Access-2026-48860). Three lanes, every step skips
# finished outputs, so the queue can be relaunched after a pre-emption by the @reboot
# cron line. Completion of each lane is read from the files the python steps write.
set -u
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; R=$V/revision_r1
PY=$IF/rap_venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
mkdir -p $R/matrices $R/perimage $R/logs
cd $R

lane_matrices_a() {
  $PY official_matrices.py --beam_dump $IF/inference_result/coco-indomain-004_biggFull/coco_beam_dump.json --out matrices/coco > logs/mat_coco.log 2>&1
  $PY official_matrices.py --beam_dump $IF/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json --out matrices/nocaps > logs/mat_nocaps.log 2>&1
  $PY official_matrices.py --beam_dump $IF/inference_result/flickr-indomain-bigg-blip/flickr30k_beam_dump.json --out matrices/flickr > logs/mat_flickr.log 2>&1
  $PY official_matrices.py --beam_dump $V/c5/viecap_coco_full_beam_dump.json --out matrices/viecap_full > logs/mat_viecap.log 2>&1
}
lane_matrices_b() {
  $PY official_matrices.py --beam_dump $IF/inference_result/c4_expand/coco_beam40_dump.json --out matrices/coco_beam40 > logs/mat_beam40.log 2>&1
  $PY official_matrices.py --beam_dump $IF/inference_result/c4_expand/coco_beam80_dump.json --out matrices/coco_beam80 > logs/mat_beam80.log 2>&1
}
lane_perimage() {
  P=$V/preds
  COCO="stack6_all_equal stack5_noLM_blip_bigG_vqa_mbr_ground zsum3_lm_bigG_blip borda3_lm_bigG_blip lm_top1 blip_top1 blip2cap \
        itmflickr_stack6_all_equal itmflickr_stack5_noLM_blip_bigG_vqa_mbr_ground itmflickr_zsum3_lm_bigG_blip itmflickr_blip_top1 \
        stack4_zsum3_vqa stack4_zsum3_mbr zsum3_plus_ground_best stack5_zsum3_vqa_mbr oracle20_approx \
        loo_minus_bigG loo_minus_BLIP loo_minus_VQA loo_minus_MBR loo_minus_ground ladder_stack6_plus_blip2itm"
  NOC="stack6_all_equal stack5_noLM_blip_bigG_vqa_mbr_ground zsum3_lm_bigG_blip borda3_lm_bigG_blip lm_top1 blip_top1 blip2cap \
       stack4_zsum3_vqa oracle20_approx loo_minus_bigG loo_minus_BLIP loo_minus_VQA loo_minus_MBR loo_minus_ground"
  FLK="stack6_all_equal stack5_noLM_blip_bigG_vqa_mbr_ground zsum3_lm_bigG_blip lm_top1 blip_top1 blip2cap stack4_zsum3_vqa oracle20_approx \
       loo_minus_bigG loo_minus_BLIP loo_minus_VQA loo_minus_MBR loo_minus_ground"
  VIE="stack6_all_equal stack5_noLM_blip_bigG_vqa_mbr_ground zsum3_lm_bigG_blip lm_top1 blip_top1 stack4_zsum3_vqa oracle20_approx"
  $PY perimage_official.py --out_dir perimage --bench coco   --preds $(for n in $COCO; do echo $P/coco/${n}__overall.json; done) > logs/pi_coco.log 2>&1
  $PY perimage_official.py --out_dir perimage --bench nocaps --preds $(for n in $NOC; do echo $P/nocaps/${n}__overall.json; done) > logs/pi_nocaps.log 2>&1
  $PY perimage_official.py --out_dir perimage --bench flickr --preds $(for n in $FLK; do echo $P/flickr/${n}__overall.json; done) > logs/pi_flickr.log 2>&1
  $PY perimage_official.py --out_dir perimage --bench viecap --preds $(for n in $VIE; do echo $P/coco_viecap_full/${n}__overall.json; done) > logs/pi_viecap.log 2>&1
}
lane_chair() {
  [ -f chair_coverage_coco.json ] && return
  A=~/ZeroShotImageCap/aic_repro/assets/chair_official; P=$V/preds/coco
  S=""; for n in lm_top1 blip_top1 borda3_lm_bigG_blip zsum3_lm_bigG_blip stack4_zsum3_vqa zsum3_plus_ground_best stack4_zsum3_mbr \
                 stack6_all_equal stack5_noLM_blip_bigG_vqa_mbr_ground oracle20_approx blip2cap itmflickr_stack6_all_equal \
                 loo_minus_ground loo_minus_VQA loo_minus_MBR; do S="$S --system $n=$P/${n}__overall.json"; done
  (cd ~/ZeroShotImageCap/aic_repro && $A/venv/bin/python $R/chair_coverage.py $S --annotation-root $A/annotations \
     --chair-source-root $A/sources/Hallucination-6e4d33c4bedf --pattern-source-root $A/sources/pattern-5b85d998c30d \
     --nltk-data-root $A/nltk_data --manifest manifests/chair_official_sources.json --output $R/chair_coverage_coco.json) > logs/chair.log 2>&1
}
lane_matrices_a & lane_matrices_b & (lane_chair; lane_perimage) & wait
echo "R1_CPU_QUEUE_EXIT $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> logs/driver.log
