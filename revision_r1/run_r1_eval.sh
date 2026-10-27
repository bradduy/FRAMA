#!/bin/bash
# R1 revision: official evaluation of the rebuilt selections in preds_r1/ (corrected MBR agent,
# fixed-rule grounding row). Three lanes; every step skips finished outputs, so the queue can be
# relaunched after a pre-emption. No completion marker is written by the shell.
set -u
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; R=$V/revision_r1; P=$V/preds_r1
PY=$IF/rap_venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
mkdir -p $R/perimage_r1 $R/results_r1 $R/logs
cd $R

lane_perimage() {
  for b in coco nocaps flickr coco_viecap_full; do
    tag=$b; [ $b = coco_viecap_full ] && tag=viecap
    $PY perimage_official.py --out_dir perimage_r1 --bench $tag --preds $P/$b/*__overall.json >> logs/pi_r1_$tag.log 2>&1
  done
  $PY perimage_official.py --out_dir perimage_r1 --bench nocaps --no_spice \
     --preds $P/nocaps/*__in_domain.json $P/nocaps/*__near_domain.json $P/nocaps/*__out_domain.json >> logs/pi_r1_nocaps_splits.log 2>&1
}
lane_cocoeval() {
  cd $IF/evaluation
  for b in coco nocaps flickr coco_viecap_full; do
    for n in stack6_all_equal stack5_noLM_blip_bigG_vqa_mbr_ground stack4_zsum3_mbr stack5_zsum3_vqa_mbr stack4_zsum3_ground \
             zsum3_lm_bigG_blip lm_top1 blip_top1 stack4_zsum3_vqa itmflickr_stack6_all_equal itmflickr_stack5_noLM_blip_bigG_vqa_mbr_ground m4_five21 m4_five20; do
      f=$P/$b/${n}__overall.json; [ -f $f ] || continue
      out=$R/results_r1/${b}__${n}__overall.txt
      if [ -f $out ] && grep -q "^SPICE:" $out; then continue; fi
      $PY cocoeval.py --result_file_path $f > $out.partial 2>&1 && grep -q "^SPICE:" $out.partial && mv $out.partial $out
    done
  done
  cd $R
}
lane_chair() {
  [ -f chair_coverage_r1_coco.json ] && return
  A=~/ZeroShotImageCap/aic_repro/assets/chair_official; C=$P/coco
  S=""; for n in lm_top1 blip_top1 borda3_lm_bigG_blip zsum3_lm_bigG_blip stack4_zsum3_vqa stack4_zsum3_ground stack4_zsum3_mbr \
                 stack5_zsum3_vqa_mbr stack6_all_equal stack5_noLM_blip_bigG_vqa_mbr_ground oracle20_approx itmflickr_stack6_all_equal \
                 loo_minus_bigG loo_minus_BLIP loo_minus_VQA loo_minus_MBR loo_minus_ground; do S="$S --system $n=$C/${n}__overall.json"; done
  (cd ~/ZeroShotImageCap/aic_repro && $A/venv/bin/python $R/chair_coverage.py $S --annotation-root $A/annotations \
     --chair-source-root $A/sources/Hallucination-6e4d33c4bedf --pattern-source-root $A/sources/pattern-5b85d998c30d \
     --nltk-data-root $A/nltk_data --manifest manifests/chair_official_sources.json --output $R/chair_coverage_r1_coco.json) > logs/chair_r1.log 2>&1
}
lane_perimage & lane_cocoeval & lane_chair & wait
echo "R1_EVAL_QUEUE_EXIT $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> logs/driver.log
