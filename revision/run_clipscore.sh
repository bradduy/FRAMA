V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; PY=$IF/rap_venv/bin/python; export HF_HOME=~/ZeroShotImageCap/hf_cache
cd $V/revision
R="lm_top1 blip_top1 borda3_lm_bigG_blip zsum3_lm_bigG_blip stack4_zsum3_vqa stack4_zsum3_mbr zsum3_plus_ground_best stack6_all_equal stack5_noLM_blip_bigG_vqa_mbr_ground oracle20_approx blip2cap"
for B in coco nocaps; do
  IMG=$IF/annotations/coco/val2014; [ $B = nocaps ] && IMG=$IF/annotations/nocaps/images
  [ -f clipscore_$B.json ] || $PY clipscore_eval.py --image_dir $IMG --preds $(for r in $R; do echo $V/preds/$B/${r}__overall.json; done) --out clipscore_$B.json --device cuda:1 > clipscore_$B.log 2>&1
done
