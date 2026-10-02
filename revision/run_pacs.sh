#!/bin/bash
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; PY=$IF/rap_venv/bin/python
cd ~/ZeroShotImageCap
[ -d pacscore ] || git clone -q https://github.com/aimagelab/pacscore.git
cd pacscore; mkdir -p checkpoints
[ -s checkpoints/PAC++_clip_ViT-L-14.pth ] || wget -q -O checkpoints/PAC++_clip_ViT-L-14.pth https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-L-14.pth
R="lm_top1 blip_top1 borda3_lm_bigG_blip zsum3_lm_bigG_blip stack4_zsum3_vqa stack6_all_equal stack5_noLM_blip_bigG_vqa_mbr_ground oracle20_approx blip2cap"
for B in coco nocaps; do
  IMG=$IF/annotations/coco/val2014; [ $B = nocaps ] && IMG=$IF/annotations/nocaps/images
  [ -f $V/revision/pacs_$B.json ] && [ $(grep -c '"n"' $V/revision/pacs_$B.json) -ge 9 ] || $PY $V/revision/pacs_eval.py --image_dir $IMG --preds $(for r in $R; do echo $V/preds/$B/${r}__overall.json; done) --out $V/revision/pacs_$B.json > $V/revision/pacs_$B.log 2>&1
done
