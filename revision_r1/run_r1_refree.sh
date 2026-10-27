#!/bin/bash
# CLIP-S / RefCLIP-S and PAC-S++ for the rebuilt six- and five-agent selections (R1). GPU 1, small memory.
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; PY=$IF/rap_venv/bin/python; R=$V/revision_r1
export CUDA_VISIBLE_DEVICES=1 HF_HOME=~/ZeroShotImageCap/hf_cache
RULES="stack6_all_equal stack5_noLM_blip_bigG_vqa_mbr_ground"
for B in coco nocaps; do
  IMG=$IF/annotations/coco/val2014; [ $B = nocaps ] && IMG=$IF/annotations/nocaps/images
  [ -f $R/clipscore_r1_$B.json ] || $PY $V/revision/clipscore_eval.py --device cuda:0 --image_dir $IMG --preds $(for r in $RULES; do echo $V/preds_r1/$B/${r}__overall.json; done) --out $R/clipscore_r1_$B.json > $R/logs/clipscore_r1_$B.log 2>&1
done
cd ~/ZeroShotImageCap
[ -d pacscore ] || git clone -q https://github.com/aimagelab/pacscore.git
cd pacscore; mkdir -p checkpoints
[ -s checkpoints/PAC++_clip_ViT-L-14.pth ] || wget -q -O checkpoints/PAC++_clip_ViT-L-14.pth https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-L-14.pth
for B in coco nocaps; do
  IMG=$IF/annotations/coco/val2014; [ $B = nocaps ] && IMG=$IF/annotations/nocaps/images
  [ -f $R/pacs_r1_$B.json ] || $PY $V/revision/pacs_eval.py --image_dir $IMG --preds $(for r in $RULES; do echo $V/preds_r1/$B/${r}__overall.json; done) --out $R/pacs_r1_$B.json > $R/logs/pacs_r1_$B.log 2>&1
done
