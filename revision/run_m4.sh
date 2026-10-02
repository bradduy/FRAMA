#!/bin/bash
# Reviewer item M4: score the BLIP-2 caption as a 21st candidate with every image agent.
# One-candidate dumps are built from the BLIP-2 prediction files; features, System One verifier and
# grounding are then computed exactly as for the beam. Resumable per step.
bench=$1; dev=$2
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; PY=$IF/rap_venv/bin/python
export HF_HOME=~/ZeroShotImageCap/hf_cache
VOC=$IF/annotations/vocabulary/vgcocooiobjects_v1_class2ind.json
if [ "$bench" = coco ]; then BD=$IF/inference_result/coco-indomain-004_biggFull/coco_beam_dump.json; IMG=$IF/annotations/coco/val2014
else BD=$IF/inference_result/nocaps-bigg-blip/nocaps_beam_dump.json; IMG=$IF/annotations/nocaps/images; fi
R=$V/revision; D=$R/m4_${bench}_blip2_dump.json
[ -f $D ] || $PY - <<PYEOF
import json
beams=json.load(open("$BD")); p={r["image_name"]:r["prediction"] for r in json.load(open("$V/preds/$bench/blip2cap__overall.json"))}
out=[{"image_name":b["image_name"],"split":b.get("split"),"captions":b["captions"],"beam":[{"caption":p[b["image_name"]],"score":0.0}]} for b in beams]
json.dump(out,open("$D","w")); print("dump",len(out))
PYEOF
[ -f $R/m4_${bench}_features.npz ] || ( cd $IF && $PY src/csfn_extract_features.py --beam_dump $D --image_folder $IMG --out_npz $R/m4_${bench}_features.npz --device $dev > $R/m4_${bench}_features.log 2>&1 )
cd $V
n=$(wc -l < $R/m4_${bench}_vqa.jsonl 2>/dev/null || echo 0); [ $n -ge $(python3 -c "import json;print(len(json.load(open('$D'))))") ] || $PY blip2vqa_scorer.py --beam_dump $D --image_dir $IMG --out_jsonl $R/m4_${bench}_vqa.jsonl --device $dev > $R/m4_${bench}_vqa.log 2>&1
n=$(wc -l < $R/m4_${bench}_ground.jsonl 2>/dev/null || echo 0); [ $n -ge $(python3 -c "import json;print(len(json.load(open('$D'))))") ] || $PY detector_ground_scorer.py --beam_dump $D --image_dir $IMG --out_jsonl $R/m4_${bench}_ground.jsonl --entity_vocab $VOC --device $dev > $R/m4_${bench}_ground.log 2>&1
echo "M4_${bench}_DONE" > $R/m4_${bench}.done
