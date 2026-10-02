#!/bin/bash
V=~/ZeroShotImageCap/macst_v1/v2_upgrade; IF=~/ZeroShotImageCap/IFCap; PY=$IF/rap_venv/bin/python; export HF_HOME=~/ZeroShotImageCap/hf_cache
cd $V/revision
D=$IF/inference_result/c4_expand/coco_beam20s_dump.json; IMG=$IF/annotations/coco/val2014
$PY -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-VL-2B-Instruct')" > cot_download.log 2>&1
$PY cot_judge_scorer.py --beam_dump $D --image_dir $IMG --out_jsonl cot_shard0.jsonl --device cuda:0 --shard 0 --nshard 2 > cot_shard0.log 2>&1 &
$PY cot_judge_scorer.py --beam_dump $D --image_dir $IMG --out_jsonl cot_shard1.jsonl --device cuda:1 --shard 1 --nshard 2 > cot_shard1.log 2>&1 &
wait
