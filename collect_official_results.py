"""Collect results_official/*.txt (pulled from GCP2) into a markdown table.

File names: <bench>__<rule>__<split>.txt, each holding lines like "CIDEr: 1.173".
"""
import glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_official")
ORDER = ["lm_top1", "bigG_top1", "blip_top1", "zmix_lm_bigG_a0.48", "zsum3_lm_bigG_blip",
         "borda3_lm_bigG_blip", "borda4_with_length", "zsum3_plus_ground_best",
         "vqasub_zsum3", "vqasub_blip_top1", "vqasub_vqa_alone", "vqasub_zsum3_vqa_w1.0",
         "vqasub_zsum3_vqa_w2.0", "vqasub_blip_vqa", "vqasub_oracle20_approx",
         "stack4_zsum3_vqa", "stack4_zsum3_mbr", "stack5_zsum3_vqa_mbr", "stack6_all_equal",
         "stack4_noLM_blip_bigG_vqa_mbr", "stack5_noLM_blip_bigG_vqa_mbr_ground", "blip_vqa_mbr_ground",
         "ladder_zsum3_bigG2siglip2", "ladder_zsum3_plus_siglip2", "ladder_zsum3_plus_blip2itm",
         "ladder_stack6_bigG2siglip2", "ladder_stack6_plus_blip2itm", "ladder_stack8_all",
         "c4_beam40_zsum3", "c4_beam40_stack6", "c4_beam40_stack5_noLM", "c4_beam80_zsum3", "c4_beam80_stack6", "c4_beam80_stack5_noLM", "oracle20_approx"]
rows = {}
for p in sorted(glob.glob(os.path.join(RES, "*.txt"))):
    name = os.path.basename(p)[:-4]
    if name.startswith("_"):
        continue
    bench, rule, split = name.split("__")
    m = dict(re.findall(r"^(Bleu_4|METEOR|ROUGE_L|CIDEr|SPICE): ([0-9.]+)", open(p).read(), re.M))
    rows[(bench, split, rule)] = m
print("| bench | split | rule | B-4 | METEOR | ROUGE-L | CIDEr | SPICE |")
print("|---|---|---|---|---|---|---|---|")
for bench in ("coco", "flickr", "nocaps", "coco_viecap", "coco_ifcapsub"):
    for split in ("overall", "in_domain", "near_domain", "out_domain"):
        for rule in ORDER:
            m = rows.get((bench, split, rule))
            if not m:
                continue
            f = lambda k: f"{100*float(m[k]):.1f}" if k in m else "-"
            print(f"| {bench} | {split} | {rule} | {f('Bleu_4')} | {f('METEOR')} | {f('ROUGE_L')} | **{f('CIDEr')}** | {f('SPICE')} |")
