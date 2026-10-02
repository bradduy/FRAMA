"""Collect the official CHAIR reports (chair_official/coco__<a>__vs__<b>.json, produced
on GCP2 by run_chair_remote.sh) into one markdown table, one row per selection rule.
Usage: python3 collect_chair.py [chair_official] > chair_table.md"""
import glob, json, os, sys
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "chair_official")
LABEL = {
    "lm_top1": "IFCap beam top-1 (LM)", "blip_top1": "BLIP-ITM top-1",
    "borda3_lm_bigG_blip": "Borda-3 LM+bigG+BLIP (v1 training-free proxy)",
    "zsum3_lm_bigG_blip": "z-sum LM+bigG+BLIP", "stack4_zsum3_vqa": "z-sum + BLIP-2 VQA",
    "zsum3_plus_ground_best": "z-sum + OWLv2 grounding", "stack4_zsum3_mbr": "z-sum + beam-MBR",
    "stack6_all_equal": "six-agent stack (proposed)", "stack5_noLM_blip_bigG_vqa_mbr_ground": "five-agent stack, no LM",
    "oracle20_approx": "Oracle@20 (reference-selected upper bound)",
}
ORDER = list(LABEL)
rows, status = {}, {}
for p in sorted(glob.glob(os.path.join(d, "coco__*__vs__*.json"))):
    r = json.load(open(p))
    a, b = os.path.basename(p)[len("coco__"):-len(".json")].split("__vs__")
    for slot, name in (("pcvr", a), ("matcher", b)):
        m = r["methods"][slot]; c = m["counts"]
        rows[name] = (m["CHAIR_s"], m["CHAIR_i"], c["hallucinated_sentences"], c["sentences"],
                      c["hallucinated_object_mentions"], c["object_mentions"])
        status[name] = r["status"]
print("| Selection rule | CHAIR_s (%) | CHAIR_i (%) | halluc. sentences / 5000 | halluc. mentions / mentions | status |")
print("|---|---|---|---|---|---|")
for k in ORDER + sorted(set(rows) - set(ORDER)):
    if k not in rows: continue
    s, i, hs, ns, hm, nm = rows[k]
    print(f"| {LABEL.get(k, k)} | {100*s:.2f} | {100*i:.2f} | {hs} / {ns} | {hm} / {nm} | {status[k]} |")
