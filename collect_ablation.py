"""Official leave-one-out and weight-transfer table from results_official/*.txt.
Usage: python3 collect_ablation.py > ablation_table_2026-09-23.md"""
import glob, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results_official")
def read(bench, rule, split="overall"):
    p = os.path.join(R, f"{bench}__{rule}__{split}.txt")
    if not os.path.exists(p): return None
    m = {}
    for line in open(p):
        mm = re.match(r"^(Bleu_4|METEOR|ROUGE_L|CIDEr|SPICE):\s*([0-9.]+)", line)
        if mm: m[mm.group(1)] = 100 * float(mm.group(2))
    return m if "CIDEr" in m else None
AG = ["LM", "bigG", "BLIP", "VQA", "MBR", "ground"]
LOO = {"LM": "stack5_noLM_blip_bigG_vqa_mbr_ground"}
for a in AG[1:]: LOO[a] = f"loo_minus_{a}"
print("## Leave-one-out from the equal six-agent stack, official CIDEr / SPICE (delta vs six-agent)\n")
print("| Removed agent | COCO | NoCaps overall | NoCaps out | Flickr30k |")
print("|---|---|---|---|---|")
cells = {}
for bench, split in (("coco", "overall"), ("nocaps", "overall"), ("nocaps", "out_domain"), ("flickr", "overall")):
    base = read(bench, "stack6_all_equal", split)
    cells[(bench, split, "base")] = base
    for a in AG:
        r = read(bench, LOO[a], split)
        cells[(bench, split, a)] = (r, base)
def fmt(x):
    if x is None or x[0] is None or x[1] is None: return "pending"
    r, b = x
    return f"{r['CIDEr']:.1f} / {r['SPICE']:.1f} ({r['CIDEr']-b['CIDEr']:+.1f})"
row = ["none (six-agent)"] + [f"{cells[(b,s,'base')]['CIDEr']:.1f} / {cells[(b,s,'base')]['SPICE']:.1f}" if cells[(b,s,'base')] else "pending" for b, s in (("coco","overall"),("nocaps","overall"),("nocaps","out_domain"),("flickr","overall"))]
print("| " + " | ".join(row) + " |")
for a in AG:
    row = [f"minus {a}"] + [fmt(cells[(b, s, a)]) for b, s in (("coco","overall"),("nocaps","overall"),("nocaps","out_domain"),("flickr","overall"))]
    print("| " + " | ".join(row) + " |")
print("\n## Weight transfer: best Dirichlet-sweep weights of one benchmark applied to the others, official CIDEr (delta vs equal weights)\n")
print("| Weights fitted on | COCO | NoCaps overall | NoCaps out | Flickr30k |")
print("|---|---|---|---|---|")
for src in ("coco", "nocaps"):
    row = [src]
    for b, s in (("coco","overall"),("nocaps","overall"),("nocaps","out_domain"),("flickr","overall")):
        if b == src: row.append("in-sample, not reported"); continue
        r = read(b, f"wtransfer_from_{src}", s); base = cells[(b, s, "base")]
        row.append(f"{r['CIDEr']:.1f} ({r['CIDEr']-base['CIDEr']:+.1f})" if r and base else "pending")
    print("| " + " | ".join(row) + " |")
