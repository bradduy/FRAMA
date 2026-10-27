"""Compare official metrics in results/ (from evaluate_official.sh) with the numbers
reported in the paper, stored in reference_results/official_metrics/.

Usage: python3 compare_results.py [results_dir] [reference folder under reference_results/]
  e.g. python3 compare_results.py results_r1 official_metrics_r1     for the revised paper
Prints one row per prediction file present in both folders and the largest absolute
difference per metric. Differences of a few 0.1 CIDEr can come from a different Java
or tokenizer build; prediction files themselves are deterministic.
"""
import glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
REF = os.path.join(HERE, "reference_results", sys.argv[2] if len(sys.argv) > 2 else "official_metrics")
KEYS = ("Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE")


def read(path):
    return {k: 100 * float(v) for k, v in re.findall(r"^(Bleu_4|METEOR|ROUGE_L|CIDEr|SPICE): ([0-9.]+)", open(path).read(), re.M)}


worst = {k: 0.0 for k in KEYS}
rows = 0
print(f"{'file':60s} " + " ".join(f"{k:>16s}" for k in KEYS))
for path in sorted(glob.glob(os.path.join(RES, "*.txt"))):
    name = os.path.basename(path)
    ref_path = os.path.join(REF, name)
    if not os.path.exists(ref_path):
        continue
    ours, ref = read(path), read(ref_path)
    cells = []
    for k in KEYS:
        if k in ours and k in ref:
            d = ours[k] - ref[k]; worst[k] = max(worst[k], abs(d))
            cells.append(f"{ours[k]:6.1f} ({d:+5.2f})")
        else:
            cells.append(f"{'-':>16s}")
    print(f"{name[:-4]:60s} " + " ".join(f"{c:>16s}" for c in cells))
    rows += 1
print(f"\n{rows} files compared; largest |difference| " + ", ".join(f"{k} {worst[k]:.2f}" for k in KEYS))
