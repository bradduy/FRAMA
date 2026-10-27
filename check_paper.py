"""Print the main tables of the revised manuscript from the released result files and compare each
value with the number printed in the paper. Needs only Python 3 and numpy; runs in a few seconds.

  python3 check_paper.py            # every table below
  python3 check_paper.py 2 12       # selected tables

Sources, all under reference_results/:
  official_metrics_r1/, official_metrics/   official COCO caption toolkit output per prediction file
  r1/perimage/<bench>__<rule>[__<split>].json   official per-image CIDEr-D and SPICE; their means are the table values
  r1/r1_numbers.json                        cross-reference oracle, CHAIR and paired bootstrap (revision_r1/analyze_r1.py)
  r1/e6_results.json                        factual reliability (revision_r1/factual_reliability/e6_analyze.py)
A value matches if it equals the printed one after rounding to the printed precision; derived rows of
Table 3 are printed in the paper as differences of rounded values, so they may differ by 0.1.
"""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "reference_results")
R1 = os.path.join(REF, "r1")
FAIL = []


def metric_file(bench, rule, split="overall"):
    for d in ("official_metrics_r1", "official_metrics"):
        p = os.path.join(REF, d, f"{bench}__{rule}__{split}.txt")
        if os.path.exists(p):
            out = {}
            for line in open(p):
                m = re.match(r"^(Bleu_4|METEOR|ROUGE_L|CIDEr|SPICE): ([0-9.]+)$", line.strip())
                if m:
                    out[m.group(1)] = 100 * float(m.group(2))
            return out
    return None


def perimage(bench, rule, split=None):
    name = f"{bench}__{rule}" + (f"__{split}" if split else "") + ".json"
    p = os.path.join(R1, "perimage", name)
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    return {"CIDEr": float(np.mean(r["cider"])), "SPICE": float(np.mean(r["spice"])) if r.get("spice") else None}


def cell(value, expected, decimals=1, tol=0.0):
    """Format a value and record a mismatch against the printed number."""
    if expected is None:
        return f"{'--':>7s}  "
    if value is None:
        FAIL.append(expected)
        return f"{'n/a':>7s} ✗"
    ok = abs(round(value, decimals) - expected) <= tol + 1e-9
    if not ok:
        FAIL.append((value, expected))
    return f"{value:7.{decimals}f} {'✓' if ok else '✗'}"


def header(title):
    print("\n" + title + "\n" + "-" * len(title))


# ------------------------------------------------------------------ Table 2
MAIN = [  # rule, label, COCO B-4 M R-L C S, NoCaps in near out overall S, Flickr C S (as printed)
    ("lm_top1", "Captioner top-1 (LM)", [31.4, 27.2, 54.6, 112.7, 21.0], [65.6, 69.0, 68.5, 70.3, 9.8], [63.7, 17.0]),
    ("blip_top1", "BLIP-ITM top-1", [32.1, 27.8, 55.3, 114.3, 22.0], [76.2, 79.3, 78.1, 80.3, 11.1], [66.7, 18.2]),
    ("borda3_lm_bigG_blip", "Borda count of 3 scores", [32.8, 27.9, 55.6, 117.3, 21.9], [75.2, 78.2, 77.5, 79.4, 10.9], [None, None]),
    ("zsum3_lm_bigG_blip", "Three-agent z-sum", [33.3, 28.0, 55.8, 118.6, 22.0], [76.1, 78.7, 78.1, 79.9, 10.9], [68.2, 18.1]),
    ("stack4_zsum3_vqa", "z-sum + System One", [33.6, 28.0, 56.2, 119.6, 22.2], [78.0, 79.8, 79.3, 81.2, 11.0], [70.1, 18.0]),
    ("stack4_zsum3_mbr", "z-sum + consensus", [33.4, 28.0, 55.9, 118.9, 22.0], [75.8, 78.6, 77.2, 79.6, 10.8], [68.8, 18.1]),
    ("stack5_zsum3_vqa_mbr", "z-sum + System One + consensus", [33.9, 28.1, 56.4, 120.3, 22.1], [76.1, 79.7, 78.7, 80.8, 10.9], [69.4, 18.1]),
    ("stack6_all_equal", "FRAMA, six agents", [34.3, 28.2, 56.5, 120.7, 22.2], [78.0, 80.0, 79.0, 81.2, 11.0], [70.7, 18.3]),
    ("stack5_noLM_blip_bigG_vqa_mbr_ground", "FRAMA, five agents, no LM", [34.1, 28.0, 56.3, 120.5, 22.2], [77.4, 81.4, 79.7, 82.2, 11.0], [72.3, 18.5]),
    ("oracle20_approx", "Reference-selected Oracle@20", [51.1, 33.1, 64.3, 162.1, 25.3], [104.0, 103.6, 100.7, 104.7, 12.4], [100.0, 20.3]),
]


def table2():
    header("Table 2. Main results, official COCO caption toolkit (value, ✓ = equals the printed number)")
    print(f"{'Selector':32s} {'B-4':>9s} {'M':>9s} {'R-L':>9s} {'CIDEr':>9s} {'SPICE':>9s} | NoCaps {'in':>7s} {'near':>9s} "
          f"{'out':>9s} {'overall':>9s} {'SPICE':>9s} | Flickr {'CIDEr':>7s} {'SPICE':>9s}")
    for rule, label, coco, noc, fl in MAIN:
        m = metric_file("coco", rule) or {}
        c = [cell(m.get(k), e) for k, e in zip(("Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"), coco)]
        n = [cell((perimage("nocaps", rule, s) or {}).get("CIDEr"), e) for s, e in zip(("in_domain", "near_domain", "out_domain"), noc[:3])]
        po = perimage("nocaps", rule) or {}
        n += [cell(po.get("CIDEr"), noc[3]), cell(po.get("SPICE"), noc[4])]
        pf = perimage("flickr", rule) or {}
        f = [cell(pf.get("CIDEr"), fl[0]), cell(pf.get("SPICE"), fl[1])]
        print(f"{label:32s} " + " ".join(c) + " |        " + " ".join(n) + " |        " + " ".join(f))
    print("Adjudicated Captioning, the published baseline, is copied from its paper and is not recomputed here.")


# ------------------------------------------------------------------ Table 3
XREF = {  # printed rows: COCO, NoCaps, Flickr30k, ViECap
    "lm_top1": [110.2, 68.1, 67.1, 89.0], "zsum3_lm_bigG_blip": [116.1, 77.4, 71.5, 103.2],
    "stack6_all_equal": [118.3, 78.7, 74.1, 104.3], "stack5_noLM_blip_bigG_vqa_mbr_ground": [118.0, 79.6, 75.5, 104.6],
    "Human reference": [87.1, 84.0, 74.9, 87.1], "XRef@20": [123.2, 88.7, 80.9, 109.8], "Oracle@20 pick": [161.0, 108.4, 103.6, 144.7],
}
XREF_D = {"headroom": [7.1, 11.3, 9.4, 6.6], "apparent": [44.9, 31.0, 32.1, 41.5], "share": [16, 36, 29, 16]}


def table3():
    header("Table 3. Leakage-free evaluation, CIDEr against one held-out reference, mean of five folds")
    x = json.load(open(os.path.join(R1, "r1_numbers.json")))["xref"]
    benches = ["coco", "nocaps", "flickr", "viecap"]
    print(f"{'Row':40s} " + " ".join(f"{b:>9s}" for b in benches))
    for k, exp in XREF.items():
        print(f"{k:40s} " + " ".join(cell(x[b]["means"][k], e) for b, e in zip(benches, exp)))
    for k, exp in XREF_D.items():
        scale = 100 if k == "share" else 1
        print(f"{k:40s} " + " ".join(cell(scale * x[b][k], e, 0 if k == "share" else 1, 0 if k == "share" else 0.1)
                                      for b, e in zip(benches, exp)))
    print("Bootstrap intervals of the headroom and the share: r1_numbers.json, keys ci_headroom and ci_share.")


# ------------------------------------------------------------------ Table 12
CHAIR = [("lm_top1", "Captioner top-1", 6.66, 4.57, 42.7), ("blip_top1", "BLIP-ITM top-1", 3.88, 2.72, 43.5),
         ("borda3_lm_bigG_blip", "Borda count of 3 scores", 4.16, 2.87, 43.0), ("zsum3_lm_bigG_blip", "Three-agent z-sum", 3.94, 2.73, 43.0),
         ("stack4_zsum3_vqa", "  + System One", 4.04, 2.79, 43.2), ("stack4_zsum3_ground", "  + grounding", 3.54, 2.45, 43.3),
         ("stack4_zsum3_mbr", "  + consensus", 4.46, 3.06, 43.3), ("stack6_all_equal", "FRAMA, six agents", 3.72, 2.54, 43.5),
         ("stack5_noLM_blip_bigG_vqa_mbr_ground", "FRAMA, five agents, no LM", 3.64, 2.48, 43.6),
         ("oracle20_approx", "Reference-selected Oracle@20", 5.44, 3.60, 44.7)]


def table12():
    header("Table 12. CHAIR on COCO, original implementation, percent")
    ch = json.load(open(os.path.join(R1, "r1_numbers.json")))["chair"]
    print(f"{'Selector':32s} {'CHAIR_s':>9s} {'CHAIR_i':>9s} {'Cov.':>9s}")
    for rule, label, s, i, cov in CHAIR:
        r = ch[rule]
        print(f"{label:32s} {cell(r['CHAIRs'], s, 2)} {cell(r['CHAIRi'], i, 2)} {cell(r['coverage'], cov)}")


# ------------------------------------------------------------------ Tables 15 and 16
FACT_RULES = [("lm_top1", "Captioner top-1"), ("borda3_lm_bigG_blip", "Borda proxy"), ("zsum3_lm_bigG_blip", "Three-agent z-sum"),
              ("stack6_all_equal", "FRAMA, six agents"), ("stack5_noLM_blip_bigG_vqa_mbr_ground", "FRAMA, five agents, no LM")]
SPICE_EXP = {"lm_top1": [73.1, 24.9, 42.0, 7.4, 18.2, 3.7], "borda3_lm_bigG_blip": [75.9, 25.6, 45.7, 8.5, 19.3, 4.0],
             "zsum3_lm_bigG_blip": [76.1, 25.5, 46.3, 8.5, 19.6, 4.0], "stack6_all_equal": [77.1, 25.8, 48.1, 8.1, 19.9, 4.1],
             "stack5_noLM_blip_bigG_vqa_mbr_ground": [77.9, 25.8, 48.0, 8.1, 20.1, 4.1]}
UNSUP_EXP = {"lm_top1": [20.2, 25.6, 37.6, 18.9, 30.4, 33.9, 37.7, 27.7], "borda3_lm_bigG_blip": [14.1, 17.6, 26.1, 10.4, 21.1, 24.7, 27.5, None],
             "zsum3_lm_bigG_blip": [13.5, 17.2, 26.1, 8.8, 20.4, 24.0, 26.1, 20.3], "stack6_all_equal": [12.9, 16.2, 24.4, 8.9, 20.8, 22.8, 25.9, 20.2],
             "stack5_noLM_blip_bigG_vqa_mbr_ground": [11.9, 14.7, 23.1, 8.0, 18.8, 21.6, 24.5, 19.8]}
SUPP_EXP = {"lm_top1": 4.71, "borda3_lm_bigG_blip": 5.11, "zsum3_lm_bigG_blip": 5.09, "stack6_all_equal": 5.05,
            "stack5_noLM_blip_bigG_vqa_mbr_ground": 5.07}
FOIL_EXP = {  # dual, cross, S1, consensus, detector, dual+cross, five
    "SugarCrepe replace object": [97.4, 97.4, 92.6, 82.0, 87.0, 97.9, 96.8], "SugarCrepe replace attribute": [85.0, 90.0, 89.0, 68.0, 62.0, 91.0, 94.0],
    "SugarCrepe replace relation": [78.3, 81.9, 77.7, 74.7, 52.4, 81.3, 87.3], "SugarCrepe swap object": [54.5, 72.7, 69.7, 51.5, 51.5, 63.6, 75.8],
    "SugarCrepe swap attribute": [80.3, 90.1, 76.1, 72.5, 47.2, 93.0, 88.7], "SugarCrepe add object": [95.7, 97.2, 73.5, 90.0, 93.6, 98.6, 97.2],
    "SugarCrepe add attribute": [89.0, 90.2, 90.2, 97.6, 66.5, 92.7, 97.6], "VALSE existence": [83.3, 91.7, 50.0, 58.3, 50.0, 83.3, 66.7],
    "VALSE plurality": [66.7, 80.6, 62.0, 66.7, 50.5, 75.0, 75.0], "VALSE counting": [58.7, 67.4, 56.5, 51.1, 48.9, 63.0, 60.9],
    "VALSE spatial relations": [63.3, 68.3, 81.7, 68.3, 49.2, 65.0, 83.3], "VALSE noun foils (FOIL-it)": [93.3, 96.7, 91.7, 89.2, 93.3, 96.7, 97.5],
}
FOIL_SEL = ["bigG", "BLIP-ITM", "System One verifier", "consensus", "grounding", "z-sum bigG + BLIP-ITM", "five-agent FRAMA"]


def table15():
    header("Table 15. Factual precision beyond objects: SPICE P / R per category on COCO; unsupported atomic facts, percent")
    e = json.load(open(os.path.join(R1, "e6_results.json")))
    a = e["E6a"]["coco"]["rules"]
    print(f"{'Selector':28s} {'Obj P':>9s} {'Obj R':>9s} {'Attr P':>9s} {'Attr R':>9s} {'Rel P':>9s} {'Rel R':>9s}")
    for rule, label in FACT_RULES:
        r = a[rule]; exp = SPICE_EXP[rule]
        vals = [r["Object"]["P"], r["Object"]["R"], r["Attribute"]["P"], r["Attribute"]["R"], r["Relation"]["P"], r["Relation"]["R"]]
        print(f"{label:28s} " + " ".join(cell(v, x) for v, x in zip(vals, exp)))
    types = ["all facts", "attribute", "color", "count", "spatial relation", "action"]
    print(f"\n{'Selector':28s} " + " ".join(f"{t[:9]:>9s}" for t in types) + f" {'supp/cap':>9s} {'NoCaps':>9s} {'Flickr':>9s}")
    for rule, label in FACT_RULES:
        exp = UNSUP_EXP[rule]
        c = e["E6c"]["coco"]["rules"][rule]
        vals = [c[t] for t in types]
        noc = e["E6c"]["nocaps"]["rules"].get(rule, {}).get("all facts")
        fl = e["E6c"]["flickr"]["rules"].get(rule, {}).get("all facts")
        print(f"{label:28s} " + " ".join(cell(v, x) for v, x in zip(vals, exp[:6])) + " "
              + cell(c["supported facts per caption"], SUPP_EXP[rule], 2) + " " + cell(noc, exp[6]) + " " + cell(fl, exp[7]))
    v = e["E6c_validation"]
    print("\nVerifier validation on THumB, Kendall tau with human precision: verifier "
          f"{v['verifier share of supported facts']['kendall_tau']:.3f}, CHAIR_s {v['CHAIR_s, negated']['kendall_tau']:.3f}, "
          f"SPICE precision {v['SPICE precision against the references']['kendall_tau']:.3f}")


def table16():
    header("Table 16. Human-validated foils, accuracy in percent (1,198 items, 439 COCO test images)")
    e = json.load(open(os.path.join(R1, "e6_results.json")))["E6b"]
    print(f"{'Phenomenon':30s} " + " ".join(f"{s[:9]:>9s}" for s in ("Dual", "Cross", "S1", "Cons.", "Det.", "D+C", "Five")))
    for g, exp in FOIL_EXP.items():
        r = e["groups"][g]
        print(f"{g:30s} " + " ".join(cell(r[s]["acc"], x) for s, x in zip(FOIL_SEL, exp)))
    print(f"{'All items':30s} " + " ".join(cell(e["all_items"][s]["acc"], x)
                                         for s, x in zip(FOIL_SEL, [84.1, 88.7, 79.9, 78.1, 70.2, 88.2, 90.1])))


# ------------------------------------------------------------------ Table 17
TRANSFER = [("lm_top1", "Captioner top-1", 91.2, 18.1, 112.7, 21.0), ("blip_top1", "BLIP-ITM top-1", 103.6, 20.5, 114.3, 22.0),
            ("zsum3_lm_bigG_blip", "Three-agent z-sum", 105.6, 20.5, 118.6, 22.0), ("stack4_zsum3_vqa", "  + System One", 107.0, 20.8, 119.6, 22.2),
            ("stack4_zsum3_mbr", "  + consensus", 105.2, 20.3, 118.9, 22.0), ("stack6_all_equal", "FRAMA, six agents", 106.7, 20.4, 120.7, 22.2),
            ("stack5_noLM_blip_bigG_vqa_mbr_ground", "FRAMA, five agents, no LM", 107.0, 20.5, 120.5, 22.2),
            ("oracle20_approx", "Reference-selected Oracle@20", 145.8, 23.6, 162.1, 25.3)]


def table17():
    header("Table 17. Transfer to a second captioner, full COCO Karpathy test, CIDEr / SPICE")
    print(f"{'Selector':32s} {'ViECap C':>9s} {'S':>9s} {'IFCap C':>9s} {'S':>9s}")
    for rule, label, vc, vs, ic, isp in TRANSFER:
        v = perimage("viecap", rule) or {}
        i = perimage("coco", rule) or {}
        print(f"{label:32s} {cell(v.get('CIDEr'), vc)} {cell(v.get('SPICE'), vs)} {cell(i.get('CIDEr'), ic)} {cell(i.get('SPICE'), isp)}")


TABLES = {"2": table2, "3": table3, "12": table12, "15": table15, "16": table16, "17": table17}

if __name__ == "__main__":
    for t in (sys.argv[1:] or TABLES):
        TABLES[t]()
    print(f"\n{'All values match the paper.' if not FAIL else f'{len(FAIL)} value(s) differ from the paper.'}")
    sys.exit(1 if FAIL else 0)
