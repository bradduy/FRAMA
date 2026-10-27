"""R1 revision analysis from official per-image scores (all inputs pulled from GCP2).

Inputs (revision_r1/):
  matrices/<bench>.npz        official per-candidate CIDEr-D matrices (official_matrices.py)
  perimage_r1/<bench>__<rule>.json   official per-image CIDEr-D and SPICE of the rebuilt selections
  chair_coverage_r1_coco.json per-image CHAIR counts and object coverage
  ../preds_r1/...             the rebuilt selection files
Outputs:
  r1_numbers.json             every number the revised manuscript quotes from this analysis
  ../paper_v2_ieeeaccess_r1/tables/tab_stats.tex
All bootstraps: 10,000 paired resamples of images, seed 0, 95% percentile intervals,
two-sided p-value = 2 * min(P(d <= 0), P(d >= 0)) over the resamples, floored at 1/B.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
sys.path.insert(0, V2); sys.path.insert(0, HERE)
PR = os.path.join(V2, "preds_r1"); PI = os.path.join(HERE, "perimage_r1"); MAT = os.path.join(HERE, "matrices")
TAB = os.path.join(V2, "paper_v2_ieeeaccess_r1", "tables")
B = 10000; RNG_SEED = 0
PD = {"coco": "coco", "nocaps": "nocaps", "flickr": "flickr", "viecap": "coco_viecap_full"}
NAME = {"coco": "COCO", "nocaps": "NoCaps", "flickr": "Flickr30k", "viecap": "COCO, ViECap beams"}
OUT = {}


def boot_idx(n, seed=RNG_SEED):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def pval(d):
    return float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / B))


def per_image(bench, rule):
    r = json.load(open(os.path.join(PI, f"{bench}__{rule}.json")))
    return r["image_names"], np.array(r["cider"]), (np.array(r["spice"]) if "spice" in r else None), r


def preds(bench, rule):
    return {x["image_name"]: x["prediction"] for x in json.load(open(os.path.join(PR, PD[bench], f"{rule}__overall.json")))}


def paired(bench, a, b, chair=None):
    """a minus b. Returns dict with CIDEr/SPICE deltas, CIs, p, changed share, gain if changed, CHAIRs delta."""
    na, ca, sa, _ = per_image(bench, a); nb, cb, sb, _ = per_image(bench, b)
    pos = {n: i for i, n in enumerate(nb)}; o = np.array([pos[n] for n in na]); cb = cb[o]; sb = sb[o] if sb is not None else None
    I = boot_idx(len(na))
    dc = ca - cb; bc = dc[I].mean(1)
    res = {"a": a, "b": b, "dCIDEr": float(dc.mean()), "ci_CIDEr": np.percentile(bc, [2.5, 97.5]).tolist(), "p": pval(bc)}
    if sa is not None and sb is not None:
        ds = sa - sb; bs = ds[I].mean(1)
        res.update({"dSPICE": float(ds.mean()), "ci_SPICE": np.percentile(bs, [2.5, 97.5]).tolist(), "p_SPICE": pval(bs)})
    pa, pb = preds(bench, a), preds(bench, b)
    ch = np.array([pa[n] != pb[n] for n in na])
    res["changed"] = float(ch.mean()); res["gain_if_changed"] = float(dc[ch].mean()) if ch.any() else 0.0
    if chair is not None and a in chair and b in chair:
        ha = np.array([chair[a][n]["hall_sent"] for n in na], float); hb = np.array([chair[b][n]["hall_sent"] for n in na], float)
        dh = 100 * (ha - hb); bh = dh[I].mean(1)
        res.update({"dCHAIRs": float(dh.mean()), "ci_CHAIRs": np.percentile(bh, [2.5, 97.5]).tolist(), "p_CHAIRs": pval(bh)})
    return res


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj.tolist()


def picks_from_preds(bench, rule, cands):
    p = preds(bench, rule); names = OUT["_names"][bench]
    return np.array([list(cands[i]).index(p[n]) for i, n in enumerate(names)])


def xref_block(bench, sel_rules):
    """Held-out-reference scale for the given selectors, plus XRef@N and the Oracle@N pick, with CIs."""
    m = np.load(os.path.join(MAT, f"{bench if bench != 'viecap' else 'viecap_full'}.npz"), allow_pickle=True)
    ok = m["ok5"]; names = [str(x) for x in m["names"]]; cands = m["cands"]
    OUT.setdefault("_names", {})[bench] = names
    M1 = m["M_one"][:, ok]; S4 = m["M_sel4"][:, ok]; Mall = m["M_all"][ok]
    n = ok.sum(); ar = np.arange(n)
    xref_img = np.mean([M1[f, ar, S4[f].argmax(1)] for f in range(5)], 0)
    tag = bench if bench != "viecap" else "viecap_full"
    five = os.path.join(MAT, f"{tag}_five.npz")
    # reference-selected pick on the five protocol references (NoCaps has ten; COCO has six on ten images)
    oracle_pick = (np.load(five)["M_five"][ok] if os.path.exists(five) else Mall).argmax(1)
    # the pick that selects on all five refs (the paper's Oracle@20 pick on the held-out scale)
    orc_img = np.mean([M1[f, ar, oracle_pick] for f in range(5)], 0)
    hum_img = np.nanmean(m["H_one"][:, ok], 0)
    rows = {"XRef@20": xref_img, "Oracle@20 pick": orc_img, "Human reference": hum_img}
    okn = [nm for nm, k in zip(names, ok) if k]
    for r in sel_rules:
        pk = picks_from_preds(bench, r, cands)[ok]
        rows[r] = np.mean([M1[f, ar, pk] for f in range(5)], 0)
    I = boot_idx(n)
    res = {"n": int(n), "means": {k: float(v.mean()) for k, v in rows.items()},
           "oracle_all_refs_5ref_scale": float(Mall.max(1).mean())}
    s = rows[sel_rules[0]]  # baseline s = first rule (three-agent sum)
    hr = rows["XRef@20"] - s; ap = rows["Oracle@20 pick"] - s
    bh = hr[I].mean(1); ba = ap[I].mean(1)
    res["headroom"] = float(hr.mean()); res["ci_headroom"] = np.percentile(bh, [2.5, 97.5]).tolist()
    res["apparent"] = float(ap.mean()); res["share"] = float(hr.mean() / ap.mean())
    res["ci_share"] = np.percentile(bh / ba, [2.5, 97.5]).tolist()
    for r in sel_rules[1:]:
        g = rows[r] - s; bg = g[I].mean(1)
        res[f"eff_{r}"] = float(g.mean() / hr.mean()); res[f"ci_eff_{r}"] = np.percentile(bg / bh, [2.5, 97.5]).tolist()
    # sensitivity to k with exact official CIDEr
    res["by_k"] = {}
    for k in (1, 2, 3, 4):
        Sk = m[f"M_sel{k}"][:, ok]
        folds = [float(M1[f, ar, Sk[f].argmax(1)].mean()) for f in range(5)]
        res["by_k"][k] = {"mean": float(np.mean(folds)), "folds": folds}
    return res


def main():
    chair_path = os.path.join(HERE, "chair_coverage_r1_coco.json")
    chair = None
    if os.path.exists(chair_path):
        cj = json.load(open(chair_path)); chair = {k: v["per_image"] for k, v in cj["systems"].items()}
        OUT["chair"] = {}
        names = sorted(next(iter(chair.values())))
        I = boot_idx(len(names))
        for sysname, rows in chair.items():
            hs = np.array([rows[n]["hall_sent"] for n in names], float)
            hm = np.array([rows[n]["hall"] for n in names], float); mm = np.array([rows[n]["mentions"] for n in names], float)
            cov = np.array([rows[n]["covered"] for n in names], float); gt = np.array([rows[n]["gt"] for n in names], float)
            OUT["chair"][sysname] = {"CHAIRs": 100 * hs.mean(), "ci_CHAIRs": (100 * np.percentile(hs[I].mean(1), [2.5, 97.5])).tolist(),
                                     "CHAIRi": 100 * hm.sum() / mm.sum(),
                                     "ci_CHAIRi": (100 * np.percentile(hm[I].sum(1) / mm[I].sum(1), [2.5, 97.5])).tolist(),
                                     "coverage": 100 * cov.sum() / gt.sum(), "mentions_per_caption": mm.mean()}
    # corpus scores of every rebuilt selection
    OUT["corpus"] = {}
    for f in sorted(os.listdir(PI)):
        r = json.load(open(os.path.join(PI, f)))
        OUT["corpus"][f[:-5]] = {"CIDEr": r["cider_corpus"], "SPICE": r.get("spice_corpus")}
    # held-out-reference blocks
    sel = {"coco": ["zsum3_lm_bigG_blip", "stack6_all_equal", "stack5_noLM_blip_bigG_vqa_mbr_ground", "lm_top1"],
           "nocaps": ["zsum3_lm_bigG_blip", "stack6_all_equal", "stack5_noLM_blip_bigG_vqa_mbr_ground", "lm_top1"],
           "flickr": ["zsum3_lm_bigG_blip", "stack6_all_equal", "stack5_noLM_blip_bigG_vqa_mbr_ground", "lm_top1"],
           "viecap": ["zsum3_lm_bigG_blip", "stack6_all_equal", "stack5_noLM_blip_bigG_vqa_mbr_ground", "lm_top1"]}
    OUT["xref"] = {}
    for b, rules in sel.items():
        if os.path.exists(os.path.join(MAT, f"{b if b != 'viecap' else 'viecap_full'}.npz")):
            OUT["xref"][b] = xref_block(b, rules)
    # pre-registered family
    fam = [("coco", "stack6_all_equal", "zsum3_lm_bigG_blip"), ("nocaps", "stack5_noLM_blip_bigG_vqa_mbr_ground", "blip_top1"),
           ("flickr", "stack5_noLM_blip_bigG_vqa_mbr_ground", "zsum3_lm_bigG_blip"),
           ("coco", "stack6_all_equal", "borda3_lm_bigG_blip"), ("nocaps", "stack5_noLM_blip_bigG_vqa_mbr_ground", "borda3_lm_bigG_blip"),
           ("coco", "stack6_all_equal", "blip2cap"), ("nocaps", "stack5_noLM_blip_bigG_vqa_mbr_ground", "blip2cap"),
           ("coco", "stack6_all_equal", "lm_top1"), ("nocaps", "stack6_all_equal", "lm_top1"), ("flickr", "stack6_all_equal", "lm_top1"),
           ("coco", "itmflickr_stack6_all_equal", "itmflickr_zsum3_lm_bigG_blip"),
           ("viecap", "stack5_noLM_blip_bigG_vqa_mbr_ground", "zsum3_lm_bigG_blip")]
    res = [dict(paired(b, a, c, chair if b == "coco" else None), bench=b) for b, a, c in fam]
    for r, h in zip(res, holm([r["p"] for r in res])):
        r["p_holm"] = h
    OUT["family"] = res
    extra = [("coco", "stack5_noLM_blip_bigG_vqa_mbr_ground", "zsum3_lm_bigG_blip"), ("nocaps", "stack6_all_equal", "blip_top1"),
             ("flickr", "stack6_all_equal", "zsum3_lm_bigG_blip"), ("nocaps", "zsum3_lm_bigG_blip", "blip_top1"),
             ("coco", "stack6_all_equal", "itmflickr_stack6_all_equal"), ("coco", "stack6_all_equal", "stack4_zsum3_ground"),
             ("viecap", "stack6_all_equal", "zsum3_lm_bigG_blip"), ("viecap", "stack5_noLM_blip_bigG_vqa_mbr_ground", "lm_top1")]
    for bnch in ("coco", "nocaps", "flickr"):
        for ag in ("bigG", "BLIP", "VQA", "MBR", "ground"):
            extra.append((bnch, f"loo_minus_{ag}", "stack6_all_equal"))
        extra.append((bnch, "stack5_noLM_blip_bigG_vqa_mbr_ground", "stack6_all_equal"))
    OUT["extra"] = [dict(paired(b, a, c, chair if b == "coco" else None), bench=b) for b, a, c in extra
                    if os.path.exists(os.path.join(PI, f"{b}__{a}.json")) and os.path.exists(os.path.join(PI, f"{b}__{c}.json"))]
    json.dump(OUT, open(os.path.join(HERE, "r1_numbers.json"), "w"), indent=1, default=float)
    write_tab_stats()
    print(json.dumps({k: OUT[k] for k in ("xref",) if k in OUT}, indent=1, default=float)[:3000])


LABEL = {"stack6_all_equal": "six agents", "stack5_noLM_blip_bigG_vqa_mbr_ground": "five agents", "zsum3_lm_bigG_blip": "three-agent sum",
         "blip_top1": "BLIP-ITM top-1", "borda3_lm_bigG_blip": "Borda proxy of \\vone{}", "blip2cap": "BLIP-2 captioning",
         "lm_top1": "captioner top-1", "itmflickr_stack6_all_equal": "six agents, Flickr30k ITM",
         "itmflickr_zsum3_lm_bigG_blip": "three-agent sum, Flickr30k ITM", "stack4_zsum3_ground": "three-agent sum + grounding",
         "loo_minus_bigG": "six without dual encoder", "loo_minus_BLIP": "six without cross-encoder", "loo_minus_VQA": "six without System One",
         "loo_minus_MBR": "six without consensus", "loo_minus_ground": "six without grounding"}


def fmt(x, ci, d=1):
    return f"{x:+.{d}f} [{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]".replace("-", "$-$")


def fp(p):
    return "$<$0.002" if p < 0.002 else f"{p:.3f}"


def write_tab_stats():
    os.makedirs(TAB, exist_ok=True)
    L = [r"\begin{table*}[t]", r"\centering\scriptsize",
         r"\caption{Paired bootstrap comparisons over images, 10,000 resamples, 95\% percentile intervals, official metrics. $\Delta$ is the first selector minus the second; for CHAIR$_s$ a negative $\Delta$ means fewer captions with a hallucinated object. Block A is the pre-registered family with Holm-adjusted $p$-values for \cider{}; block B gives the held-out-reference quantities of \cref{tab:xref}; block C gives further comparisons quoted in the text, with unadjusted $p$-values. Changed is the share of images on which the two selections differ, and gain if changed the mean \cider{} difference on those images. $\Delta$ is computed from unrounded scores and can differ by 0.1 from the difference of the rounded values in the other tables.}\label{tab:stats}",
         r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Benchmark & Comparison & $\Delta$\cider{} & $p$ & $\Delta$\spice{} & $\Delta$CHAIR$_s$ & Changed & \shortstack[r]{Gain if\\changed} \\", r"\midrule",
         r"\multicolumn{8}{@{}l}{\emph{A. Pre-registered family, Holm-adjusted}} \\"]
    def row(r, pkey):
        sp = fmt(r["dSPICE"], r["ci_SPICE"], 2) if "dSPICE" in r else "--"
        ch = fmt(r["dCHAIRs"], r["ci_CHAIRs"], 2) if "dCHAIRs" in r else "--"
        comp = f"{LABEL.get(r['a'], r['a'])} vs {LABEL.get(r['b'], r['b'])}".replace(
            "six agents, Flickr30k ITM vs three-agent sum, Flickr30k ITM", "Flickr30k ITM: six agents vs three-agent sum")
        return (f"{NAME[r['bench']]} & {comp} & {fmt(r['dCIDEr'], r['ci_CIDEr'])} & "
                f"{fp(r[pkey])} & {sp} & {ch} & {100 * r['changed']:.0f}\\% & {r['gain_if_changed']:+.1f} \\\\")
    for r in OUT["family"]:
        L.append(row(r, "p_holm"))
    L += [r"\midrule", r"\multicolumn{8}{@{}l}{\emph{B. Held-out-reference scale, baseline $s$ = three-agent sum}} \\"]
    for b, x in OUT.get("xref", {}).items():
        effs = "; ".join(f"efficiency {LABEL[k[4:]]} {100 * x[k]:.0f}\\% [{100 * x['ci_' + k][0]:.0f}, {100 * x['ci_' + k][1]:.0f}]"
                         for k in x if k.startswith("eff_") and "lm_top1" not in k)
        L.append(f"{NAME[b]} & \\multicolumn{{7}}{{l}}{{headroom {fmt(x['headroom'], x['ci_headroom'])}; surviving share "
                 f"{100 * x['share']:.0f}\\% [{100 * x['ci_share'][0]:.0f}, {100 * x['ci_share'][1]:.0f}]; {effs}}} \\\\")
    L += [r"\midrule", r"\multicolumn{8}{@{}l}{\emph{C. Further comparisons, unadjusted}} \\"]
    for r in OUT["extra"]:
        L.append(row(r, "p"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(TAB, "tab_stats.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
