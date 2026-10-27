"""E6 analysis (e6_prereg.md), on the laptop after wait_e6.sh has pulled the GCP2 outputs.

E6a  SPICE by tuple category: pooled precision, recall and F per selector, paired bootstrap of the differences.
E6b  SugarCrepe and VALSE foils inside the beam: accuracy of every agent and of the five-agent sum, by phenomenon.
E6c  Facts checked by Qwen3-VL-4B: validation against THumB human Precision, then unsupported-fact rates of the
     selectors by fact type, facts and supported facts per caption, paired bootstrap.
Bootstrap: 10,000 resamples (2,000 for Kendall tau), seed 0, 95% percentile intervals; differences are A minus B in points.
Needs NLTK WordNet for the action rule (NLTK_DATA). Writes e6_results.json.

  NLTK_DATA=... python e6_analyze.py
"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import kendalltau
from nltk.corpus import wordnet as wn

HERE = os.path.dirname(os.path.abspath(__file__)); R1 = os.path.dirname(HERE); V2 = os.path.dirname(R1)
sys.path.insert(0, V2); sys.path.insert(0, R1)
from pilot_headroom import z, ROOT
B = 10000; RNG = np.random.default_rng(0)
RULES = {"lm_top1": "captioner top-1", "borda3_lm_bigG_blip": "Borda proxy", "zsum3_lm_bigG_blip": "three-agent z-sum",
         "stack6_all_equal": "six-agent FRAMA", "stack5_noLM_blip_bigG_vqa_mbr_ground": "five-agent FRAMA"}
PAIRS = [("stack6_all_equal", "lm_top1"), ("stack6_all_equal", "borda3_lm_bigG_blip"), ("stack6_all_equal", "zsum3_lm_bigG_blip"),
         ("stack5_noLM_blip_bigG_vqa_mbr_ground", "lm_top1"), ("stack5_noLM_blip_bigG_vqa_mbr_ground", "borda3_lm_bigG_blip"),
         ("stack5_noLM_blip_bigG_vqa_mbr_ground", "zsum3_lm_bigG_blip")]
UNITS = ["coco", "nocaps", "flickr"]
out = {}


def weights(n, b=B, block=500):
    """Bootstrap resampling as multinomial count vectors, in blocks."""
    for s in range(0, b, block):
        yield RNG.multinomial(n, np.full(n, 1.0 / n), size=min(block, b - s)).astype(np.float64)


def ci(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def spice_rows(unit, rule):
    p = os.path.join(HERE, "spice_out", f"{unit}__{rule}.json")
    if not os.path.exists(p):
        return None, None
    names = json.load(open(os.path.join(HERE, "spice_in", f"{unit}__{rule}.names.json")))
    src = {s["image_id"]: s["test"] for s in json.load(open(os.path.join(HERE, "spice_in", f"{unit}__{rule}.json")))}
    res = sorted(json.load(open(p)), key=lambda r: r["image_id"])
    return [(names[r["image_id"]], src[r["image_id"]]) for r in res], res


# ---------------- E6a ----------------
CATS = ["All", "Object", "Attribute", "Relation", "Color", "Cardinality", "Size"]
e6a = {}
for unit in UNITS:
    T = {}
    for rule in RULES:
        keys, res = spice_rows(unit, rule)
        if res is None:
            continue
        T[rule] = (keys, np.array([[[r["scores"][c][k] for k in ("tp", "fp", "fn")] for c in CATS] for r in res], float))
    if not T:
        continue
    names0 = [k[0] for k in next(iter(T.values()))[0]]
    assert all([k[0] for k in v[0]] == names0 for v in T.values()), unit
    N = len(names0); u = {"n": N, "rules": {}, "diffs": {}}

    def prf(S):  # S: (..., C, 3) pooled counts
        tp, fp, fn = S[..., 0], S[..., 1], S[..., 2]
        P = 100 * tp / np.maximum(tp + fp, 1e-9); R = 100 * tp / np.maximum(tp + fn, 1e-9)
        return P, R, 2 * P * R / np.maximum(P + R, 1e-9)

    for rule, (_, X) in T.items():
        P, R, F = prf(X.sum(0))
        u["rules"][rule] = {c: {"P": float(P[i]), "R": float(R[i]), "F": float(F[i]), "tp": float(X[:, i, 0].sum()),
                                "tuples_per_caption": float((X[:, i, 0] + X[:, i, 1]).sum() / N)} for i, c in enumerate(CATS)}
    for a, b in PAIRS:
        if a not in T or b not in T:
            continue
        Xa, Xb = T[a][1].reshape(N, -1), T[b][1].reshape(N, -1)
        dP, dR = [], []
        for W in weights(N):
            Pa, Ra, _ = prf((W @ Xa).reshape(-1, len(CATS), 3)); Pb, Rb, _ = prf((W @ Xb).reshape(-1, len(CATS), 3))
            dP.append(Pa - Pb); dR.append(Ra - Rb)
        dP, dR = np.concatenate(dP), np.concatenate(dR)
        u["diffs"][f"{a} - {b}"] = {c: {"dP": u["rules"][a][c]["P"] - u["rules"][b][c]["P"], "ci_dP": ci(dP[:, i]),
                                        "dR": u["rules"][a][c]["R"] - u["rules"][b][c]["R"], "ci_dR": ci(dR[:, i])}
                                    for i, c in enumerate(CATS)}
    e6a[unit] = u
out["E6a"] = e6a

# ---------------- E6c: fact types and verifier results ----------------
NUM = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "several",
       "many", "few", "multiple", "couple", "dozen", "numerous", "both", "pair"}
COLORS = {"white", "black", "red", "blue", "green", "yellow", "brown", "orange", "pink", "purple", "gray", "grey", "silver",
          "gold", "golden", "tan", "beige", "maroon", "teal", "violet", "colorful", "multicolored", "dark", "light"}
SIZES = {"big", "small", "large", "little", "tall", "short", "huge", "tiny", "giant", "long", "wide", "narrow", "thick",
         "thin", "massive", "enormous", "miniature", "high", "low"}
PREP = {"on", "in", "at", "near", "beside", "behind", "above", "below", "under", "over", "inside", "outside", "along",
        "across", "around", "by", "against", "between", "into", "onto", "through", "toward", "towards", "down", "up", "off",
        "atop", "underneath", "beneath", "next", "to", "front", "top", "of", "side", "within", "among", "amid", "past",
        "beyond", "upon", "out", "close", "far", "from", "back", "bottom", "middle", "center", "edge", "inside"}
NONSPATIAL = {"with", "of", "for", "as", "like", "than", "about", "from", "without", "including"}
TYPES = ["object", "attribute", "color", "count", "size", "spatial relation", "action", "other relation"]


def ftype(t, caption):
    if len(t) == 1:
        return "object"
    if len(t) == 2:
        a = t[1]
        if a in NUM or a.isdigit():
            return "count"
        if a in COLORS:
            return "color"
        if a in SIZES:
            return "size"
        toks = re.findall(r"[a-z]+", caption.lower())
        if any(w.endswith("ing") and wn.morphy(w, wn.VERB) == a for w in toks):
            return "action"
        return "attribute"
    words = t[1].split()
    if len(words) == 1 and words[0] in NONSPATIAL:
        return "other relation"
    if all(w in PREP for w in words):
        return "spatial relation"
    return "action"


def load_verify(unit):
    d = {}
    for p in glob.glob(os.path.join(HERE, "e6c_out", unit, "chunk_*.jsonl")):
        for line in open(p):
            r = json.loads(line); d[(r["image_name"], r["caption"])] = (r["tuples"], r["p_yes"])
    return d


VER = {u: load_verify(u) for u in ["thumb", "thumb2"] + UNITS}


def fact_counts(unit, keys):
    """(N, len(TYPES), 2): facts and unsupported facts per caption and type."""
    X = np.zeros((len(keys), len(TYPES), 2))
    for i, (im, cap) in enumerate(keys):
        tups, ps = VER[unit][(im, cap)]
        for t, p in zip(tups, ps):
            k = TYPES.index(ftype(t, cap)); X[i, k, 0] += 1; X[i, k, 1] += p <= 0.5
    return X


# validation on THumB: caption-level share of supported facts against human Precision
th = np.load(os.path.join(R1, "human_bench", "thumb_cider.npz"), allow_pickle=True)
images, systems, cands, HP = [str(x) for x in th["images"]], [str(s) for s in th["systems"]], th["cands"], th["P"]
ch = json.load(open(os.path.join(R1, "human_bench", "thumb_chair.json")))["systems"]
sp = {}
for unit, sysl in (("thumb", ["Up-Down", "Unified-VLP", "VinVL-base"]), ("thumb2", ["VinVL-large"])):
    for s in sysl:
        keys, res = spice_rows(unit, s)
        for (im, cap), r in zip(keys, res):
            sp[(s, im)] = r["scores"]["All"]["pr"]
rows = []
for i, im in enumerate(images):
    for j, s in enumerate(systems):
        unit = "thumb2" if s == "VinVL-large" else "thumb"
        tups, ps = VER[unit][(im, str(cands[i, j]))]
        if not ps:
            continue
        rows.append((i, j, float(np.mean(np.array(ps) > 0.5)), float(np.mean(ps)), -float(ch[s]["per_image"][im]["hall_sent"]),
                     sp[(s, im)] if sp[(s, im)] is not None else np.nan, float(HP[i, j])))
R = np.array(rows)
val = {"n_captions": len(R)}
for k, name in ((2, "verifier share of supported facts"), (3, "verifier mean P(yes)"), (4, "CHAIR_s, negated"),
                (5, "SPICE precision against the references")):
    m = ~np.isnan(R[:, k])
    tau = kendalltau(R[m, k], R[m, 6]).correlation
    img = R[m, 0].astype(int); uimg = np.unique(img); grp = [np.where(img == q)[0] for q in uimg]; taus = []
    for _ in range(2000):  # images resampled; 2,000 resamples for the rank correlation
        idx = np.concatenate([grp[q] for q in RNG.integers(0, len(uimg), len(uimg))])
        taus.append(kendalltau(R[m][idx, k], R[m][idx, 6]).correlation)
    num = den = 0.0
    for i in np.unique(img):
        rr = R[m][img == i]
        for a in range(len(rr)):
            for b in range(a + 1, len(rr)):
                if rr[a, 6] != rr[b, 6]:
                    den += 1; s = np.sign(rr[a, k] - rr[b, k])
                    num += 0.5 if s == 0 else float(s == np.sign(rr[a, 6] - rr[b, 6]))
    val[name] = {"kendall_tau": float(tau), "ci_tau": ci(taus), "pairwise_agreement": num / den, "n_pairs": den}
# paired difference of tau, verifier share against CHAIR and against SPICE precision, on the same image resamples
m = ~np.isnan(R[:, 5]); Rm = R[m]; img = Rm[:, 0].astype(int); uimg = np.unique(img); grp = [np.where(img == q)[0] for q in uimg]
tk = lambda X, k: kendalltau(X[:, k], X[:, 6]).correlation
dd = []
for _ in range(2000):
    X = Rm[np.concatenate([grp[q] for q in RNG.integers(0, len(uimg), len(uimg))])]
    dd.append([tk(X, 2) - tk(X, 4), tk(X, 2) - tk(X, 5)])
dd = np.array(dd)
val["tau_verifier_minus_chair"] = {"d": tk(Rm, 2) - tk(Rm, 4), "ci": ci(dd[:, 0])}
val["tau_verifier_minus_spice"] = {"d": tk(Rm, 2) - tk(Rm, 5), "ci": ci(dd[:, 1])}
print("tau differences", val["tau_verifier_minus_chair"], val["tau_verifier_minus_spice"])
out["E6c_validation"] = val

e6c = {}
for unit in UNITS:
    T = {}
    for rule in RULES:
        keys, res = spice_rows(unit, rule)
        if res is not None:
            T[rule] = fact_counts(unit, keys)
    N = len(next(iter(T.values()))); u = {"n": N, "rules": {}, "diffs": {}}

    def stats(S):  # S: (..., types, 2) pooled; returns unsupported % by type + all, facts and supported per caption
        allf, allu = S[..., 0].sum(-1), S[..., 1].sum(-1)
        return np.concatenate([100 * S[..., 1] / np.maximum(S[..., 0], 1e-9), (100 * allu / allf)[..., None],
                               (allf / N)[..., None], ((allf - allu) / N)[..., None]], -1)
    LAB = TYPES + ["all facts", "facts per caption", "supported facts per caption"]
    for rule, X in T.items():
        s = stats(X.sum(0)); anyu = float(100 * np.mean(X[:, :, 1].sum(1) > 0))
        u["rules"][rule] = {**{l: float(v) for l, v in zip(LAB, s)}, "n_facts": {t: float(X[:, i, 0].sum()) for i, t in enumerate(TYPES)},
                            "captions with an unsupported fact": anyu}
    for a, b in PAIRS:
        if a not in T or b not in T:
            continue
        Xa, Xb = T[a].reshape(N, -1), T[b].reshape(N, -1); D = []
        for W in weights(N):
            D.append(stats((W @ Xa).reshape(-1, len(TYPES), 2)) - stats((W @ Xb).reshape(-1, len(TYPES), 2)))
        D = np.concatenate(D)
        u["diffs"][f"{a} - {b}"] = {l: {"d": u["rules"][a][l] - u["rules"][b][l], "ci": ci(D[:, i])} for i, l in enumerate(LAB)}
    e6c[unit] = u
out["E6c"] = e6c

# ---------------- E6b ----------------
import analyze_grounding as AG
import mbr_train_idf as MB
DUMP = MB.DUMPS["coco"]; AG.PATHS["coco"] = DUMP
beams = json.load(open(DUMP)); K = min(len(b["beam"]) for b in beams); pidx = {b["image_name"]: i for i, b in enumerate(beams)}
f = np.load(f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_csfn/features.npz", allow_pickle=True)
pos = {str(n): i for i, n in enumerate(f["image_names"])}
feats = f["feats"].astype(np.float64)[[pos[b["image_name"]] for b in beams]][:, :K]
G, _ = AG.grounding_features("coco", beams, K); V = AG.load_vqascore("coco", beams, K)
A = {"bigG": feats[..., 1], "BLIP-ITM": feats[..., 2], "System One verifier": V, "grounding": G[..., 0]}
eb = json.load(open(os.path.join(HERE, "e6b_beams.json"))); eidx = {r["image_name"]: i for i, r in enumerate(eb)}
al = np.load(os.path.join(HERE, "e6b_align.npz"), allow_pickle=True)
assert [str(x) for x in al["image_names"]] == [r["image_name"] for r in eb]
vq = {json.loads(l)["image_name"]: json.loads(l)["vqascore"] for l in open(os.path.join(HERE, "e6b_vqa.jsonl"))}
gr = {json.loads(l)["image_name"]: json.loads(l) for l in open(os.path.join(HERE, "e6b_ground.jsonl"))}


def extra(im, cap):
    e = eidx[im]; j = [x["caption"] for x in eb[e]["beam"]].index(cap); r = gr[im]
    sc = [r["nouns"][n][0] for n in r["cands"][j]["nouns"] if n in r["nouns"]]
    return {"bigG": al["feats"][e, j, 1], "BLIP-ITM": al["feats"][e, j, 2], "System One verifier": vq[im][j],
            "grounding": float(np.mean(sc)) if sc else np.nan}


df, logn = MB.build_df("coco"); MB.init(df, logn)
items = json.load(open(os.path.join(HERE, "e6b_items.json")))
SEL = ["bigG", "BLIP-ITM", "System One verifier", "consensus", "grounding", "z-sum bigG + BLIP-ITM", "five-agent FRAMA"]
correct = np.zeros((len(items), len(SEL)))
for n, it in enumerate(items):
    i = pidx[it["image_name"]]; ep, ef = extra(it["image_name"], it["pos"]), extra(it["image_name"], it["foil"])
    pool = {a: np.r_[A[a][i], ep[a], ef[a]].astype(float) for a in A}
    g = pool["grounding"]; g[np.isnan(g)] = np.nanmean(g) if not np.isnan(g).all() else 0.0
    caps = [x["caption"] for x in beams[i]["beam"]][:K] + [it["pos"], it["foil"]]
    pool["consensus"] = np.array(MB.beam_scores(caps))
    zz = {a: z(v[None])[0] for a, v in pool.items()}
    sc = {a: pool[a] for a in ("bigG", "BLIP-ITM", "System One verifier", "consensus", "grounding")}
    sc["z-sum bigG + BLIP-ITM"] = zz["bigG"] + zz["BLIP-ITM"]
    sc["five-agent FRAMA"] = sum(zz[a] for a in ("bigG", "BLIP-ITM", "System One verifier", "consensus", "grounding"))
    for k, s in enumerate(SEL):
        d = sc[s][K] - sc[s][K + 1]; correct[n, k] = 0.5 if d == 0 else float(d > 0)
GROUPS = {"SugarCrepe replace object": ["sugarcrepe:replace_obj"], "SugarCrepe replace attribute": ["sugarcrepe:replace_att"],
          "SugarCrepe replace relation": ["sugarcrepe:replace_rel"], "SugarCrepe swap object": ["sugarcrepe:swap_obj"],
          "SugarCrepe swap attribute": ["sugarcrepe:swap_att"], "SugarCrepe add object": ["sugarcrepe:add_obj"],
          "SugarCrepe add attribute": ["sugarcrepe:add_att"], "VALSE existence": ["valse:existence"],
          "VALSE plurality": ["valse:plurals"], "VALSE counting": ["valse:counting-hard", "valse:counting-small-quant", "valse:counting-adversarial"],
          "VALSE spatial relations": ["valse:relations"], "VALSE noun foils (FOIL-it)": ["valse:foil-it"]}
e6b = {"n_items": len(items), "groups": {}}
for g, pre in GROUPS.items():
    m = np.array([any(it["id"].startswith(p + ":") for p in pre) for it in items])
    X = correct[m]; n = len(X); bs = np.concatenate([W @ X / n for W in weights(n)])
    e6b["groups"][g] = {"n": int(n), **{s: {"acc": float(100 * X[:, k].mean()), "ci": ci(100 * bs[:, k])} for k, s in enumerate(SEL)}}
gm = np.array([[e6b["groups"][g][s]["acc"] for s in SEL] for g in GROUPS])
e6b["macro_mean"] = {s: float(gm[:, k].mean()) for k, s in enumerate(SEL)}
n = len(correct); bs = np.concatenate([W @ correct / n for W in weights(n)])
e6b["all_items"] = {s: {"acc": float(100 * correct[:, k].mean()), "ci": ci(100 * bs[:, k])} for k, s in enumerate(SEL)}
# paired item bootstrap of the five-agent sum against every other selector, on all items and on the macro mean over groups
gid = np.array([next(g for g, pre in GROUPS.items() if any(it["id"].startswith(p + ":") for p in pre)) for it in items])
F = SEL.index("five-agent FRAMA"); e6b["five_minus"] = {}
for k, s in enumerate(SEL):
    if k == F:
        continue
    d = correct[:, F] - correct[:, k]; ba, bm = [], []
    for W in weights(n):
        ba.append(100 * W @ d / n)
        bm.append(np.mean([100 * (W[:, gid == g] @ d[gid == g]) / np.maximum(W[:, gid == g].sum(1), 1) for g in GROUPS], 0))
    ba, bm = np.concatenate(ba), np.concatenate(bm)
    e6b["five_minus"][s] = {"all": float(100 * d.mean()), "ci_all": ci(ba),
                            "macro": e6b["macro_mean"]["five-agent FRAMA"] - e6b["macro_mean"][s], "ci_macro": ci(bm)}
out["E6b"] = e6b

json.dump(out, open(os.path.join(HERE, "e6_results.json"), "w"), indent=1)

# ---------------- print ----------------
print("E6c validation on THumB (n captions %d)" % val["n_captions"])
for k, v in val.items():
    if isinstance(v, dict) and "kendall_tau" in v:
        print(f"  {k:42s} tau {v['kendall_tau']:.3f} [{v['ci_tau'][0]:.3f}, {v['ci_tau'][1]:.3f}]  pairwise {v['pairwise_agreement']:.3f}")
for unit, u in e6a.items():
    print(f"\nE6a {unit}  P / R by category")
    for rule, r in u["rules"].items():
        print(f"  {RULES[rule]:20s} " + "  ".join(f"{c[:5]} {r[c]['P']:.1f}/{r[c]['R']:.1f}" for c in CATS))
    for k, d in u["diffs"].items():
        print(f"  {k:60s} " + "  ".join(f"{c[:5]} dP {d[c]['dP']:+.1f} [{d[c]['ci_dP'][0]:+.1f},{d[c]['ci_dP'][1]:+.1f}] dR {d[c]['dR']:+.1f} [{d[c]['ci_dR'][0]:+.1f},{d[c]['ci_dR'][1]:+.1f}]" for c in CATS[:4]))
for unit, u in e6c.items():
    print(f"\nE6c {unit}  unsupported % by type")
    for rule, r in u["rules"].items():
        print(f"  {RULES[rule]:20s} all {r['all facts']:.1f}  " + "  ".join(f"{t[:8]} {r[t]:.1f}" for t in TYPES)
              + f"  facts/cap {r['facts per caption']:.2f} supported/cap {r['supported facts per caption']:.2f}")
    for k, d in u["diffs"].items():
        print(f"  {k:60s} all {d['all facts']['d']:+.2f} [{d['all facts']['ci'][0]:+.2f},{d['all facts']['ci'][1]:+.2f}]  supported/cap "
              f"{d['supported facts per caption']['d']:+.3f} [{d['supported facts per caption']['ci'][0]:+.3f},{d['supported facts per caption']['ci'][1]:+.3f}]")
print(f"\nE6b ({e6b['n_items']} items) accuracy %")
print("  " + " " * 32 + "  ".join(f"{s[:12]:>12s}" for s in SEL))
for g, r in e6b["groups"].items():
    print(f"  {g:28s} n={r['n']:4d} " + "  ".join(f"{r[s]['acc']:12.1f}" for s in SEL))
print(f"  {'macro mean':37s}" + "  ".join(f"{e6b['macro_mean'][s]:12.1f}" for s in SEL))
for s, d in e6b["five_minus"].items():
    print(f"  five-agent - {s:24s} all {d['all']:+.1f} [{d['ci_all'][0]:+.1f},{d['ci_all'][1]:+.1f}]  macro {d['macro']:+.1f} [{d['ci_macro'][0]:+.1f},{d['ci_macro'][1]:+.1f}]")
