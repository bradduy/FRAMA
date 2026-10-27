"""Rebuild every selection file of the paper for the R1 revision, from stored agent scores.

Two corrections relative to the submitted manuscript:
  1. The beam-consensus (MBR) agent uses document frequencies from the captioner's training
     captions (mbr_trainidf_<bench>.npy, see mbr_train_idf.py) instead of the test references.
  2. The "z-sum + grounding" row uses the fixed rule of the stack, z(LM)+z(bigG)+z(BLIP)+z(g_mean),
     instead of the grounding variant and weight chosen on test references (zsum3_plus_ground_best).
All other agents and all rules are unchanged; weights are equal unless a rule is an explicit
reliability or transfer analysis. Writes ../preds_r1/<bench>/<rule>__<split>.json and a
manifest ../preds_r1/manifest.json. Rules that do not involve the MBR agent are also rebuilt and
checked to be identical to the submitted ../preds/ files, which validates the loaders.

  python3 rebuild_preds_r1.py
"""
import glob, json, os, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
sys.path.insert(0, V2); sys.path.insert(0, HERE)
from pilot_headroom import z, ROOT
from pilot_c2_weighted_consensus import spectral_weights, mallows_em_weights
import analyze_grounding as AG
from analyze_ladder import load_scores
import mbr_train_idf as MB

OUT = os.path.join(V2, "preds_r1"); OLD = os.path.join(V2, "preds")
DUMP = dict(MB.DUMPS)
FEAT = {"coco": f"{ROOT}/inference_result/coco-indomain/coco-indomain-004_csfn/features.npz",
        "nocaps": f"{ROOT}/inference_result/nocaps-bigg-blip/features.npz",
        "flickr": f"{ROOT}/inference_result/flickr-indomain-bigg-blip/features.npz",
        "viecapfull": f"{V2}/c5/viecap_full_features.npz"}
PRED_DIR = {"coco": "coco", "nocaps": "nocaps", "flickr": "flickr", "viecapfull": "coco_viecap_full"}
for b, p in DUMP.items():
    AG.PATHS[b] = p
manifest = {"rules": {}, "identity_checks": {}}


def load(bench):
    beams = json.load(open(DUMP[bench])); N = len(beams); K = min(len(b["beam"]) for b in beams)
    f = np.load(FEAT[bench], allow_pickle=True)
    pos = {str(n): i for i, n in enumerate(f["image_names"])}
    feats = f["feats"].astype(np.float64)[[pos[b["image_name"]] for b in beams]][:, :K]
    G, miss = AG.grounding_features(bench, beams, K)
    V = AG.load_vqascore(bench, beams, K)
    assert miss == 0 and V is not None and not np.isnan(V).any(), f"{bench}: agent files incomplete"
    S = np.load(os.path.join(V2, f"mbr_trainidf_{bench}.npy"))
    assert S.shape == (N, K), (bench, S.shape)
    return beams, K, {"LM": feats[..., 0], "bigG": feats[..., 1], "BLIP": feats[..., 2], "VQA": V, "MBR": S, "ground": G[..., 0]}


def write(bench, beams, name, score, cands=None, idx=None, desc=""):
    """score: (n, K') over the candidates of the beams listed in idx (default all)."""
    idx = list(range(len(beams))) if idx is None else list(idx)
    K = score.shape[1]
    cands = [[x["caption"] for x in beams[i]["beam"]][:K] for i in idx] if cands is None else cands
    pick = np.asarray(score).argmax(1)
    recs = [{"image_name": beams[i]["image_name"], "captions": beams[i]["captions"], "prediction": cands[k][int(pick[k])]}
            for k, i in enumerate(idx)]
    d = os.path.join(OUT, PRED_DIR[bench]); os.makedirs(d, exist_ok=True)
    splits = {"overall": recs}
    if bench == "nocaps" and len(idx) == len(beams):
        for sp in ("in_domain", "near_domain", "out_domain"):
            splits[sp] = [r for r, i in zip(recs, idx) if beams[i].get("split") == sp]
    for sp, sel in splits.items():
        json.dump(sel, open(os.path.join(d, f"{name}__{sp}.json"), "w"))
    manifest["rules"][f"{PRED_DIR[bench]}/{name}"] = {"n": len(recs), "desc": desc}
    return pick


def check_identity(bench, name):
    """A rule without the MBR agent must reproduce the submitted selection exactly."""
    old = os.path.join(OLD, PRED_DIR[bench], f"{name}__overall.json")
    new = os.path.join(OUT, PRED_DIR[bench], f"{name}__overall.json")
    if not os.path.exists(old):
        manifest["identity_checks"][f"{PRED_DIR[bench]}/{name}"] = "no submitted file"; return
    a = {r["image_name"]: r["prediction"] for r in json.load(open(old))}
    b = {r["image_name"]: r["prediction"] for r in json.load(open(new))}
    same = sum(a[k] == b.get(k) for k in a) / len(a)
    manifest["identity_checks"][f"{PRED_DIR[bench]}/{name}"] = same
    assert same == 1.0, f"loader mismatch on {bench}/{name}: {same:.4f}"


def zs(A, names):
    return sum(z(A[n]) for n in names)


SIX = ["LM", "bigG", "BLIP", "VQA", "MBR", "ground"]
FIVE = ["bigG", "BLIP", "VQA", "MBR", "ground"]
STANDARD = {
    "lm_top1": (None, "LM"), "blip_top1": (None, "BLIP"),
    "zsum3_lm_bigG_blip": (["LM", "bigG", "BLIP"], None),
    "stack4_zsum3_vqa": (["LM", "bigG", "BLIP", "VQA"], None),
    "stack4_zsum3_mbr": (["LM", "bigG", "BLIP", "MBR"], None),
    "stack4_zsum3_ground": (["LM", "bigG", "BLIP", "ground"], None),
    "stack5_zsum3_vqa_mbr": (["LM", "bigG", "BLIP", "VQA", "MBR"], None),
    "stack6_all_equal": (SIX, None),
    "stack5_noLM_blip_bigG_vqa_mbr_ground": (FIVE, None),
    "stack4_noLM_blip_bigG_vqa_mbr": (["bigG", "BLIP", "VQA", "MBR"], None),
}
UNCHANGED = ["lm_top1", "blip_top1", "zsum3_lm_bigG_blip", "stack4_zsum3_vqa"]

data = {}
for bench in ("coco", "nocaps", "flickr", "viecapfull"):
    beams, K, A = load(bench); data[bench] = (beams, K, A)
    for name, (agents, single) in STANDARD.items():
        sc = A[single] if single else zs(A, agents)
        write(bench, beams, name, sc, desc=single or "+".join(agents))
    for name in UNCHANGED:
        check_identity(bench, name)
    if bench in ("coco", "nocaps", "flickr"):
        for drop in SIX:
            if drop == "LM":
                continue
            write(bench, beams, f"loo_minus_{drop}", zs(A, [a for a in SIX if a != drop]), desc=f"six without {drop}")
        sigs = [A[a] for a in SIX]; Z = [z(s) for s in sigs]
        ws, _, _ = spectral_weights(sigs); wm, _ = mallows_em_weights(sigs, iters=8)
        write(bench, beams, "rel6_spectral", sum(w * zz for w, zz in zip(ws, Z)), desc="spectral " + str(np.round(6 * ws, 3).tolist()))
        write(bench, beams, "rel6_mallows", sum(w * zz for w, zz in zip(wm, Z)), desc="mallows " + str(np.round(6 * wm, 3).tolist()))
    # copy files that do not depend on any agent score we changed
    for keep in ("borda3_lm_bigG_blip", "oracle20_approx", "blip2cap"):
        src = os.path.join(OLD, PRED_DIR[bench], f"{keep}__overall.json")
        if os.path.exists(src):
            for f in glob.glob(os.path.join(OLD, PRED_DIR[bench], f"{keep}__*.json")):
                shutil.copy(f, os.path.join(OUT, PRED_DIR[bench], os.path.basename(f)))
            manifest["rules"][f"{PRED_DIR[bench]}/{keep}"] = {"desc": "copied from submitted preds, no changed agent"}

# ladder (COCO, NoCaps)
for bench in ("coco", "nocaps"):
    beams, K, A = data[bench]
    sg = load_scores(os.path.join(V2, f"scores_siglip2_{bench}.jsonl"), beams, K)
    b2 = load_scores(os.path.join(V2, f"scores_blip2itm_{bench}.jsonl"), beams, K)
    assert not np.isnan(sg).any() and not np.isnan(b2).any()
    six = zs(A, SIX)
    rules = {"ladder_zsum3_bigG2siglip2": z(A["LM"]) + z(sg) + z(A["BLIP"]),
             "ladder_zsum3_plus_siglip2": zs(A, ["LM", "bigG", "BLIP"]) + z(sg),
             "ladder_zsum3_plus_blip2itm": zs(A, ["LM", "bigG", "BLIP"]) + z(b2),
             "ladder_stack6_bigG2siglip2": z(A["LM"]) + z(sg) + zs(A, ["BLIP", "VQA", "MBR", "ground"]),
             "ladder_stack6_plus_blip2itm": six + z(b2),
             "ladder_stack8_all": six + z(sg) + z(b2)}
    for n, sc in rules.items():
        write(bench, beams, n, sc, desc="ladder")
    for n in ("ladder_zsum3_bigG2siglip2", "ladder_zsum3_plus_siglip2", "ladder_zsum3_plus_blip2itm"):
        check_identity(bench, n)

# cross-encoder swap to the Flickr30k-fine-tuned BLIP-ITM (COCO)
beams, K, A = data["coco"]
itm = {}
for l in open(os.path.join(V2, "revision", "itm_flickr_coco.jsonl")):
    r = json.loads(l); itm[r["image_name"]] = r["itm"][:K]
F = np.array([itm[b["image_name"]] for b in beams], float)
AF = dict(A, BLIP=F)
for n, sc in {"itmflickr_blip_top1": F, "itmflickr_zsum3_lm_bigG_blip": zs(AF, ["LM", "bigG", "BLIP"]),
              "itmflickr_stack6_all_equal": zs(AF, SIX), "itmflickr_stack5_noLM_blip_bigG_vqa_mbr_ground": zs(AF, FIVE)}.items():
    write("coco", beams, n, sc, desc="BLIP-ITM flickr checkpoint")
for n in ("itmflickr_blip_top1", "itmflickr_zsum3_lm_bigG_blip"):
    check_identity("coco", n)

# small System One models on the 1,000-image subsample (every fifth image)
sub = np.arange(0, len(beams), 5)
names = [beams[i]["image_name"] for i in sub]
As = {k: v[sub] for k, v in A.items()}
for path in sorted(glob.glob(os.path.join(V2, "smalljudge_*_coco20s.jsonl"))):
    tag = os.path.basename(path)[len("smalljudge_"):-len("_coco20s.jsonl")]
    d = {}
    for line in open(path):
        r = json.loads(line); d[r["image_name"]] = r["vqascore"]
    J = np.array([d[n][:K] for n in names], float)
    write("coco", beams, f"sj_{tag}_zsum3plus", zs(As, ["LM", "bigG", "BLIP"]) + z(J), idx=sub, desc="subsample")
    write("coco", beams, f"sj_{tag}_stack6swap", zs(As, ["LM", "bigG", "BLIP", "MBR", "ground"]) + z(J), idx=sub, desc="subsample")
    write("coco", beams, f"sj_{tag}_alone", J, idx=sub, desc="subsample")
    check_identity("coco", f"sj_{tag}_zsum3plus")
write("coco", beams, "sub1000_stack6_all_equal", zs(As, SIX), idx=sub, desc="subsample six with BLIP-2")
write("coco", beams, "sub1000_stack5_noVQA", zs(As, ["LM", "bigG", "BLIP", "MBR", "ground"]), idx=sub, desc="subsample, no System One")
write("coco", beams, "sub1000_zsum3", zs(As, ["LM", "bigG", "BLIP"]), idx=sub, desc="subsample")
write("coco", beams, "sub1000_zsum3_vqa", zs(As, ["LM", "bigG", "BLIP", "VQA"]), idx=sub, desc="subsample")

# System 1 versus System 2 on 500 images
rows = {}
for f in glob.glob(os.path.join(V2, "revision", "cot_shard*.jsonl")):
    for l in open(f):
        r = json.loads(l); rows[r["image_name"]] = r
idx_of = {b["image_name"]: i for i, b in enumerate(beams)}
I = np.array(sorted(idx_of[n] for n in rows))
S1 = np.array([rows[beams[i]["image_name"]]["s1"] for i in I]); S2 = np.array([rows[beams[i]["image_name"]]["s2"] for i in I])
Ac = {k: v[I] for k, v in A.items()}
for n, sc in {"cot500_zsum3": zs(Ac, ["LM", "bigG", "BLIP"]), "cot500_six_blip2": zs(Ac, SIX),
              "cot500_s1_alone": S1, "cot500_s2_alone": S2,
              "cot500_zsum3_s1": zs(Ac, ["LM", "bigG", "BLIP"]) + z(S1), "cot500_zsum3_s2": zs(Ac, ["LM", "bigG", "BLIP"]) + z(S2),
              "cot500_six_s1": zs(Ac, ["LM", "bigG", "BLIP", "MBR", "ground"]) + z(S1),
              "cot500_six_s2": zs(Ac, ["LM", "bigG", "BLIP", "MBR", "ground"]) + z(S2)}.items():
    write("coco", beams, n, sc, idx=I, desc="cot500")
for n in ("cot500_zsum3", "cot500_s1_alone", "cot500_zsum3_s1", "cot500_zsum3_s2"):
    check_identity("coco", n)

# BLIP-2 caption as a 21st candidate (outside the text-only regime), COCO and NoCaps
df, logn = MB.build_df("coco"); MB.init(df, logn)
for bench in ("coco", "nocaps"):
    beams, K, A = data[bench]; names = [b["image_name"] for b in beams]; R = os.path.join(V2, "revision")
    f = np.load(os.path.join(R, f"m4_{bench}_features.npz"), allow_pickle=True)
    pos = {str(n): i for i, n in enumerate(f["image_names"])}; Fm = f["feats"][[pos[n] for n in names]][:, 0, :]
    vq = {json.loads(l)["image_name"]: json.loads(l)["vqascore"][0] for l in open(os.path.join(R, f"m4_{bench}_vqa.jsonl"))}
    gr = {}
    for l in open(os.path.join(R, f"m4_{bench}_ground.jsonl")):
        r = json.loads(l); c = r["cands"][0]; sc = [r["nouns"][n][0] for n in c["nouns"] if n in r["nouns"]]
        gr[r["image_name"]] = float(np.mean(sc)) if sc else np.nan
    p2 = {r["image_name"]: r["prediction"] for r in json.load(open(os.path.join(OLD, bench, "blip2cap__overall.json")))}
    cands21 = [[x["caption"] for x in b["beam"]][:K] + [p2[b["image_name"]]] for b in beams]
    S21 = np.array([MB.beam_scores(c) for c in cands21])
    BG = np.c_[A["bigG"], Fm[:, 1]]; BL = np.c_[A["BLIP"], Fm[:, 2]]; VQ = np.c_[A["VQA"], [vq[n] for n in names]]
    GR = np.c_[A["ground"], [gr[n] for n in names]]
    GR[np.isnan(GR[:, K]), K] = GR[np.isnan(GR[:, K]), :K].mean(1)
    rules = {"m4_five21": z(BG) + z(BL) + z(VQ) + z(S21) + z(GR),
             "m4_five20": z(BG[:, :K]) + z(BL[:, :K]) + z(VQ[:, :K]) + z(S21[:, :K]) + z(GR[:, :K]),
             "m4_blip21": z(BL), "m4_bigG21": z(BG), "m4_s1_21": z(VQ), "m4_align2_21": z(BG) + z(BL),
             "m4_align2s1_21": z(BG) + z(BL) + z(VQ), "m4_four21_noMBR": z(BG) + z(BL) + z(VQ) + z(GR)}
    for n, sc in rules.items():
        k = sc.shape[1]
        pick = write(bench, beams, n, sc, cands=[c[:k] for c in cands21], desc="21-candidate pool")
        manifest["rules"][f"{PRED_DIR[bench]}/{n}"]["frac_blip2_picked"] = float(np.mean(pick == K)) if k == K + 1 else 0.0
    for n in ("m4_blip21", "m4_align2s1_21", "m4_four21_noMBR"):
        check_identity(bench, n)

json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print("rules written:", len(manifest["rules"]))
print("identity checks:", json.dumps(manifest["identity_checks"], indent=0))
