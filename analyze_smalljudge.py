"""Small-VLM yes-probability judges vs the BLIP-2 FlanT5-XL agent on the 1,000-image COCO
subsample: within-beam Spearman with CIDEr, correlation with BLIP-2 P(yes), and the six-agent
stack with the VQA agent replaced by each small judge. Writes preds/coco/sj_<tag>_*__overall.json
for official evaluation (1,000 images each; compare with preds/coco_ifcapsub/)."""
import json, os, sys, glob
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_headroom import z
from pilot_c2_weighted_consensus import load
import analyze_grounding as AG

HERE = os.path.dirname(os.path.abspath(__file__))
feats, M, splits = load("coco")
beams = json.load(open(AG.PATHS["coco"]))
N, K = M.shape
G, _ = AG.grounding_features("coco", beams, K)
V = AG.load_vqascore("coco", beams, K)
S = np.load(os.path.join(HERE, "mbr_self_coco.npy"))
sub = np.arange(0, N, 5)
names = [beams[i]["image_name"] for i in sub]
assert names == [l.strip() for l in open(os.path.join(HERE, "c5", "coco_sub1000_images.txt")) if l.strip()]
LM, BG, BL, VQ, MB, GR, Ms = (feats[sub, :K, 0], feats[sub, :K, 1], feats[sub, :K, 2], V[sub], S[sub], G[sub, :, 0], M[sub])
cands = [[x["caption"] for x in beams[i]["beam"]][:K] for i in sub]
idx = np.arange(len(sub))

def rho_q(A):
    return np.nanmean([spearmanr(A[i], Ms[i]).correlation for i in range(len(sub)) if np.std(Ms[i]) > 0 and np.std(A[i]) > 0])
def rho_pair(A, B):
    return np.nanmean([spearmanr(A[i], B[i]).correlation for i in range(len(sub)) if np.std(A[i]) > 0 and np.std(B[i]) > 0])
def approx(score):
    return 100 * Ms[idx, score.argmax(1)].mean()
def write(name, score):
    pick = score.argmax(1)
    recs = [{"image_name": beams[i]["image_name"], "captions": beams[i]["captions"], "prediction": cands[k][int(pick[k])]} for k, i in enumerate(sub)]
    os.makedirs(os.path.join(HERE, "preds", "coco"), exist_ok=True)
    json.dump(recs, open(os.path.join(HERE, "preds", "coco", f"{name}__overall.json"), "w"))

base3 = z(LM) + z(BG) + z(BL)
stack6 = base3 + z(VQ) + z(MB) + z(GR)
print(f"subsample N={len(sub)}  approx CIDEr: z-sum {approx(base3):.1f}  six-agent {approx(stack6):.1f}  six minus VQA {approx(base3+z(MB)+z(GR)):.1f}")
print(f"BLIP-2 FlanT5-XL VQA: rho(quality) {rho_q(VQ):.3f}  rho(BLIP) {rho_pair(VQ, BL):.3f}  rho(bigG) {rho_pair(VQ, BG):.3f}")
rows = []
for path in sorted(glob.glob(os.path.join(HERE, "smalljudge_*_coco20s.jsonl"))):
    tag = os.path.basename(path)[len("smalljudge_"):-len("_coco20s.jsonl")]
    d = {}
    for line in open(path):
        r = json.loads(line); d[r["image_name"]] = r["vqascore"]
    if len(d) < len(sub):
        print(f"{tag}: only {len(d)}/{len(sub)} images, skipped"); continue
    J = np.array([d[n][:K] for n in names], float)
    rq, rb, rblip = rho_q(J), rho_pair(J, VQ), rho_pair(J, BL)
    swap6 = base3 + z(J) + z(MB) + z(GR); plus3 = base3 + z(J); seven = stack6 + z(J)
    print(f"{tag:12s} rho(quality) {rq:.3f}  rho(BLIP-2 VQA) {rb:.3f}  rho(BLIP-ITM) {rblip:.3f} | judge alone {approx(J):.1f}  z-sum+judge {approx(plus3):.1f}  six with VQA->judge {approx(swap6):.1f}  six+judge {approx(seven):.1f}  mean P(yes) {J.mean():.2f}")
    write(f"sj_{tag}_zsum3plus", plus3); write(f"sj_{tag}_stack6swap", swap6); write(f"sj_{tag}_alone", J)
    rows.append(tag)
print("wrote preds for", rows)
