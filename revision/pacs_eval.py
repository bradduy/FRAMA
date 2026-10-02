"""Reviewer item m2: learned reference-free metric. PAC-S++ (Sarto et al., IJCV 2025) with its
LoRA-tuned CLIP ViT-L/14 checkpoint and RefPAC-S++, computed with the authors' PACScore and
RefPACScore functions (w = 2.0). Run from inside a clone of aimagelab/pacscore."""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.expanduser("~/ZeroShotImageCap/IFCap/evaluation"))
from models.clip_lora import clip_lora
from evaluation import PACScore, RefPACScore
ap = argparse.ArgumentParser(); ap.add_argument("--image_dir", required=True); ap.add_argument("--preds", nargs="+", required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
dev = "cuda"
model, preprocess = clip_lora.load("ViT-L/14", device=dev, lora=4)
model = model.to(dev).float(); model.load_state_dict(torch.load("checkpoints/PAC++_clip_ViT-L-14.pth", map_location=dev)["state_dict"]); model.eval()
res = {}
for f in a.preds:
    d = json.load(open(f))
    ims = [os.path.join(a.image_dir, r["image_name"]) for r in d]; cands = [r["prediction"] for r in d]; refs = [list(r["captions"]) for r in d]
    with torch.no_grad():
        _, pac, feats, lens = PACScore(model, preprocess, ims, cands, dev, w=2.0)
        _, tt = RefPACScore(model, refs, feats, dev, torch.tensor(lens))
    ref = 2 * pac * tt / (pac + tt)
    res[os.path.basename(f)] = {"n": len(d), "PAC-S++": round(100 * float(np.mean(pac)), 2), "RefPAC-S++": round(100 * float(np.mean(ref)), 2)}
    print(os.path.basename(f), res[os.path.basename(f)], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
