"""Reviewer item 5: reference-free metric. CLIP-S = 2.5 * max(cos(img, cap), 0) and RefCLIP-S
(harmonic mean with the max cosine to the references), Hessel et al. 2021, with OpenAI CLIP
ViT-B/32, a backbone that no agent of the selector uses (the selector uses ViT-bigG/14 and BLIP).
Scores every prediction file given on the command line; writes one json of means."""
import argparse, json, os, glob
import torch, numpy as np
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("--image_dir", required=True); ap.add_argument("--preds", nargs="+", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:1")
ap.add_argument("--model", default="openai/clip-vit-base-patch32")
a = ap.parse_args()
from transformers import CLIPModel, CLIPProcessor
m = CLIPModel.from_pretrained(a.model, use_safetensors=True).to(a.device).eval(); p = CLIPProcessor.from_pretrained(a.model)
def txt(caps):
    out = []
    for k in range(0, len(caps), 256):
        t = p(text=["A photo depicts " + c for c in caps[k:k+256]], return_tensors="pt", padding=True, truncation=True, max_length=77).to(a.device)
        with torch.no_grad(): e = m.get_text_features(**t)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
    return torch.cat(out)
first = json.load(open(a.preds[0])); names = [r["image_name"] for r in first]
imgE = []
for k in range(0, len(names), 64):
    ims = [Image.open(os.path.join(a.image_dir, n)).convert("RGB") for n in names[k:k+64]]
    with torch.no_grad(): e = m.get_image_features(**p(images=ims, return_tensors="pt").to(a.device))
    imgE.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
imgE = torch.cat(imgE); idx = {n: i for i, n in enumerate(names)}
refs = [r["captions"] for r in first]; flat = [c for rs in refs for c in rs]; RE = txt(flat)
offs = np.cumsum([0] + [len(r) for r in refs])
res = {}
for f in a.preds:
    d = json.load(open(f)); ii = torch.tensor([idx[r["image_name"]] for r in d])
    cE = txt([r["prediction"] for r in d]); cos = (cE * imgE[ii]).sum(-1)
    clips = 2.5 * torch.clamp(cos, min=0)
    refmax = torch.stack([(RE[offs[i]:offs[i+1]] @ cE[k]).max() for k, i in enumerate(ii.tolist())])
    refmax = torch.clamp(refmax, min=0)
    ref = 2 * clips * refmax / (clips + refmax + 1e-8)
    res[os.path.basename(f)] = {"n": len(d), "CLIP-S": round(100 * clips.mean().item(), 2), "RefCLIP-S": round(100 * ref.mean().item(), 2)}
    print(os.path.basename(f), res[os.path.basename(f)], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
