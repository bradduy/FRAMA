"""Extract per-beam-candidate features for the Cross-Scorer Fusion head.

Inputs: a beam-20 dump JSON (each item has image_name, beam=[{caption, score (LM logprob)}, ...]).
Outputs: features.npz with arrays:
  image_names: (N,) str
  captions:    (N, 20) str
  feats:       (N, 20, 5) float32  -- [LM_lp, bigG_score, BLIP_score, beam_rank, length_norm]
  lm_z:        (N, 20) float32     -- z-normalised LM log-prob (kept for fixed-alpha comparison)
  bigG_z:      (N, 20) float32     -- z-normalised bigG CLIPScore
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam_dump", required=True)
    ap.add_argument("--image_folder", default="annotations/coco/val2014")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bigG_model", default="ViT-bigG-14")
    ap.add_argument("--bigG_pretrained", default="laion2b_s39b_b160k")
    ap.add_argument("--blip_model", default="Salesforce/blip-itm-base-coco")
    args = ap.parse_args()

    print("[load] open_clip bigG-14", flush=True)
    bigG, _, prep_g = open_clip.create_model_and_transforms(args.bigG_model, pretrained=args.bigG_pretrained)
    bigG = bigG.to(args.device).eval()
    tok_g = open_clip.get_tokenizer(args.bigG_model)

    print("[load] BLIP-ITM", flush=True)
    from transformers import BlipProcessor, BlipForImageTextRetrieval
    blip_proc = BlipProcessor.from_pretrained(args.blip_model)
    blip = BlipForImageTextRetrieval.from_pretrained(args.blip_model, use_safetensors=True).to(args.device).eval()

    with open(args.beam_dump) as f:
        beams = json.load(f)
    N = len(beams)
    K = max(len(r["beam"]) for r in beams)
    print(f"[load] {N} images, beam-{K}", flush=True)

    image_names = [r["image_name"] for r in beams]
    captions = np.empty((N, K), dtype=object)
    feats = np.zeros((N, K, 5), dtype=np.float32)

    for i, rec in enumerate(tqdm(beams, desc="features")):
        beam = rec["beam"][:K]
        # Pad if beam shorter than K
        while len(beam) < K:
            beam.append({"caption": "", "score": -100.0})
        caps = [b["caption"] for b in beam]
        lm_lp = np.array([b["score"] for b in beam], dtype=np.float32)
        captions[i] = caps

        # Open image once
        path = os.path.join(args.image_folder, rec["image_name"])
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"[warn] {rec['image_name']}: {exc}", flush=True)
            continue

        # bigG image + text features
        x = prep_g(img).unsqueeze(0).to(args.device)
        i_g = bigG.encode_image(x).float()
        i_g = i_g / i_g.norm(2, dim=-1, keepdim=True).clamp_min(1e-6)
        t_tok = tok_g(caps).to(args.device)
        t_g = bigG.encode_text(t_tok).float()
        t_g = t_g / t_g.norm(2, dim=-1, keepdim=True).clamp_min(1e-6)
        bigG_score = (i_g @ t_g.T).squeeze(0).cpu().numpy()  # (K,)

        # BLIP-ITM on (image, candidate)
        blip_score = np.zeros(K, dtype=np.float32)
        bs = 16
        for j in range(0, K, bs):
            batch = caps[j : j + bs]
            inputs = blip_proc(images=[img] * len(batch), text=batch,
                               return_tensors="pt", padding=True, truncation=True).to(args.device)
            o = blip(**inputs)
            s = o.itm_score.softmax(dim=-1)[:, 1]
            blip_score[j : j + len(batch)] = s.cpu().numpy()

        # length
        lengths = np.array([len(c.split()) for c in caps], dtype=np.float32)
        len_norm = (lengths - lengths.mean()) / (lengths.std() + 1e-6)

        # beam rank: 0..K-1
        ranks = np.arange(K, dtype=np.float32) / (K - 1)  # 0..1, lower is better-by-LM

        feats[i, :, 0] = lm_lp
        feats[i, :, 1] = bigG_score
        feats[i, :, 2] = blip_score
        feats[i, :, 3] = ranks
        feats[i, :, 4] = len_norm

    # also compute per-image z-normalised LM and bigG scores for the fixed-alpha baseline
    lm_z = (feats[..., 0] - feats[..., 0].mean(axis=1, keepdims=True)) / (feats[..., 0].std(axis=1, keepdims=True) + 1e-6)
    bigG_z = (feats[..., 1] - feats[..., 1].mean(axis=1, keepdims=True)) / (feats[..., 1].std(axis=1, keepdims=True) + 1e-6)

    np.savez_compressed(args.out_npz,
                        image_names=np.array(image_names),
                        captions=captions,
                        feats=feats, lm_z=lm_z, bigG_z=bigG_z)
    print(f"\nwrote {args.out_npz}; shape={feats.shape}", flush=True)


if __name__ == "__main__":
    main()
