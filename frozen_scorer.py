"""C3 verifier ladder: score every beam candidate with a frozen image-text scorer.

Supported scorers (plain transformers, no extra packages):
  siglip2   : google/siglip2-so400m-patch14-384, dual encoder, cosine (Tier A)
  clipl14   : openai/clip-vit-large-patch14, dual encoder, cosine (Tier A)
  blip2itm  : Salesforce/blip2-itm-vit-g, cross-attention ITM head, P(match) (Tier B)
Output: one jsonl line per image, {"image_name", "split", "scores": [K floats]}.
Resumable: images already present in --out_jsonl are skipped.

Usage (GCP2, rap_venv):
  python frozen_scorer.py --scorer siglip2 --beam_dump <dump> --image_dir <dir> \
      --out_jsonl scores_siglip2_coco.jsonl --device cuda:0
"""
import argparse, json, os, time
import torch
from PIL import Image

MODELS = {
    "siglip2": "google/siglip2-so400m-patch14-384",
    "clipl14": "openai/clip-vit-large-patch14",
    "blip2itm": "Salesforce/blip2-itm-vit-g",
}


def load(scorer, device):
    name = MODELS[scorer]
    if scorer == "blip2itm":
        from transformers import Blip2ForImageTextRetrieval, Blip2Processor
        model = Blip2ForImageTextRetrieval.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
        proc = Blip2Processor.from_pretrained(name)
    else:
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
        proc = AutoProcessor.from_pretrained(name)
    return model, proc


@torch.no_grad()
def score(scorer, model, proc, img, caps, device):
    if scorer == "blip2itm":
        out = []
        for j in range(0, len(caps), 10):
            chunk = caps[j : j + 10]
            inputs = proc(images=[img] * len(chunk), text=chunk, return_tensors="pt", padding=True).to(device)
            inputs["pixel_values"] = inputs["pixel_values"].half()
            logits = model(**inputs, use_image_text_matching_head=True).logits_per_image  # (n, 2)
            out.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
        return out
    # dual encoders: cosine between image and each caption embedding
    if scorer == "siglip2":
        ii = proc(images=img, return_tensors="pt").to(device)
        ti = proc(text=caps, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
        ii["pixel_values"] = ii["pixel_values"].half()
        ie = model.get_image_features(**ii); te = model.get_text_features(**ti)
    else:
        ii = proc(images=img, return_tensors="pt").to(device)
        ti = proc(text=caps, padding=True, truncation=True, return_tensors="pt").to(device)
        ii["pixel_values"] = ii["pixel_values"].half()
        ie = model.get_image_features(**ii); te = model.get_text_features(**ti)
    ie = torch.nn.functional.normalize(ie.float(), dim=-1); te = torch.nn.functional.normalize(te.float(), dim=-1)
    return (te @ ie.T).squeeze(-1).cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", required=True, choices=list(MODELS))
    ap.add_argument("--beam_dump", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    model, proc = load(args.scorer, args.device)
    beams = json.load(open(args.beam_dump))
    if args.stride > 1:
        beams = beams[:: args.stride]
    if args.limit:
        beams = beams[: args.limit]
    done = set()
    if os.path.exists(args.out_jsonl):
        for line in open(args.out_jsonl):
            try:
                done.add(json.loads(line)["image_name"])
            except Exception:
                pass
    print(f"{args.scorer}: {len(beams)} images, {len(done)} already scored", flush=True)
    t0 = time.time(); n = 0
    with open(args.out_jsonl, "a") as out:
        for rec in beams:
            name = rec["image_name"]
            if name in done:
                continue
            img = Image.open(os.path.join(args.image_dir, name)).convert("RGB")
            caps = [c["caption"].strip() for c in rec["beam"]]
            s = score(args.scorer, model, proc, img, caps, args.device)
            out.write(json.dumps({"image_name": name, "split": rec.get("split"), "scores": [round(float(x), 6) for x in s]}) + "\n")
            out.flush(); n += 1
            if n % 200 == 0:
                print(f"  {n} new, {(time.time()-t0)/n:.2f} s/img", flush=True)
    print(f"done: {n} new images in {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
