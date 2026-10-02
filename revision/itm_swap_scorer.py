"""Reviewer item on regime: the cross-encoder agent uses Salesforce/blip-itm-base-coco, which was
fine-tuned on COCO train image-caption pairs. Re-score every beam candidate with a BLIP-ITM
checkpoint that never saw COCO caption fine-tuning (default: blip-itm-base-flickr) so the COCO
rows can be recomputed with it. Same read-out as csfn_extract_features.py: softmax(itm)[:,1]."""
import argparse, json, os, time
import torch
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument("--beam_dump", required=True); ap.add_argument("--image_dir", required=True)
ap.add_argument("--out_jsonl", required=True); ap.add_argument("--model", default="Salesforce/blip-itm-base-flickr")
ap.add_argument("--device", default="cuda:1"); ap.add_argument("--batch", type=int, default=20)
a = ap.parse_args()
from transformers import BlipProcessor, BlipForImageTextRetrieval
proc = BlipProcessor.from_pretrained(a.model)
m = BlipForImageTextRetrieval.from_pretrained(a.model, use_safetensors=True).to(a.device).eval()
beams = json.load(open(a.beam_dump)); done = set()
if os.path.exists(a.out_jsonl):
    for l in open(a.out_jsonl):
        try: done.add(json.loads(l)["image_name"])
        except Exception: pass
t0 = time.time(); n = 0
with open(a.out_jsonl, "a") as out:
    for b in beams:
        if b["image_name"] in done: continue
        img = Image.open(os.path.join(a.image_dir, b["image_name"])).convert("RGB")
        caps = [c["caption"] for c in b["beam"]][:20]; sc = []
        for j in range(0, len(caps), a.batch):
            ch = caps[j:j + a.batch]
            inp = proc(images=[img] * len(ch), text=ch, return_tensors="pt", padding=True).to(a.device)
            with torch.no_grad(): sc += m(**inp).itm_score.softmax(-1)[:, 1].float().cpu().tolist()
        out.write(json.dumps({"image_name": b["image_name"], "itm": [round(x, 5) for x in sc]}) + "\n"); n += 1
        if n % 250 == 0: out.flush(); print(f"  {n} {(time.time()-t0)/n:.2f} s/img", flush=True)
print("done", flush=True)
