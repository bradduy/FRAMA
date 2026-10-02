"""Reviewer item 1: BLIP-2 FlanT5-XL used directly as a zero-shot captioner (no caption
fine-tuning), on the same images as the beam dump. Writes a prediction file in the format of
preds/<bench>/<rule>__<split>.json. Resumable jsonl cache. fp16 on one T4."""
import argparse, json, os, time
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--beam_dump", required=True); ap.add_argument("--image_dir", required=True)
ap.add_argument("--cache", required=True); ap.add_argument("--out_prefix", required=True)
ap.add_argument("--model", default="Salesforce/blip2-flan-t5-xl"); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--prompt", default="")
a = ap.parse_args()
from transformers import Blip2Processor, Blip2ForConditionalGeneration
proc = Blip2Processor.from_pretrained(a.model)
model = Blip2ForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float16).to(a.device).eval()
beams = json.load(open(a.beam_dump))
done = {}
if os.path.exists(a.cache):
    for l in open(a.cache):
        try: r = json.loads(l); done[r["image_name"]] = r["caption"]
        except Exception: pass
todo = [b for b in beams if b["image_name"] not in done]
print(f"{len(beams)} images, {len(done)} cached", flush=True)
t0 = time.time()
with open(a.cache, "a") as out:
    for k in range(0, len(todo), a.batch):
        chunk = todo[k:k + a.batch]
        imgs = [Image.open(os.path.join(a.image_dir, b["image_name"])).convert("RGB") for b in chunk]
        kw = dict(images=imgs, return_tensors="pt")
        if a.prompt: kw["text"] = [a.prompt] * len(imgs)
        inp = proc(**kw).to(a.device); inp["pixel_values"] = inp["pixel_values"].half()
        with torch.no_grad():
            ids = model.generate(**inp, num_beams=5, max_new_tokens=30, min_new_tokens=5, length_penalty=1.0)
        caps = [c.strip() for c in proc.batch_decode(ids, skip_special_tokens=True)]
        for b, c in zip(chunk, caps):
            c = (a.prompt + " " + c).strip() if a.prompt else c
            done[b["image_name"]] = c; out.write(json.dumps({"image_name": b["image_name"], "caption": c}) + "\n")
        out.flush()
        if (k // a.batch) % 25 == 0:
            n = k + len(chunk); print(f"  {n}/{len(todo)} {(time.time()-t0)/n:.2f} s/img  e.g. {caps[0]}", flush=True)
splits = {"overall": beams}
for b in beams:
    if b.get("split"): splits.setdefault(b["split"], []).append(b)
for sp, lst in splits.items():
    recs = [{"image_name": b["image_name"], "captions": b["captions"], "prediction": done[b["image_name"]]} for b in lst]
    json.dump(recs, open(f"{a.out_prefix}__{sp}.json", "w")); print("wrote", sp, len(recs), flush=True)
