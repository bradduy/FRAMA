"""C2b pilot, fallback compositional scorer: VQAScore-style P(yes) with BLIP-2 FlanT5-XL.

Same recipe as VQAScore (Lin et al. 2024) but with the plain-transformers
Salesforce/blip2-flan-t5-xl checkpoint instead of CLIP-FlanT5, so no extra
package is needed. For each (image, candidate) the prompt is
  "Question: Does this figure show \"<caption>\"? Please answer yes or no. Answer:"
and the score is the softmax probability of the first decoder token being
"yes" against "no". Resumable jsonl, one line per image.

Usage (GCP2, rap_venv, fp16 about 8 GB on a T4):
  python blip2vqa_scorer.py --beam_dump <coco_beam_dump.json> \
      --image_dir ~/ZeroShotImageCap/IFCap/annotations/coco/val2014 \
      --out_jsonl blip2vqa_coco.jsonl --stride 5 --device cuda:0
"""
import argparse, json, os, time
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam_dump", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--model", default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="score every k-th image")
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()

    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    proc = Blip2Processor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16).to(args.device).eval()
    tok = proc.tokenizer
    yes_id = tok("yes", add_special_tokens=False).input_ids[0]
    no_id = tok("no", add_special_tokens=False).input_ids[0]
    print(f"yes_id={yes_id} no_id={no_id}", flush=True)

    beams = json.load(open(args.beam_dump))
    if args.stride > 1:
        beams = beams[:: args.stride]
    if args.limit:
        beams = beams[: args.limit]
    done = set()
    if os.path.exists(args.out_jsonl):
        with open(args.out_jsonl) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["image_name"])
                except Exception:
                    pass
    print(f"{len(beams)} images selected, {len(done)} already scored", flush=True)
    t0 = time.time(); n_new = 0
    dec_start = torch.tensor([[model.config.text_config.decoder_start_token_id]], device=args.device)
    with open(args.out_jsonl, "a") as out:
        for i, rec in enumerate(beams):
            name = rec["image_name"]
            if name in done:
                continue
            img = Image.open(os.path.join(args.image_dir, name)).convert("RGB")
            caps = [c["caption"].strip().rstrip(".") for c in rec["beam"]]
            scores = []
            for j in range(0, len(caps), args.batch):
                chunk = caps[j : j + args.batch]
                prompts = [f'Question: Does this figure show "{c}"? Please answer yes or no. Answer:' for c in chunk]
                inputs = proc(images=[img] * len(chunk), text=prompts, return_tensors="pt", padding=True).to(args.device)
                inputs["pixel_values"] = inputs["pixel_values"].half()
                with torch.no_grad():
                    outp = model(**inputs, decoder_input_ids=dec_start.expand(len(chunk), 1))
                logits = outp.logits[:, 0, :].float()
                two = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=1)
                p_yes = torch.softmax(two, dim=1)[:, 0]
                scores.extend([float(x) for x in p_yes.cpu().tolist()])
            out.write(json.dumps({"image_name": name, "split": rec.get("split"),
                                  "vqascore": [round(x, 5) for x in scores]}) + "\n")
            out.flush(); n_new += 1
            if n_new % 50 == 0:
                el = time.time() - t0
                print(f"  {n_new} new ({i+1}/{len(beams)}), {el/n_new:.2f} s/img", flush=True)
    print(f"done: {n_new} new images in {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
