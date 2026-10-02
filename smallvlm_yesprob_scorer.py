"""Small-VLM yes-probability agent: for every candidate caption in a beam dump, ask a small
vision-language model whether the caption describes the image and read P(yes) from the first
answer token. Resumable jsonl, same format as blip2vqa_scorer.py ({image_name, vqascore}).

Usage (GCP2, rap_venv):
  python smallvlm_yesprob_scorer.py --family smolvlm --model HuggingFaceTB/SmolVLM-500M-Instruct \
      --beam_dump <dump.json> --image_dir <val2014> --out_jsonl smalljudge_smolvlm500m.jsonl --device cuda:1
"""
import argparse, json, os, time
import torch
from PIL import Image

QUESTION = "Does this caption describe the image? Caption: \"{c}\" Answer yes or no."


def load_model(family, name, device, dtype):
    if family == "smolvlm":
        from transformers import AutoProcessor, AutoModelForVision2Seq
        proc = AutoProcessor.from_pretrained(name)
        try:
            proc.image_processor.do_image_splitting = False
        except Exception:
            pass
        model = AutoModelForVision2Seq.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    elif family == "qwen2vl":
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        proc = AutoProcessor.from_pretrained(name, min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28)
        model = Qwen2VLForConditionalGeneration.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    else:
        raise ValueError(family)
    proc.tokenizer.padding_side = "left"
    return proc, model


def yes_no_ids(tok):
    ys, ns = set(), set()
    for v in ("Yes", " Yes", "yes", " yes", "YES"):
        ids = tok.encode(v, add_special_tokens=False)
        if len(ids) == 1: ys.add(ids[0])
    for v in ("No", " No", "no", " no", "NO"):
        ids = tok.encode(v, add_special_tokens=False)
        if len(ids) == 1: ns.add(ids[0])
    assert ys and ns, "no single-token yes/no variants"
    return sorted(ys), sorted(ns)


def build_prompts(family, proc, caps):
    out = []
    for c in caps:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION.format(c=c)}]}]
        out.append(proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) if family == "qwen2vl"
                   else proc.apply_chat_template(msgs, add_generation_prompt=True))
    return out


@torch.no_grad()
def score_image(family, proc, model, img, caps, ys, ns, device, batch):
    scores = []
    for i in range(0, len(caps), batch):
        chunk = caps[i:i + batch]
        prompts = build_prompts(family, proc, chunk)
        if family == "smolvlm":
            inputs = proc(text=prompts, images=[[img]] * len(chunk), return_tensors="pt", padding=True)
        else:
            inputs = proc(text=prompts, images=[img] * len(chunk), return_tensors="pt", padding=True)
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.float32 and model.dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        logits = model(**inputs).logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        py = probs[:, ys].sum(-1); pn = probs[:, ns].sum(-1)
        scores.extend((py / (py + pn + 1e-9)).tolist())
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["smolvlm", "qwen2vl"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--beam_dump", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    proc, model = load_model(args.family, args.model, args.device, dtype)
    ys, ns = yes_no_ids(proc.tokenizer)
    print(f"model {args.model} dtype {dtype} yes ids {ys} no ids {ns}", flush=True)
    beams = json.load(open(args.beam_dump))
    if args.limit: beams = beams[:args.limit]
    done = set()
    if os.path.exists(args.out_jsonl):
        for line in open(args.out_jsonl):
            try: done.add(json.loads(line)["image_name"])
            except Exception: pass
    print(f"{len(beams)} images, {len(done)} already scored", flush=True)
    t0 = time.time(); n_new = 0; nan_count = 0
    with open(args.out_jsonl, "a") as out:
        for i, rec in enumerate(beams):
            name = rec["image_name"]
            if name in done: continue
            img = Image.open(os.path.join(args.image_dir, name)).convert("RGB")
            caps = [c["caption"] for c in rec["beam"]]
            sc = score_image(args.family, proc, model, img, caps, ys, ns, args.device, args.batch)
            if any(s != s for s in sc):
                nan_count += 1; sc = [0.5 if s != s else s for s in sc]
            out.write(json.dumps({"image_name": name, "split": rec.get("split"), "vqascore": sc}) + "\n"); out.flush(); n_new += 1
            if n_new % 100 == 0:
                print(f"  {n_new} new ({i+1}/{len(beams)}), {(time.time()-t0)/n_new:.2f} s/img, nan images {nan_count}", flush=True)
    print(f"done: {n_new} new images in {time.time()-t0:.0f} s ({(time.time()-t0)/max(n_new,1):.2f} s/img), nan images {nan_count}", flush=True)


if __name__ == "__main__":
    main()
