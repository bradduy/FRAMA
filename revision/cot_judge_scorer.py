"""Reviewer item M3: System 1 versus System 2 use of the same VLM judge. For every candidate the
same Qwen2-VL-2B model is queried in two modes on the same image:
  S1  one forward pass, P(yes) read from the first answer token (the paper's System One verifier);
  S2  chain-of-thought: the model first reasons step by step (greedy, up to --max_new tokens) about
      objects, attributes, counts and relations, then P(yes) is read at 'Final answer:'.
Both modes are timed in the same code path. Resumable jsonl, one line per image."""
import argparse, json, os, time
import torch
from PIL import Image

Q1 = "Does this caption describe the image? Caption: \"{c}\" Answer yes or no."
Q2 = ("Caption: \"{c}\"\nIs this caption an accurate description of the image? Think step by step: list the "
      "objects the caption mentions and check whether each is visible, then check their attributes, counts and "
      "relations. Keep the reasoning short. End with 'Final answer: yes' or 'Final answer: no'.")

ap = argparse.ArgumentParser()
ap.add_argument("--beam_dump", required=True); ap.add_argument("--image_dir", required=True)
ap.add_argument("--out_jsonl", required=True); ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--max_new", type=int, default=96)
ap.add_argument("--stride", type=int, default=2); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
a = ap.parse_args()
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
proc = AutoProcessor.from_pretrained(a.model, min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28)
proc.tokenizer.padding_side = "left"
model = Qwen2VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float16).to(a.device).eval()
tok = proc.tokenizer
def ids(vs): return sorted({tok.encode(v, add_special_tokens=False)[0] for v in vs if len(tok.encode(v, add_special_tokens=False)) == 1})
YS, NS = ids(["Yes", " Yes", "yes", " yes"]), ids(["No", " No", "no", " no"])
def chat(q):
    return proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}], add_generation_prompt=True, tokenize=False)
def inputs_for(texts, img):
    x = proc(text=texts, images=[img] * len(texts), return_tensors="pt", padding=True)
    return {k: (v.to(a.device) if hasattr(v, "to") else v) for k, v in x.items()}
@torch.no_grad()
def p_yes(texts, img):
    lg = model(**inputs_for(texts, img)).logits[:, -1, :].float(); pr = torch.softmax(lg, -1)
    py, pn = pr[:, YS].sum(-1), pr[:, NS].sum(-1); return (py / (py + pn + 1e-9)).tolist()
beams = json.load(open(a.beam_dump))[::a.stride]
beams = [b for k, b in enumerate(beams) if k % a.nshard == a.shard]
done = set()
if os.path.exists(a.out_jsonl):
    for l in open(a.out_jsonl):
        try: done.add(json.loads(l)["image_name"])
        except Exception: pass
print(f"{len(beams)} images in shard, {len(done)} done", flush=True)
n = 0
with open(a.out_jsonl, "a") as out:
    for b in beams:
        if b["image_name"] in done: continue
        img = Image.open(os.path.join(a.image_dir, b["image_name"])).convert("RGB")
        caps = [c["caption"] for c in b["beam"]][:20]
        torch.cuda.synchronize(); t = time.time()
        s1 = p_yes([chat(Q1.format(c=c)) for c in caps], img)
        torch.cuda.synchronize(); t1 = time.time() - t; t = time.time()
        pr = [chat(Q2.format(c=c)) + "Step 1: The caption mentions" for c in caps]
        with torch.no_grad():
            g = model.generate(**inputs_for(pr, img), max_new_tokens=a.max_new, min_new_tokens=24, do_sample=False)
        L = inputs_for(pr, img)["input_ids"].shape[1]
        reas = [tok.decode(x[L:], skip_special_tokens=True) for x in g]
        cut = [r.split("Final answer")[0].rstrip() for r in reas]
        s2 = p_yes([p + r + "\nFinal answer:" for p, r in zip(pr, cut)], img)
        reas = ["Step 1: The caption mentions" + r for r in reas]
        torch.cuda.synchronize(); t2 = time.time() - t
        ntok = float(sum(len(tok.encode(r, add_special_tokens=False)) for r in reas)) / len(reas)
        out.write(json.dumps({"image_name": b["image_name"], "s1": [round(x, 5) for x in s1], "s2": [round(x, 5) for x in s2],
                              "t1": round(t1, 3), "t2": round(t2, 3), "gen_tokens": round(ntok, 1), "example": reas[0][:400]}) + "\n"); out.flush(); n += 1
        if n % 10 == 0: print(f"  {n}  t1 {t1:.2f}s  t2 {t2:.2f}s  tokens {ntok:.0f}", flush=True)
print("done", flush=True)
