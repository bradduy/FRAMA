"""E6c on GCP2: check every SPICE tuple of every selected caption against the image with Qwen3-VL-4B-Instruct.

The verifier is not a FRAMA agent. For each fact the prompt holds the caption and the fact; the read-out is
P(yes) / (P(yes) + P(no)) over the first answer token, both capitalisations pooled. Images are resized so that
the longer side is at most 448 pixels. Work is split per unit (thumb, coco, nocaps, flickr) into chunks of images;
a worker claims a chunk with an atomic mkdir, appends to <chunk>.jsonl.partial and renames it to <chunk>.jsonl when
the chunk is complete, so two workers can share the queue and a restarted worker resumes where it stopped.
run_e6.sh removes stale locks at boot, before any worker starts.

  python e6c_verify.py --device cuda:0
"""
import argparse, glob, json, math, os, sys, time
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); IF = os.path.expanduser("~/ZeroShotImageCap/IFCap/annotations")
IMG = {"thumb": f"{IF}/coco/val2014", "thumb2": f"{IF}/coco/val2014", "coco": f"{IF}/coco/val2014", "nocaps": f"{IF}/nocaps/images",
       "flickr": f"{IF}/flickr30k/flickr30k-images"}
UNITS = ["thumb", "coco", "nocaps", "flickr", "thumb2"]
NUM = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "several",
       "many", "few", "multiple", "couple", "dozen", "numerous", "both", "pair"}
NCH = 24


def claim(t):
    if len(t) == 1:
        return f"there is a {t[0]} in the image"
    if len(t) == 2:
        o, a = t
        return f"there are {a} {o}" if (a in NUM or a.isdigit()) else f"the {o} is {a}"
    return f"the {t[0]} {t[1]} the {t[2]}"


def prompt(caption, t):
    return (f'A caption of this image reads: "{caption}"\n'
            f"A parser extracted this fact from the caption: {claim(t)}.\n"
            "The extraction may be worded clumsily; judge the part of the caption that the fact refers to.\n"
            "Is that part of the caption true for the image? Answer with yes or no.")


def work(unit):
    """Deterministic list of (image_name, caption, tuples) for a unit, or None while SPICE outputs are missing."""
    ins = sorted(glob.glob(os.path.join(HERE, "spice_in", f"{unit}__*.json")))
    ins = [p for p in ins if not p.endswith(".names.json")]
    outs = [os.path.join(HERE, "spice_out", os.path.basename(p)) for p in ins]
    if not ins or not all(os.path.exists(o) for o in outs):
        return None
    seen = {}
    for pi, po in zip(ins, outs):
        names = json.load(open(pi[:-5] + ".names.json")); src = json.load(open(pi)); res = json.load(open(po))
        rid = {r["image_id"]: r for r in res}
        for s in src:
            key = (names[s["image_id"]], s["test"])
            if key not in seen:
                seen[key] = [x["tuple"] for x in rid[s["image_id"]]["test_tuples"]]
    return sorted((k[0], k[1], v) for k, v in seen.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_side", type=int, default=448)
    ap.add_argument("--test", type=int, default=0, help="score the first n facts of the thumb unit and exit")
    a = ap.parse_args()
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float16, device_map={"": a.device},
                                                            attn_implementation="sdpa").eval()
    proc = AutoProcessor.from_pretrained(a.model); proc.tokenizer.padding_side = "left"
    tk = proc.tokenizer
    yes = [tk.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
    no = [tk.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]
    tmpl = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "QQQ"}]}],
                                    tokenize=False, add_generation_prompt=True)

    def load(unit, im):
        img = Image.open(os.path.join(IMG[unit], im)).convert("RGB")
        s = a.max_side / max(img.size)
        return img.resize((max(28, round(img.width * s)), max(28, round(img.height * s)))) if s < 1 else img

    @torch.no_grad()
    def score(img, qs):
        out = []
        for i in range(0, len(qs), a.batch):
            q = qs[i:i + a.batch]
            x = proc(text=[tmpl.replace("QQQ", s) for s in q], images=[img] * len(q), return_tensors="pt", padding=True).to(a.device)
            lg = model(**x).logits[:, -1].float()
            ly, ln = torch.logsumexp(lg[:, yes], -1), torch.logsumexp(lg[:, no], -1)
            p = torch.sigmoid(ly - ln).cpu().tolist()
            assert not any(math.isnan(v) for v in p), "NaN read-out"
            out += p
        return out

    if a.test:
        W = work("thumb"); n = 0; t0 = time.time()
        for im, cap, tups in W:
            img = load("thumb", im); ps = score(img, [prompt(cap, t) for t in tups])
            for t, p in zip(tups, ps):
                print(f"{p:.3f}  {claim(t):40s} | {cap}")
            n += len(tups)
            if n >= a.test:
                break
        print(f"{n} facts in {time.time() - t0:.1f} s"); return

    for unit in UNITS:
        while (W := work(unit)) is None:
            time.sleep(60)
        od = os.path.join(HERE, "e6c_out", unit); os.makedirs(od, exist_ok=True)
        ims = sorted({w[0] for w in W}); per = math.ceil(len(ims) / NCH)
        chunk_of = {im: i // per for i, im in enumerate(ims)}
        for c in range(NCH):
            fin = os.path.join(od, f"chunk_{c:02d}.jsonl")
            if os.path.exists(fin):
                continue
            try:
                os.mkdir(fin + ".lock")
            except FileExistsError:
                continue
            part = fin + ".partial"; done = set()
            if os.path.exists(part):
                done = {(r["image_name"], r["caption"]) for r in map(json.loads, open(part))}
            mine = [w for w in W if chunk_of[w[0]] == c and (w[0], w[1]) not in done]
            t0 = time.time()
            with open(part, "a") as f:
                cur, img = None, None
                for im, cap, tups in mine:
                    if im != cur:
                        cur, img = im, load(unit, im)
                    ps = score(img, [prompt(cap, t) for t in tups]) if tups else []
                    f.write(json.dumps({"image_name": im, "caption": cap, "tuples": tups, "p_yes": ps}) + "\n"); f.flush()
            os.replace(part, fin)
            print(f"{unit} chunk {c} done: {len(mine)} captions in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
