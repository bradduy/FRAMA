"""E5 (reviewer 2): inference cost of every fixed configuration, measured on one Tesla T4 in one process.

Every stage of the pipeline is timed separately on the same 500 COCO Karpathy-test images (every 9th image
of the beam dump; 20 further images before them are warm-up and are not counted), with the code paths and
precisions of the paper. A stage loads only its own model; loading time is excluded. Per-image time is wall
clock between two torch.cuda.synchronize() calls and includes reading the image from the page cache, which
is filled before any timing. A configuration's time per image is the sum of its stages for that image; we
report the median and the mean over the 500 images.

Stages
  ret_bigg     bigG image encoding (fp32) and top-9 search over a 566,747 x 1280 caption matrix. The search
               uses a random matrix of the corpus shape in fp16 (the encoded corpus is not on this VM; the
               cost of a matrix product does not depend on its values, and fp16 keeps bigG and the matrix
               within one card).
  ret_blip     BLIP-ITM re-ranking of the nine retrieved captions to five, and the entity count of the five.
  decode       CLIP ViT-B/32 image and text encoding, the IFCap prompt, beam search of width 20.
  bigg32       alignment agent, bigG dual encoder in fp32 as in the paper; bigg16 is the same in fp16.
  blip         alignment agent, BLIP-ITM cross-encoder, 20 candidates.
  s1_blip2     System One verifier, BLIP-2 FlanT5-XL yes-probability, fp16, batches of 10 candidates.
  s1_qwen2b    the 2B System One alternative of tab:smalljudge, Qwen2-VL-2B, fp16, batches of 10.
  ground       noun extraction and one OWLv2 call per image, fp16.
  mbr          beam consensus on the CPU, document frequencies from the training captions.
  blip2cap_b1  BLIP-2 FlanT5-XL used as a captioner, beam width 5, one image per call;
  blip2cap_b8  the same with eight images per call (the setting of the earlier 0.23 s figure), time per
               image = call time / 8.

Resumable per stage: a finished stage is read back from <out_dir>/<stage>.json. Each stage file records
the other compute processes seen on the card before and after it, so exclusivity can be checked.

  python time_configs.py --device cuda:1 --out_dir timing
"""
import argparse, gc, json, os, subprocess, sys, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
IF = os.path.expanduser("~/ZeroShotImageCap/IFCap")
sys.path[:0] = [os.path.join(IF, "src"), V2, HERE]
BEAM_DUMP = f"{IF}/inference_result/coco-indomain-004_biggFull/coco_beam_dump.json"
IMG_DIR = f"{IF}/annotations/coco/val2014"
RET9 = f"{IF}/annotations/retrieved_sentences/caption_coco_image_coco_bigG_9.json"
RET5 = f"{IF}/annotations/retrieved_sentences/caption_coco_image_coco_bigG_blip5.json"
ENT5 = f"{IF}/annotations/retrieved_entity/caption_coco_image_coco_bigG_blip5.json"
CKPT = f"{IF}/checkpoints/train_coco/coco-indomain-004.pt"
VOCAB = f"{IF}/annotations/vocabulary/vgcocooiobjects_v1_class2ind.json"
N_CORPUS, WARM, N_TIMED, STRIDE = 566747, 20, 500, 9

CONFIGS = {
    "captioner_top1": ["ret_bigg", "ret_blip", "decode"],
    "zsum3": ["ret_bigg", "ret_blip", "decode", "bigg32", "blip"],
    "zsum3_s1": ["ret_bigg", "ret_blip", "decode", "bigg32", "blip", "s1_blip2"],
    "six_agent": ["ret_bigg", "ret_blip", "decode", "bigg32", "blip", "s1_blip2", "mbr", "ground"],
    "five_agent_noLM": ["ret_bigg", "ret_blip", "decode", "bigg32", "blip", "s1_blip2", "mbr", "ground"],
    "six_agent_bigg_fp16": ["ret_bigg", "ret_blip", "decode", "bigg16", "blip", "s1_blip2", "mbr", "ground"],
    "six_agent_qwen2b": ["ret_bigg", "ret_blip", "decode", "bigg32", "blip", "s1_qwen2b", "mbr", "ground"],
    "blip2cap_b1": ["blip2cap_b1"],
    "blip2cap_b8": ["blip2cap_b8"],
}


def other_procs(dev):
    idx = dev.split(":")[1] if ":" in dev else "0"
    try:
        out = subprocess.run(["nvidia-smi", "-i", idx, "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return [f"nvidia-smi failed: {e}"]
    return [l.strip() for l in out.splitlines() if l.strip() and l.split(",")[0].strip() != str(os.getpid())]


def img(rec):
    return Image.open(os.path.join(IMG_DIR, rec["image_name"])).convert("RGB")


def caps(rec):
    return [c["caption"] for c in rec["beam"]]


# ---------------------------------------------------------------- stages: setup(dev) -> state; step(state, rec)
def open_clip_bigg(dev, half):
    import open_clip
    m, _, prep = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained="laion2b_s39b_b160k")
    m = (m.half() if half else m).to(dev).eval()
    return m, prep, open_clip.get_tokenizer("ViT-bigG-14")


def setup_ret_bigg(dev):
    m, prep, _ = open_clip_bigg(dev, False)
    g = torch.Generator(device="cpu").manual_seed(0)
    corpus = torch.randn(N_CORPUS, 1280, generator=g, dtype=torch.float32)
    corpus = torch.nn.functional.normalize(corpus, dim=-1).half().to(dev)
    return {"m": m, "prep": prep, "corpus": corpus}


def step_ret_bigg(s, rec, dev):
    x = s["prep"](img(rec)).unsqueeze(0).to(dev)
    e = s["m"].encode_image(x).float()
    e = e / e.norm(2, dim=-1, keepdim=True).clamp_min(1e-6)
    (e.half() @ s["corpus"].T).topk(9, dim=1).indices.cpu()


def setup_blip_itm(dev):
    from transformers import BlipProcessor, BlipForImageTextRetrieval
    name = "Salesforce/blip-itm-base-coco"
    return {"proc": BlipProcessor.from_pretrained(name),
            "m": BlipForImageTextRetrieval.from_pretrained(name, use_safetensors=True).to(dev).eval()}


def blip_scores(s, im, texts, dev, bs=16):
    out = []
    for j in range(0, len(texts), bs):
        inp = s["proc"](images=[im] * len(texts[j:j + bs]), text=texts[j:j + bs], return_tensors="pt",
                        padding=True, truncation=True).to(dev)
        out.extend(s["m"](**inp).itm_score.softmax(dim=-1)[:, 1].cpu().tolist())
    return out


def setup_ret_blip(dev):
    import nltk
    from nltk.stem import WordNetLemmatizer
    s = setup_blip_itm(dev)
    s.update(ret9=json.load(open(RET9)), nltk=nltk, lem=WordNetLemmatizer())
    return s


def step_ret_blip(s, rec, dev):
    cands = s["ret9"][rec["image_name"]]
    sc = blip_scores(s, img(rec), cands, dev)
    top5 = [c for _, c in sorted(zip(sc, cands), key=lambda x: -x[0])][:5]
    cnt = {}
    for c in top5:  # entity_filtering.main for one image
        for w, t in s["nltk"].pos_tag(s["nltk"].word_tokenize(c)):
            if t in ("NN", "NNS"):
                e = s["lem"].lemmatize(w.lower().strip()); cnt[e] = cnt.get(e, 0) + 1
    sorted(((f, w) for w, f in cnt.items()), reverse=True)


def setup_decode(dev):
    import clip
    from argparse import Namespace
    from transformers import GPT2Tokenizer
    from ClipCap import ClipCaptionModel
    args = Namespace(device=dev, k=5)
    model = ClipCaptionModel(args, 10, 10, 512, gpt_type="gpt2", k=5)
    model.load_state_dict(torch.load(CKPT, map_location=dev), strict=False)
    model.to(dev).eval()
    enc, prep = clip.load("ViT-B/32", device=dev)
    return {"model": model, "enc": enc, "prep": prep, "clip": clip, "tok": GPT2Tokenizer.from_pretrained("gpt2"),
            "ret5": json.load(open(RET5)), "ent5": json.load(open(ENT5))}


def step_decode(s, rec, dev):
    from utils import compose_discrete_prompts
    from validation_dumpbeam import beam_search_dump
    name, model = rec["image_name"], s["model"]
    f = s["enc"].encode_image(s["prep"](img(rec)).unsqueeze(0).to(dev)).float()
    f = f / f.norm(2, dim=-1, keepdim=True)
    rt = s["enc"].encode_text(s["clip"].tokenize(s["ret5"][name][:5], truncate=True).to(dev)).float()
    cont = model.mapping_network(f, rt.unsqueeze(0)).view(-1, 10, model.gpt_hidden_size)
    det = [l[1] for l in s["ent5"][name] if l[0] >= 3]
    disc = compose_discrete_prompts(s["tok"], det).unsqueeze(0).to(dev)
    emb = torch.cat((cont, model.gpt.transformer.wte(disc)), dim=1)
    beam_search_dump(emb, s["tok"], model.gpt, beam_width=20)


def setup_bigg(dev, half):
    m, prep, tok = open_clip_bigg(dev, half)
    return {"m": m, "prep": prep, "tok": tok, "half": half}


def step_bigg(s, rec, dev):  # csfn_extract_features.py
    x = s["prep"](img(rec)).unsqueeze(0).to(dev)
    if s["half"]:
        x = x.half()
    i = s["m"].encode_image(x).float(); i = i / i.norm(2, dim=-1, keepdim=True).clamp_min(1e-6)
    t = s["m"].encode_text(s["tok"](caps(rec)).to(dev)).float(); t = t / t.norm(2, dim=-1, keepdim=True).clamp_min(1e-6)
    (i @ t.T).squeeze(0).cpu()


def step_blip(s, rec, dev):
    blip_scores(s, img(rec), caps(rec), dev)


def setup_blip2(dev):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    name = "Salesforce/blip2-flan-t5-xl"
    proc = Blip2Processor.from_pretrained(name)
    m = Blip2ForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
    tok = proc.tokenizer
    return {"proc": proc, "m": m, "yes": tok("yes", add_special_tokens=False).input_ids[0],
            "no": tok("no", add_special_tokens=False).input_ids[0],
            "dec": torch.tensor([[m.config.text_config.decoder_start_token_id]], device=dev)}


def step_s1_blip2(s, rec, dev):  # blip2vqa_scorer.py
    im = img(rec); cs = [c.strip().rstrip(".") for c in caps(rec)]
    for j in range(0, len(cs), 10):
        chunk = cs[j:j + 10]
        prompts = [f'Question: Does this figure show "{c}"? Please answer yes or no. Answer:' for c in chunk]
        inp = s["proc"](images=[im] * len(chunk), text=prompts, return_tensors="pt", padding=True).to(dev)
        inp["pixel_values"] = inp["pixel_values"].half()
        lg = s["m"](**inp, decoder_input_ids=s["dec"].expand(len(chunk), 1)).logits[:, 0, :].float()
        torch.softmax(torch.stack([lg[:, s["yes"]], lg[:, s["no"]]], 1), 1)[:, 0].cpu()


def step_blip2cap(s, recs, dev):  # revision/blip2_caption.py; recs is a list of 1 or 8 records
    inp = s["proc"](images=[img(r) for r in recs], return_tensors="pt").to(dev)
    inp["pixel_values"] = inp["pixel_values"].half()
    ids = s["m"].generate(**inp, num_beams=5, max_new_tokens=30, min_new_tokens=5, length_penalty=1.0)
    s["proc"].batch_decode(ids, skip_special_tokens=True)


def setup_qwen2b(dev):
    import smallvlm_yesprob_scorer as Q
    proc, m = Q.load_model("qwen2vl", "Qwen/Qwen2-VL-2B-Instruct", dev, torch.float16)
    ys, ns = Q.yes_no_ids(proc.tokenizer)
    return {"Q": Q, "proc": proc, "m": m, "ys": ys, "ns": ns}


def step_s1_qwen2b(s, rec, dev):
    s["Q"].score_image("qwen2vl", s["proc"], s["m"], img(rec), caps(rec), s["ys"], s["ns"], dev, 10)


def setup_ground(dev):
    import detector_ground_scorer as D
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    name = "google/owlv2-base-patch16-ensemble"
    pos_tag, wt, lem = D.load_nltk()
    return {"D": D, "nl": (pos_tag, wt, lem), "vocab": D.load_vocab(VOCAB), "proc": Owlv2Processor.from_pretrained(name),
            "m": Owlv2ForObjectDetection.from_pretrained(name).to(dev).eval().half()}


def step_ground(s, rec, dev):  # detector_ground_scorer.py
    nouns = set()
    for c in caps(rec):
        n, _ = s["D"].candidate_nouns(c, *s["nl"], s["vocab"]); nouns.update(n)
    q = sorted(nouns)[:40]
    if not q:
        return
    inp = s["proc"](text=[[f"a photo of a {x}" for x in q]], images=img(rec), return_tensors="pt").to(dev)
    inp["pixel_values"] = inp["pixel_values"].half()
    p = torch.sigmoid(s["m"](**inp).logits[0].float())
    p.max(dim=0).values.cpu(), (p > 0.3).sum(dim=0).cpu()


def setup_mbr(dev):
    import mbr_train_idf as M
    if os.path.exists(f"{IF}/annotations/coco/coco_with_entities.pickle"):  # pilot_headroom.ROOT is the laptop path
        M.CORPUS["coco"] = f"{IF}/annotations/coco/coco_with_entities.pickle"
    df, logn = M.build_df("coco"); M.init(df, logn)
    return {"M": M}


def step_mbr(s, rec, dev):
    s["M"].beam_scores(caps(rec))


STAGES = {
    "ret_bigg": (setup_ret_bigg, step_ret_bigg),
    "ret_blip": (setup_ret_blip, step_ret_blip),
    "decode": (setup_decode, step_decode),
    "bigg32": (lambda d: setup_bigg(d, False), step_bigg),
    "bigg16": (lambda d: setup_bigg(d, True), step_bigg),
    "blip": (setup_blip_itm, step_blip),
    "s1_blip2": (setup_blip2, step_s1_blip2),
    "s1_qwen2b": (setup_qwen2b, step_s1_qwen2b),
    "ground": (setup_ground, step_ground),
    "mbr": (setup_mbr, step_mbr),
    "blip2cap_b1": (setup_blip2, None),
    "blip2cap_b8": (setup_blip2, None),
}


def sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize(dev)


@torch.no_grad()
def run_stage(name, recs, dev, out_dir):
    path = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(path):
        return json.load(open(path))["times"]
    setup, step = STAGES[name]
    before = other_procs(dev); t_setup = time.time()
    s = setup(dev); t_setup = time.time() - t_setup
    times = []
    if name == "blip2cap_b8":
        for k in range(0, len(recs), 8):
            chunk = recs[k:k + 8]
            sync(dev); t = time.perf_counter(); step_blip2cap(s, chunk, dev); sync(dev)
            times.extend([(time.perf_counter() - t) / len(chunk)] * len(chunk))
    else:
        for rec in recs:
            sync(dev); t = time.perf_counter()
            step_blip2cap(s, [rec], dev) if name == "blip2cap_b1" else step(s, rec, dev)
            sync(dev); times.append(time.perf_counter() - t)
    times = times[WARM:]
    rec = {"stage": name, "device": dev, "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "n_timed": len(times), "warmup": WARM, "median_s": float(np.median(times)), "mean_s": float(np.mean(times)),
           "setup_s": t_setup, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "boot_id": open("/proc/sys/kernel/random/boot_id").read().strip(),
           "other_procs_before": before, "other_procs_after": other_procs(dev), "times": times}
    json.dump(rec, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
    print(f"{name:12s} median {rec['median_s']:.3f} s  mean {rec['mean_s']:.3f} s  (setup {t_setup:.0f} s)  "
          f"others before/after: {len(before)}/{len(rec['other_procs_after'])}", flush=True)
    del s; gc.collect(); torch.cuda.empty_cache()
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "timing"))
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--smoke", type=int, default=0, help="time only this many images after 2 warm-up images")
    a = ap.parse_args()
    global WARM, N_TIMED
    if a.smoke:
        WARM, N_TIMED = 2, a.smoke
    os.makedirs(a.out_dir, exist_ok=True)
    torch.cuda.init(); torch.zeros(1, device=a.device)  # hold the card from the start
    recs = json.load(open(BEAM_DUMP))[::STRIDE][:WARM + N_TIMED]
    for r in recs:  # fill the page cache so no stage pays for the first disk read
        open(os.path.join(IMG_DIR, r["image_name"]), "rb").read()
    T = {n: np.array(run_stage(n, recs, a.device, a.out_dir)) for n in a.stages.split(",")}
    summary = {"device": a.device, "gpu": torch.cuda.get_device_name(a.device), "n_images": N_TIMED,
               "images": [r["image_name"] for r in recs[WARM:]],
               "stages": {n: {"median_s": float(np.median(t)), "mean_s": float(np.mean(t))} for n, t in T.items()},
               "configs": {}}
    for c, st in CONFIGS.items():
        if all(x in T for x in st):
            tot = sum(T[x] for x in st)
            summary["configs"][c] = {"stages": st, "median_s": float(np.median(tot)), "mean_s": float(np.mean(tot))}
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "images"}, indent=1))


if __name__ == "__main__":
    main()
