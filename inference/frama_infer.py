"""FRAMA inference: caption your own images with the frozen IFCap COCO captioner and the
six-agent adjudicator of "A Reranking Reality Check for Zero-Shot Image Captioning:
Leakage-Free Oracles and System One Adjudication".

    python frama_infer.py --images photo1.jpg photo2.jpg --out captions.json
    python frama_infer.py --image_dir my_photos/ --out captions.json --config six --dump_beam

Pipeline, the paper's COCO setting end to end:
  1. ViT-bigG/14 retrieves 9 captions from the 566,747 COCO training captions.
  2. BLIP-ITM keeps the best 5; nouns occurring at least 3 times in them form the hard prompt.
  3. The IFCap COCO decoder, CLIP ViT-B/32 with GPT-2, emits a beam of 20 candidates.
  4. Six frozen agents score every candidate: LM log-likelihood, bigG cosine, BLIP-ITM match
     probability, BLIP-2 FlanT5-XL yes-probability, beam-internal MBR consensus, OWLv2 grounding.
  5. Scores are z-normalised within the beam and summed with equal weights.
     --config five, the default, drops the LM agent: the better configuration off COCO.

Stages run one after another over all images and release their models before the next stage
loads, so one 16 GB GPU is enough. The IFCap source, pinned to one commit, and its COCO
checkpoint are fetched from the authors' GitHub on first use; the other models come from the
Hugging Face Hub.
"""
import argparse, gc, glob, gzip, hashlib, io, json, math, os, re, shutil, sys, urllib.request, zipfile
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
sys.path.insert(0, HERE)

IFCAP_COMMIT = "84602a73885432ef8db8eafbff05fa24a6566a3f"
IFCAP_SRC_ZIP = f"https://codeload.github.com/boreng0817/IFCap/zip/{IFCAP_COMMIT}"
IFCAP_CKPT_ZIP = "https://github.com/boreng0817/IFCap/releases/download/v1.0/checkpoints.zip"
IFCAP_CKPT_MEMBER = "checkpoints/train_coco/coco-indomain-004.pt"
IFCAP_CKPT_SHA256 = "a689768e89c0bf6cba7d5fd68cd0b7e95489fb3f1940f11d8cff09443b842e55"
IFCAP_ANN_ZIP = "https://github.com/boreng0817/IFCap/releases/download/v1.0/annotations.zip"
IFCAP_VOCAB_MEMBER = "annotations/vocabulary/vgcocooiobjects_v1_class2ind.json"
IFCAP_VOCAB_SHA256 = "f000ebce6eb1f34508a97150b220bc7bb0a11be4ec9f74fb80bd993598de3b62"

BIGG = ("ViT-bigG-14", "laion2b_s39b_b160k")
BLIP_ITM = "Salesforce/blip-itm-base-coco"
BLIP2 = "Salesforce/blip2-flan-t5-xl"
OWLV2 = "google/owlv2-base-patch16-ensemble"
AGENTS = ("LM", "bigG", "BLIP", "VQA", "MBR", "ground")
CONFIGS = {"five": ("bigG", "BLIP", "VQA", "MBR", "ground"), "six": AGENTS}
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def log(msg):
    print(f"[frama] {msg}", flush=True)


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def open_rgb(path):
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------- IFCap source and checkpoint

class _HttpRange(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests, so one member of
    the 1.96 GB IFCap checkpoint archive can be extracted without downloading the rest."""

    def __init__(self, url):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.url, self.size = r.geturl(), int(r.headers["Content-Length"])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_member(url, member, dest, expected_sha):
    raw = io.BufferedReader(_HttpRange(url), buffer_size=16 << 20)
    with zipfile.ZipFile(raw) as z, z.open(member) as fin, open(dest + ".part", "wb") as fout:
        shutil.copyfileobj(fin, fout, 16 << 20)
    got = sha256(dest + ".part")
    if got != expected_sha:
        raise RuntimeError(f"checksum mismatch for {member}: {got}")
    os.replace(dest + ".part", dest)


def fetch_ifcap(cache, ifcap_root=None, ckpt=None):
    """Return (IFCap src dir, COCO checkpoint, OWLv2 noun vocabulary), downloading what is
    missing into cache from the IFCap repository and its v1.0 release."""
    os.makedirs(cache, exist_ok=True)
    src = os.path.join(ifcap_root, "src") if ifcap_root else os.path.join(cache, f"IFCap-{IFCAP_COMMIT}", "src")
    if not os.path.exists(os.path.join(src, "ClipCap.py")):
        log(f"downloading IFCap source at commit {IFCAP_COMMIT[:7]}")
        data = urllib.request.urlopen(IFCAP_SRC_ZIP).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(cache)
    ckpt = ckpt or os.path.join(cache, "coco-indomain-004.pt")
    if not os.path.exists(ckpt):
        log("extracting coco-indomain-004.pt from the IFCap v1.0 release, about 650 MB")
        _extract_member(IFCAP_CKPT_ZIP, IFCAP_CKPT_MEMBER, ckpt, IFCAP_CKPT_SHA256)
    vocab = os.path.join(ifcap_root, IFCAP_VOCAB_MEMBER) if ifcap_root else os.path.join(cache, os.path.basename(IFCAP_VOCAB_MEMBER))
    if not os.path.exists(vocab):
        log("extracting the object vocabulary from the IFCap v1.0 annotations")
        _extract_member(IFCAP_ANN_ZIP, IFCAP_VOCAB_MEMBER, vocab, IFCAP_VOCAB_SHA256)
    return src, ckpt, vocab


# ---------------------------------------------------------------- stage 1: bigG retrieval

@torch.no_grad()
def stage_retrieve(recs, dev, top_k=9, batch=16, chunk=65536):
    import open_clip
    caps = json.load(open(os.path.join(ASSETS, "coco_train_captions.json")))
    corpus = torch.from_numpy(np.load(os.path.join(ASSETS, "coco_train_captions_bigG14_fp16.npy"))).to(dev)
    model, _, prep = open_clip.create_model_and_transforms(*BIGG)
    model = model.to(dev).eval()
    for i in range(0, len(recs), batch):
        part = recs[i : i + batch]
        x = torch.stack([prep(open_rgb(r["path"])) for r in part]).to(dev)
        e = model.encode_image(x).float()
        e = e / e.norm(2, dim=-1, keepdim=True).clamp_min(1e-6)
        sims = torch.cat([e @ corpus[j : j + chunk].float().T for j in range(0, corpus.shape[0], chunk)], dim=1)
        top = sims.topk(top_k, dim=1).indices.cpu().tolist()
        for r, idx in zip(part, top):
            r["retrieved"] = [caps[k] for k in idx]
    del corpus
    release()
    return model, prep, open_clip.get_tokenizer(BIGG[0])


# ---------------------------------------------------------------- stage 2: BLIP-ITM keeps 5, entities

def load_blip_itm(dev):
    from transformers import BlipForImageTextRetrieval, BlipProcessor
    proc = BlipProcessor.from_pretrained(BLIP_ITM)
    model = BlipForImageTextRetrieval.from_pretrained(BLIP_ITM, use_safetensors=True).to(dev).eval()
    return model, proc


@torch.no_grad()
def blip_itm_scores(model, proc, img, caps, dev, bs=16):
    out = []
    for j in range(0, len(caps), bs):
        part = caps[j : j + bs]
        inputs = proc(images=[img] * len(part), text=part, return_tensors="pt", padding=True, truncation=True).to(dev)
        out.extend(model(**inputs).itm_score.softmax(dim=-1)[:, 1].cpu().tolist())
    return out


def frequent_nouns(caps, min_count=3):
    """IFCap frequency-based entity filtering: lemmatised NN/NNS tokens of the kept captions,
    sorted by (count, word) descending, kept when they occur at least min_count times."""
    from nltk import pos_tag, word_tokenize
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    counts = {}
    for cap in caps:
        for w, t in pos_tag(word_tokenize(cap)):
            if t in ("NN", "NNS"):
                e = lem.lemmatize(w.lower().strip())
                counts[e] = counts.get(e, 0) + 1
    ranked = sorted(((f, w) for w, f in counts.items()), reverse=True)
    return [w for f, w in ranked if f >= min_count]


def stage_rerank(recs, blip, proc, dev, keep=5):
    for r in recs:
        s = blip_itm_scores(blip, proc, open_rgb(r["path"]), r["retrieved"], dev)
        ranked = sorted(zip(s, r["retrieved"]), key=lambda x: -x[0])
        r["kept"] = [c for _, c in ranked[:keep]]
        r["entities"] = frequent_nouns(r["kept"])


# ---------------------------------------------------------------- stage 3: IFCap decoder, beam 20

@torch.no_grad()
def beam_search_dump(embeddings, tokenizer, model, beam_width=20, max_len=64, end_of_sentences=(".", " .")):
    """Beam search of IFCap's search.beam_search, returning every candidate with its
    length-normalised log-probability, best first."""
    device = model.device
    eos = [tokenizer.encode(e)[-1] for e in end_of_sentences]
    scores = None
    seq_lengths = torch.ones(beam_width, device=device)
    is_stopped = torch.zeros(beam_width, device=device, dtype=torch.bool)
    tokens = None
    generated = embeddings.float()
    for _ in range(max_len):
        out = model(inputs_embeds=generated.type(model.dtype))
        logits = out.logits[:, -1, :].softmax(-1).log()
        if scores is None:
            scores, next_tokens = logits.topk(beam_width, -1)
            generated = generated.expand(beam_width, *generated.shape[1:])
            next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
            tokens = next_tokens if tokens is None else torch.cat((tokens, next_tokens), dim=1)
        else:
            logits[is_stopped] = -float(np.inf)
            logits[is_stopped, 0] = 0
            scores_sum = scores[:, None] + logits
            seq_lengths[~is_stopped] += 1
            scores_sum_avg = scores_sum / seq_lengths[:, None]
            scores_sum_avg, next_tokens = scores_sum_avg.view(-1).topk(beam_width, -1)
            src = torch.div(next_tokens, scores_sum.shape[1], rounding_mode="trunc")
            seq_lengths = seq_lengths[src]
            next_tokens = (next_tokens % scores_sum.shape[1]).unsqueeze(1)
            tokens = torch.cat((tokens[src], next_tokens), dim=1)
            generated = generated[src]
            scores = scores_sum_avg * seq_lengths
            is_stopped = is_stopped[src]
        next_emb = model.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
        generated = torch.cat((generated, next_emb), dim=1)
        is_stopped = is_stopped + (next_tokens.eq(eos[0]) | next_tokens.eq(eos[1])).squeeze()
        if is_stopped.all():
            break
    scores = scores / seq_lengths
    texts = [tokenizer.decode(t[: int(L)]) for t, L in zip(tokens.cpu().numpy(), seq_lengths)]
    order = scores.argsort(descending=True).cpu().numpy().tolist()
    return [(texts[i], float(scores[i].item())) for i in order]


@torch.no_grad()
def stage_decode(recs, ifcap_src, ckpt, dev, beam_width=20):
    sys.path.insert(0, ifcap_src)
    import clip
    from transformers import GPT2Tokenizer
    from ClipCap import ClipCaptionModel
    from utils import compose_discrete_prompts
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = ClipCaptionModel(argparse.Namespace(device=dev), 10, 10, 512, gpt_type="gpt2", k=5)
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True), strict=False)
    model.to(dev)
    encoder, preprocess = clip.load("ViT-B/32", device=dev)
    for r in recs:
        f = encoder.encode_image(preprocess(open_rgb(r["path"])).unsqueeze(0).to(dev)).float()
        f = f / f.norm(2, dim=-1, keepdim=True)
        rt = encoder.encode_text(clip.tokenize(r["kept"], truncate=True).to(dev)).float()
        cont = model.mapping_network(f, rt.unsqueeze(0)).view(-1, 10, model.gpt_hidden_size)
        disc = compose_discrete_prompts(tokenizer, r["entities"]).unsqueeze(0).to(dev)
        emb = torch.cat((cont, model.gpt.transformer.wte(disc)), dim=1)
        r["beam"] = [{"caption": t, "LM": s} for t, s in beam_search_dump(emb, tokenizer, model.gpt, beam_width)]
    del model, encoder
    release()


# ---------------------------------------------------------------- stage 4: alignment agents

@torch.no_grad()
def stage_alignment(recs, bigG, prep, tok_g, blip, proc, dev):
    for r in recs:
        img = open_rgb(r["path"])
        caps = [b["caption"] for b in r["beam"]]
        i_g = bigG.encode_image(prep(img).unsqueeze(0).to(dev)).float()
        i_g = i_g / i_g.norm(2, dim=-1, keepdim=True).clamp_min(1e-6)
        t_g = bigG.encode_text(tok_g(caps).to(dev)).float()
        t_g = t_g / t_g.norm(2, dim=-1, keepdim=True).clamp_min(1e-6)
        for b, s in zip(r["beam"], (i_g @ t_g.T).squeeze(0).cpu().tolist()):
            b["bigG"] = s
        for b, s in zip(r["beam"], blip_itm_scores(blip, proc, img, caps, dev)):
            b["BLIP"] = s


# ---------------------------------------------------------------- stage 5: System One verifier

@torch.no_grad()
def stage_vqa(recs, dev, bs=10):
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    half = dev.startswith("cuda")
    proc = Blip2Processor.from_pretrained(BLIP2)
    model = Blip2ForConditionalGeneration.from_pretrained(BLIP2, torch_dtype=torch.float16 if half else torch.float32).to(dev).eval()
    tok = proc.tokenizer
    yes_id = tok("yes", add_special_tokens=False).input_ids[0]
    no_id = tok("no", add_special_tokens=False).input_ids[0]
    dec_start = torch.tensor([[model.config.text_config.decoder_start_token_id]], device=dev)
    for r in recs:
        img = open_rgb(r["path"])
        caps = [b["caption"].strip().rstrip(".") for b in r["beam"]]
        scores = []
        for j in range(0, len(caps), bs):
            part = caps[j : j + bs]
            prompts = [f'Question: Does this figure show "{c}"? Please answer yes or no. Answer:' for c in part]
            inputs = proc(images=[img] * len(part), text=prompts, return_tensors="pt", padding=True).to(dev)
            if half:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            logits = model(**inputs, decoder_input_ids=dec_start.expand(len(part), 1)).logits[:, 0, :].float()
            p_yes = torch.softmax(torch.stack([logits[:, yes_id], logits[:, no_id]], dim=1), dim=1)[:, 0]
            scores.extend(p_yes.cpu().tolist())
        for b, s in zip(r["beam"], scores):
            b["VQA"] = round(s, 5)
    del model
    release()


# ---------------------------------------------------------------- stage 6: OWLv2 grounding

@torch.no_grad()
def stage_ground(recs, dev, vocab_path, max_queries=40):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    from detector_ground_scorer import candidate_nouns, load_nltk, load_vocab
    pos_tag, word_tokenize, lem = load_nltk()
    vocab = load_vocab(vocab_path)
    half = dev.startswith("cuda")
    proc = Owlv2Processor.from_pretrained(OWLV2)
    model = Owlv2ForObjectDetection.from_pretrained(OWLV2).to(dev).eval()
    if half:
        model = model.half()
    for r in recs:
        nouns = [candidate_nouns(b["caption"], pos_tag, word_tokenize, lem, vocab)[0] for b in r["beam"]]
        queries = sorted(set(n for ns in nouns for n in ns))[:max_queries]
        best = {}
        if queries:
            inputs = proc(text=[[f"a photo of a {q}" for q in queries]], images=open_rgb(r["path"]), return_tensors="pt").to(dev)
            if half:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            mx = torch.sigmoid(model(**inputs).logits[0].float()).max(dim=0).values
            best = {q: round(float(mx[i]), 4) for i, q in enumerate(queries)}
        g = [float(np.mean([best[n] for n in ns if n in best])) if any(n in best for n in ns) else np.nan for ns in nouns]
        g = np.array(g)
        if np.isnan(g).any():  # candidates without a scored noun get the beam mean
            g[np.isnan(g)] = np.nanmean(g) if not np.isnan(g).all() else 0.0
        for b, s, ns in zip(r["beam"], g.tolist(), nouns):
            b["ground"], b["nouns"] = s, ns
    del model
    release()


# ---------------------------------------------------------------- stage 7: beam-internal MBR

_punct = re.compile(r"[^a-z0-9 ]+")


def _tok(s):
    return _punct.sub(" ", s.lower()).split()


class MbrCider:
    """CIDEr-D between two candidates, with the paper's fixed n-gram document frequencies."""

    def __init__(self, path):
        d = json.load(gzip.open(path, "rt"))
        self.n, self.sigma = d["n"], d["sigma"]
        self.df = {tuple(k.split(" ")): float(v) for k, v in d["df"].items()}
        self.log_ref = math.log(float(d["n_docs"]))

    def vec(self, toks):
        vecs, norms = [defaultdict(float) for _ in range(self.n)], [0.0] * self.n
        for k in range(self.n):
            cnt = defaultdict(int)
            for i in range(len(toks) - k):
                cnt[tuple(toks[i : i + k + 1])] += 1
            for g, c in cnt.items():
                v = float(c) * (self.log_ref - math.log(max(1.0, self.df.get(g, 0.0))))
                vecs[k][g] = v
                norms[k] += v * v
        return vecs, [math.sqrt(x) for x in norms], len(toks)

    def pair(self, a, b):
        (ha, na, la), (hb, nb, lb) = a, b
        delta, tot = float(la - lb), 0.0
        for k in range(self.n):
            s = sum(min(v, hb[k].get(g, 0.0)) * hb[k].get(g, 0.0) for g, v in ha[k].items())
            if na[k] and nb[k]:
                s /= na[k] * nb[k]
            tot += s * math.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
        return tot / self.n * 10.0


def stage_mbr(recs):
    cider = MbrCider(os.path.join(ASSETS, "mbr_df_coco_karpathy_test.json.gz"))
    for r in recs:
        v = [cider.vec(_tok(b["caption"])) for b in r["beam"]]
        K = len(v)
        for j, b in enumerate(r["beam"]):
            b["MBR"] = float(np.mean([cider.pair(v[j], v[l]) for l in range(K) if l != j]))


# ---------------------------------------------------------------- adjudication

def z(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def adjudicate(rec, config):
    # LM, bigG and BLIP were stored as float32 in the paper's feature files
    cols = {a: np.array([b[a] for b in rec["beam"]], dtype=np.float32 if a in ("LM", "bigG", "BLIP") else np.float64).astype(np.float64)
            for a in AGENTS}
    total = sum(z(cols[a]) for a in CONFIGS[config])
    for b, s in zip(rec["beam"], total.tolist()):
        b["score"] = s
    best = int(np.argmax(total))
    return rec["beam"][best]["caption"], best


# ---------------------------------------------------------------- driver

def run(paths, dev="cuda:0", config="five", cache=None, ifcap_root=None, ckpt=None, quiet=False):
    """Caption a list of image paths. Returns one record per image with the selected caption,
    the retrieval context and the scored beam."""
    from detector_ground_scorer import load_nltk
    load_nltk()
    cache = cache or os.path.join(os.path.expanduser("~"), ".cache", "frama")
    ifcap_src, ckpt, vocab = fetch_ifcap(cache, ifcap_root, ckpt)
    recs = [{"path": p} for p in paths]
    log(f"{len(recs)} images on {dev}")
    log("1/6 ViT-bigG/14 retrieval of 9 training captions")
    bigG, prep, tok_g = stage_retrieve(recs, dev)
    log("2/6 BLIP-ITM keeps 5 captions, frequent nouns form the hard prompt")
    blip, proc = load_blip_itm(dev)
    stage_rerank(recs, blip, proc, dev)
    log("3/6 IFCap decoder, beam of 20")
    stage_decode(recs, ifcap_src, ckpt, dev)
    log("4/6 agents: LM, bigG, BLIP-ITM")
    stage_alignment(recs, bigG, prep, tok_g, blip, proc, dev)
    del bigG, blip
    release()
    log("5/6 agent: BLIP-2 FlanT5-XL System One verifier")
    stage_vqa(recs, dev)
    log("6/6 agents: OWLv2 grounding, beam-internal MBR")
    stage_ground(recs, dev, vocab)
    stage_mbr(recs)
    for r in recs:
        r["caption"], r["selected"] = adjudicate(r, config)
        r["config"] = config
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*", default=[], help="image files")
    ap.add_argument("--image_dir", help="caption every image in this folder")
    ap.add_argument("--out", default="frama_captions.json")
    ap.add_argument("--config", choices=sorted(CONFIGS), default="five",
                    help="five: bigG, BLIP, VQA, MBR, ground (default); six: adds the LM agent, best on COCO")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache_dir", help="where the IFCap source and checkpoint are stored, default ~/.cache/frama")
    ap.add_argument("--ifcap_root", help="use an existing IFCap checkout instead of downloading one")
    ap.add_argument("--ifcap_ckpt", help="use an existing coco-indomain-004.pt")
    ap.add_argument("--dump_beam", action="store_true", help="also write all 20 candidates with every agent score")
    args = ap.parse_args()
    paths = list(args.images)
    if args.image_dir:
        paths += sorted(p for p in glob.glob(os.path.join(args.image_dir, "*")) if p.lower().endswith(IMAGE_EXT))
    if not paths:
        ap.error("no images given")
    recs = run(paths, args.device, args.config, args.cache_dir, args.ifcap_root, args.ifcap_ckpt)
    out = []
    for r in recs:
        o = {"image": r["path"], "caption": r["caption"].strip(), "config": r["config"]}
        if args.dump_beam:
            o.update(retrieved=r["kept"], entities=r["entities"], selected=r["selected"], beam=r["beam"])
        out.append(o)
        print(f"{os.path.basename(r['path'])}: {o['caption']}")
    json.dump(out, open(args.out, "w"), indent=1)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
