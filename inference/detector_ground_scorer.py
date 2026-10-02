"""C2b pilot: detector-grounded evidence for every noun in a beam-20 dump.

For each image, collect the nouns mentioned by any of its 20 candidates, query a
frozen open-vocabulary detector (OWLv2) once with all of them, and store the
maximum detection score and the number of boxes above a threshold per noun.
Per-candidate grounding scores are computed later, offline, from this file.

Resumable: images already present in --out_jsonl are skipped, and each image is
flushed as soon as it is scored, so a pre-emption loses at most one image.

Usage (GCP2, rap_venv):
  python detector_ground_scorer.py --beam_dump <coco_beam_dump.json> \
      --image_dir ~/ZeroShotImageCap/IFCap/annotations/coco/val2014 \
      --out_jsonl ground_coco.jsonl --device cuda:0
"""
import argparse, json, os, re, sys, time
import torch
from PIL import Image

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10}
# nouns that name the picture or a region rather than an object
SKIP = {"photo", "picture", "image", "front", "top", "side", "group", "couple", "lot",
        "background", "middle", "view", "close", "day", "time", "kind", "bunch", "pair",
        "piece", "row", "area", "scene", "shot", "part", "way", "distance", "edge",
        "line", "corner", "bit", "set", "number", "variety", "type", "sort", "half",
        "night", "air", "ground", "outdoors", "inside", "outside", "foreground",
        "back", "end", "rest", "someone", "something", "thing", "place", "spot"}


def load_nltk():
    import nltk
    for res, path in (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                      ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
                      ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
                      ("wordnet", "corpora/wordnet")):
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass
    from nltk import pos_tag, word_tokenize
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    return pos_tag, word_tokenize, lem


def candidate_nouns(caption, pos_tag, word_tokenize, lem, vocab=frozenset()):
    """Nouns = tokens tagged NN/NNS by nltk, plus any token whose lemma is in the
    object vocabulary (the nltk tagger mislabels e.g. 'couch' as JJ)."""
    toks = word_tokenize(caption.lower())
    tags = pos_tag(toks)
    nouns, nums = [], []
    prev_num = None
    for w, t in tags:
        if w in NUM_WORDS or re.fullmatch(r"\d+", w):
            prev_num = NUM_WORDS.get(w, int(w) if w.isdigit() else None)
            continue
        n = lem.lemmatize(w) if w.isalpha() else w
        is_noun = (t in ("NN", "NNS") and w.isalpha() and len(w) > 2) or (n in vocab)
        if is_noun:
            if n in SKIP:
                prev_num = None
                continue
            nouns.append(n)
            if prev_num:
                nums.append([n, prev_num])
            prev_num = None
            continue
        if t not in ("JJ", "JJR", "JJS", "CC"):  # allow "two small dogs"
            prev_num = None
    return sorted(set(nouns)), nums


def load_vocab(path):
    if not path:
        return frozenset()
    obj = json.load(open(path))
    names = obj.keys() if isinstance(obj, dict) else obj
    out = set()
    for s in names:
        s = str(s).lower().strip()
        out.add(s)
        if " " in s:
            out.add(s.split()[-1])  # head noun of multi-word classes
    return frozenset(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam_dump", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_queries", type=int, default=40)
    ap.add_argument("--stride", type=int, default=1, help="score every k-th image")
    ap.add_argument("--entity_vocab", default="",
                    help="json list/dict of object class names used as a noun whitelist")
    args = ap.parse_args()

    pos_tag, word_tokenize, lem = load_nltk()
    vocab = load_vocab(args.entity_vocab)
    print(f"vocab whitelist: {len(vocab)} names", flush=True)
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    proc = Owlv2Processor.from_pretrained(args.model)
    model = Owlv2ForObjectDetection.from_pretrained(args.model).to(args.device).eval()
    if args.device.startswith("cuda"):
        model = model.half()

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
    print(f"{len(beams)} images, {len(done)} already scored", flush=True)
    t0 = time.time(); n_new = 0
    with open(args.out_jsonl, "a") as out:
        for i, rec in enumerate(beams):
            name = rec["image_name"]
            if name in done:
                continue
            cands = []
            allnouns = set()
            for c in rec["beam"]:
                nouns, nums = candidate_nouns(c["caption"], pos_tag, word_tokenize, lem, vocab)
                cands.append({"nouns": nouns, "nums": nums})
                allnouns.update(nouns)
            queries = sorted(allnouns)[: args.max_queries]
            noun_scores = {}
            if queries:
                img = Image.open(os.path.join(args.image_dir, name)).convert("RGB")
                texts = [[f"a photo of a {q}" for q in queries]]
                inputs = proc(text=texts, images=img, return_tensors="pt").to(args.device)
                if args.device.startswith("cuda"):
                    inputs["pixel_values"] = inputs["pixel_values"].half()
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits[0].float()  # (num_patches, num_queries)
                probs = torch.sigmoid(logits)
                mx = probs.max(dim=0).values
                cnt = (probs > args.threshold).sum(dim=0)
                for qi, q in enumerate(queries):
                    noun_scores[q] = [round(float(mx[qi]), 4), int(cnt[qi])]
            out.write(json.dumps({"image_name": name, "split": rec.get("split"),
                                  "nouns": noun_scores, "cands": cands}) + "\n")
            out.flush(); n_new += 1
            if n_new % 100 == 0:
                el = time.time() - t0
                print(f"  {n_new} new ({i+1}/{len(beams)}), {el/n_new:.2f} s/img", flush=True)
    print(f"done: {n_new} new images in {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
