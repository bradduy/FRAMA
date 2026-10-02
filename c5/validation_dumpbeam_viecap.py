"""C5 (captioner-agnostic transfer): beam-20 dump of the released ViECap COCO
captioner (Fei et al., ICCV 2023) in the IFCap dump format
({image_name, captions, prediction, beam:[{caption, score}]}).

Run from the ViECap repo root under rap_venv (same code lineage as IFCap):
  python validation_dumpbeam_viecap.py --weight_path checkpoints/train_coco/coco_prefix-0014.pt \
      --annotations_root ../IFCap/annotations --image_list <txt> --out_json <path> --device cuda:0
Entity prompts follow the README COCO command: coco_entities, prompt ensemble,
top_k 3, threshold 0.4, soft prompt first. Pre-extracted ViT-B/32 test features
(test_captions_ViT-B32.pickle) are used, as in eval_coco.sh.
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer

# the script lives outside the ViECap repo: import its modules from the cwd or the default clone path
for _p in (os.getcwd(), os.environ.get("VIECAP_ROOT", os.path.expanduser("~/ZeroShotImageCap/ViECap"))):
    if os.path.exists(os.path.join(_p, "ClipCap.py")) and _p not in sys.path:
        sys.path.insert(0, _p)
from ClipCap import ClipCaptionModel
from utils import compose_discrete_prompts
from load_annotations import load_entities_text
from retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories


@torch.no_grad()
def beam_search_dump(embeddings, tokenizer, model, beam_width=20, max_len=64, end_of_sentences=(".", " .")):
    """search.beam_search returning the full ranked list with length-normalised log-prob scores."""
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
            tokens = next_tokens
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
        wte = model.transformer.wte if hasattr(model, "transformer") else model.model.embed_tokens
        next_emb = wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
        generated = torch.cat((generated, next_emb), dim=1)
        is_stopped = is_stopped + (next_tokens.eq(eos[0]) | next_tokens.eq(eos[1])).squeeze()
        if is_stopped.all():
            break
    scores = scores / seq_lengths
    out_list = tokens.cpu().numpy()
    texts = [tokenizer.decode(o[: int(L)]) for o, L in zip(out_list, seq_lengths)]
    order = scores.argsort(descending=True).cpu().numpy().tolist()
    return [(texts[i], float(scores[i].item())) for i in order]


@torch.no_grad()
def main(args):
    device = args.device
    clip_name = args.clip_model.replace("/", "")
    A = args.annotations_root
    entities_text = load_entities_text("coco_entities", f"{A}/vocabulary/coco_categories.json", True)
    texts_embeddings = clip_texts_embeddings(entities_text, f"{A}/vocabulary/coco_embeddings_{clip_name}_with_ensemble.pickle")
    tokenizer = GPT2Tokenizer.from_pretrained(args.language_model)
    model = ClipCaptionModel(args.continuous_prompt_length, args.clip_project_length, 512, gpt_type=args.language_model)
    sd = torch.load(args.weight_path, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(device).eval()
    with open(f"{A}/coco/test_captions_{clip_name}.pickle", "rb") as f:
        annotations = pickle.load(f)  # [[image_name, features, [captions]], ...]
    keep = None
    if args.image_list:
        keep = set(l.strip() for l in open(args.image_list) if l.strip())
        annotations = [a for a in annotations if a[0] in keep]
        print(f"image_list: {len(keep)} names, {len(annotations)} matched", flush=True)
    done = {}
    if os.path.exists(args.out_json) and args.resume:
        for r in json.load(open(args.out_json)):
            done[r["image_name"]] = r
        print(f"resume: {len(done)} images already dumped", flush=True)
    predicts = []
    for image_id, image_features, captions in tqdm(annotations, dynamic_ncols=True):
        if image_id in done:
            predicts.append(done[image_id]); continue
        feats = image_features.float().unsqueeze(0).to(device)
        feats = feats / feats.norm(2, dim=-1, keepdim=True)
        cont = model.mapping_network(feats).view(-1, args.continuous_prompt_length, model.gpt_hidden_size)
        logits = image_text_simiarlity(texts_embeddings, temperature=args.temperature, images_features=feats)
        det, _ = top_k_categories(entities_text, logits, args.top_k, args.threshold)
        det = det[0]
        disc = compose_discrete_prompts(tokenizer, det).unsqueeze(0).to(device)
        disc_emb = model.word_embed(disc)
        emb = torch.cat((cont, disc_emb), dim=1) if args.soft_prompt_first else torch.cat((disc_emb, cont), dim=1)
        beam = beam_search_dump(emb, tokenizer, model.gpt, beam_width=args.beam_width)
        predicts.append({"image_name": image_id, "captions": captions, "prediction": beam[0][0],
                         "beam": [{"caption": t, "score": s} for t, s in beam], "detected_entities": det})
        if len(predicts) % 100 == 0:
            json.dump(predicts, open(args.out_json + ".partial", "w"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(predicts, open(args.out_json, "w"), indent=1)
    print(f"wrote {args.out_json} ({len(predicts)} images)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--clip_model", default="ViT-B/32")
    p.add_argument("--language_model", default="gpt2")
    p.add_argument("--continuous_prompt_length", type=int, default=10)
    p.add_argument("--clip_project_length", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--soft_prompt_first", action="store_true", default=True)
    p.add_argument("--beam_width", type=int, default=20)
    p.add_argument("--weight_path", default="checkpoints/train_coco/coco_prefix-0014.pt")
    p.add_argument("--annotations_root", default="../IFCap/annotations")
    p.add_argument("--image_list", default="")
    p.add_argument("--out_json", required=True)
    p.add_argument("--resume", action="store_true", default=True)
    main(p.parse_args())
