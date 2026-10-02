"""Is the 17/96 beam drift caused by recomputing CLIP ViT-B/32 image features?
Decode the first 96 test images with (a) the authors' stored features and (b) features
recomputed on this GPU, text features batched exactly as in the original dump."""
import argparse, io, json, os, pickle, shutil, sys, zipfile
import torch
HOME = os.path.expanduser("~/ZeroShotImageCap")
sys.path.insert(0, f"{HOME}/frama_hf")
import frama_infer as F
IF = f"{HOME}/IFCap"
dest = f"{HOME}/frama_hf/test_captions_ViT-B32.pickle"
if not os.path.exists(dest):
    raw = io.BufferedReader(F._HttpRange(F.IFCAP_ANN_ZIP), buffer_size=16 << 20)
    with zipfile.ZipFile(raw) as z, z.open("annotations/coco/test_captions_ViT-B32.pickle") as fin, open(dest, "wb") as fout:
        shutil.copyfileobj(fin, fout)
stored = {a[0]: a[1] for a in pickle.load(open(dest, "rb"))}
A = lambda p: json.load(open(f"{IF}/annotations/{p}"))
bigG9, blip5, ent = A("retrieved_sentences/caption_coco_image_coco_bigG_9.json"), A("retrieved_sentences/caption_coco_image_coco_bigG_blip5.json"), A("retrieved_entity/caption_coco_image_coco_bigG_blip5.json")
ids = list(blip5)
names = list(bigG9)[:96]
assert ids[:96] == names, "blip5 order differs from retrieval order"
dump = {b["image_name"]: b for b in json.load(open(f"{HOME}/frama_hf/ref/coco_beam_dump.json"))}
sys.path.insert(0, f"{IF}/src")
import clip
from transformers import GPT2Tokenizer
from ClipCap import ClipCaptionModel
from utils import compose_discrete_prompts
dev = "cuda:0"
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = ClipCaptionModel(argparse.Namespace(device=dev), 10, 10, 512, gpt_type="gpt2", k=5)
model.load_state_dict(torch.load(f"{IF}/checkpoints/train_coco/coco-indomain-004.pt", map_location=dev, weights_only=True), strict=False)
model.to(dev)
enc, prep = clip.load("ViT-B/32", device=dev)
with torch.no_grad():
    flat = [c for k in ids[:103] for c in blip5[k][:5]]          # original batches of 256 over the whole list
    rt_all = torch.cat([enc.encode_text(clip.tokenize(flat[i:i + 256], truncate=True).to(dev)).cpu() for i in range(0, len(flat), 256)])
    res = {"stored_feats_beam_match": 0, "recomputed_feats_beam_match": 0, "recomputed_feats_batch5_text_match": 0, "min_cos_stored_vs_recomputed": 1.0}
    for n_i, nm in enumerate(names):
        det = [w for f, w in ent[nm] if f >= 3]
        disc = model.gpt.transformer.wte(compose_discrete_prompts(tok, det).unsqueeze(0).to(dev))
        f_rec = enc.encode_image(prep(F.open_rgb(f"{IF}/annotations/coco/val2014/{nm}")).unsqueeze(0).to(dev)).float()
        f_sto = stored[nm].float().unsqueeze(0).to(dev)
        res["min_cos_stored_vs_recomputed"] = min(res["min_cos_stored_vs_recomputed"], float(torch.nn.functional.cosine_similarity(f_rec, f_sto).item()))
        rt_orig = rt_all[n_i * 5:(n_i + 1) * 5].to(dev).float()
        rt_b5 = enc.encode_text(clip.tokenize(blip5[nm][:5], truncate=True).to(dev)).float()
        ref = [b["caption"] for b in dump[nm]["beam"]]
        for key, f, rt in (("stored_feats_beam_match", f_sto, rt_orig), ("recomputed_feats_beam_match", f_rec, rt_orig), ("recomputed_feats_batch5_text_match", f_rec, rt_b5)):
            f = f / f.norm(2, dim=-1, keepdim=True)
            cont = model.mapping_network(f, rt.unsqueeze(0)).view(-1, 10, model.gpt_hidden_size)
            beam = F.beam_search_dump(torch.cat((cont, disc), dim=1), tok, model.gpt, 20)
            res[key] += [t for t, _ in beam] == ref
print(json.dumps(res, indent=1))
json.dump(res, open(f"{HOME}/frama_hf/diag_decode.json", "w"), indent=1)
