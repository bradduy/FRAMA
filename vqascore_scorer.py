"""C2b pilot: VQAScore (Lin et al. 2024) for every beam candidate.

Scores P(yes | image, "Does this figure show '<caption>'?") with CLIP-FlanT5,
a pointwise compositional alignment scorer, via the t2v_metrics package.
Resumable: images already in --out_jsonl are skipped, one line per image is
flushed as soon as it is scored, so a pre-emption loses at most one image.

Usage (GCP2, vqa_venv):
  python vqascore_scorer.py --beam_dump <coco_beam_dump.json> \
      --image_dir ~/ZeroShotImageCap/IFCap/annotations/coco/val2014 \
      --out_jsonl vqascore_coco.jsonl --limit 1000 --device cuda:0
"""
import argparse, json, os, time
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam_dump", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--model", default="clip-flant5-xl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1,
                    help="score every k-th image (uniform subsample for a pilot)")
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()

    import t2v_metrics
    scorer = t2v_metrics.VQAScore(model=args.model, device=args.device)

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
    with open(args.out_jsonl, "a") as out:
        for i, rec in enumerate(beams):
            name = rec["image_name"]
            if name in done:
                continue
            caps = [c["caption"].strip() for c in rec["beam"]]
            path = os.path.join(args.image_dir, name)
            scores = []
            for j in range(0, len(caps), args.batch):
                chunk = caps[j : j + args.batch]
                with torch.no_grad():
                    s = scorer(images=[path], texts=chunk)  # (1, len(chunk))
                scores.extend([float(x) for x in s[0].detach().cpu().tolist()])
            out.write(json.dumps({"image_name": name, "split": rec.get("split"),
                                  "vqascore": [round(x, 5) for x in scores]}) + "\n")
            out.flush(); n_new += 1
            if n_new % 50 == 0:
                el = time.time() - t0
                print(f"  {n_new} new ({i+1}/{len(beams)}), {el/n_new:.2f} s/img", flush=True)
    print(f"done: {n_new} new images in {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
