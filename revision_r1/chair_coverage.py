"""Per-image official CHAIR counts plus object coverage, for the R1 revision.

Uses the hash-pinned adapter objects of aic_repro/chair exactly as
aic_repro/scripts/chair_per_image.py does (same verification, same algorithm, same
ground-truth union), and adds two per-image fields that the adapter computes but does
not report: the set of ground-truth COCO object nodes of the image and the set of
object nodes the caption mentions. Coverage = |mentioned & GT| / |GT| measures how many
annotated objects a caption names, which CHAIR alone does not. The per-image CHAIR
counts must sum to the adapter's aggregate counts or the script fails.

Run from ~/ZeroShotImageCap/aic_repro with the CHAIR venv:
  $A/venv/bin/python <this> --system NAME=preds.json ... --annotation-root $A/annotations \
     --chair-source-root ... --pattern-source-root ... --nltk-data-root $A/nltk_data \
     --manifest manifests/chair_official_sources.json --output out.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path.home() / "ZeroShotImageCap" / "aic_repro"
sys.path.insert(0, str(REPO_ROOT))

from eval.common import validate_ifcap_predictions  # noqa: E402
from chair.algorithm import OfficialChairAlgorithm, map_prediction_identifiers  # noqa: E402
from chair.pattern36 import load_singularize  # noqa: E402
from chair.sources import verify_annotations, verify_nltk_runtime, verify_sources  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", action="append", required=True)
    p.add_argument("--annotation-root", type=Path, required=True)
    p.add_argument("--chair-source-root", type=Path, required=True)
    p.add_argument("--pattern-source-root", type=Path, required=True)
    p.add_argument("--nltk-data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    systems = {}
    for item in args.system:
        name, path = item.split("=", 1)
        systems[name] = (Path(path), validate_ifcap_predictions(Path(path)))
    sources = verify_sources(manifest_path=args.manifest, repo_root=REPO_ROOT,
                             chair_root=args.chair_source_root, pattern_root=args.pattern_source_root)
    annotations = verify_annotations(args.manifest, args.annotation_root)
    nltk = verify_nltk_runtime(args.manifest, args.nltk_data_root)
    algorithm = OfficialChairAlgorithm(synonyms_path=sources.synonyms_path,
                                       word_tokenize=nltk.word_tokenize,
                                       singularize=load_singularize(sources.pattern_inflect_path))
    first = next(iter(systems.values()))[1]
    images = map_prediction_identifiers(first.rows)
    for name, (_, predictions) in systems.items():
        if map_prediction_identifiers(predictions.rows) != images:
            raise SystemExit("image mapping of %s differs from the first system" % name)
    ground_truth, mapping = algorithm.build_ground_truth(annotations.files, images)

    out = {"schema": "frama-r1.chair-coverage.v1", "mapping_sha256": mapping["mapping_sha256"],
           "gt_objects": {n: sorted(ground_truth[im.image_id]) for n, im in images.items()},
           "systems": {}}
    for name, (path, predictions) in systems.items():
        rows = {}
        for row in predictions.rows:
            image = images[row["image_name"]]
            _, node_words, _, _ = algorithm.caption_to_words(row["prediction"])
            gt = ground_truth[image.image_id]
            hallucinated = sum(1 for w in node_words if w not in gt)
            rows[row["image_name"]] = {"mentions": len(node_words), "hall": hallucinated,
                                       "hall_sent": int(hallucinated > 0),
                                       "nodes": sorted(set(node_words)),
                                       "covered": len(set(node_words) & gt), "gt": len(gt)}
        official = algorithm.evaluate(predictions.rows, images, ground_truth)
        sums = [sum(v[k] for v in rows.values()) for k in ("mentions", "hall", "hall_sent")]
        if sums != [official.object_mentions, official.hallucinated_object_mentions,
                    official.hallucinated_sentences] or len(rows) != official.sentences:
            raise SystemExit("per-image counts of %s do not sum to the adapter counts" % name)
        out["systems"][name] = {"predictions": str(path), "CHAIR_s": official.chair_s,
                                "CHAIR_i": official.chair_i, "per_image": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, sort_keys=True))
    print(json.dumps({n: [round(100 * s["CHAIR_s"], 2), round(100 * s["CHAIR_i"], 2)]
                      for n, s in out["systems"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
