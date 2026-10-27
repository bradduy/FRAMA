# Revision R1 (October 2026)

Code and outputs added for the revised manuscript. Two corrections apply to every number of the revision:

1. **MBR document frequencies.** The beam-consensus agent weighs n-grams with CIDEr-D document frequencies.
   Earlier releases computed them on the COCO Karpathy-test reference captions. R1 computes them on the
   captioner's training captions, 566,747 COCO and 145,000 Flickr30k captions (`mbr_train_idf.py`). The
   resulting scores are `../mbr_trainidf_<benchmark>.npy`, aligned with the beam dumps.
2. **Fixed z-sum + grounding rule.** The row "z-sum + grounding" uses the fixed rule of the stack,
   z(LM) + z(bigG) + z(BLIP) + z(ground), instead of a grounding variant and weight chosen on test references.

## Rebuild the R1 prediction files (CPU)

`../verify_r1.sh` runs the two commands below and checks every file against `reference_results/r1/predictions_r1.sha256`;
`../check_paper.py` prints the main tables from the released results without rebuilding anything.

```bash
./reproduce.sh all                                   # submitted-version files in preds/ and mbr_self_*.npy
IFCAP_ROOT=data python3 revision_r1/rebuild_preds_r1.py
```

This writes `preds_r1/<benchmark>/<rule>__<split>.json` and `preds_r1/manifest.json`. Rules that do not involve the
MBR agent are checked to be identical to `preds/`. The rules that add the BLIP-2 caption as a 21st candidate recompute
the MBR agent and need the COCO training corpus of the captioner, `annotations/coco/coco_with_entities.pickle` from the
IFCap release; link IFCap's `annotations/` into `data/` for them. On a clean checkout this reproduces 226 of the 236
released files byte for byte, the manifest included. The other ten, `coco_d4/`, come from `supervision/d4_preds.py`,
which needs the re-decoded beams of that control. The same files are released at
[huggingface.co/bradduy/FRAMA](https://huggingface.co/bradduy/FRAMA) under `predictions_r1/`, and their official
metrics are in `../reference_results/official_metrics_r1/`.

| Benchmark, CIDEr / SPICE | LM top-1 | z-sum of 3 | FRAMA, 6 agents | FRAMA, 5 agents without LM |
|---|---|---|---|---|
| COCO Karpathy test | 112.7 / 21.0 | 118.6 / 22.0 | 120.7 / 22.2 | 120.5 / 22.2 |
| NoCaps val overall | 70.3 / 9.8 | 79.9 / 10.9 | 81.2 / 11.0 | 82.2 / 11.0 |
| Flickr30k Karpathy test | 63.7 / 17.0 | 68.2 / 18.1 | 70.7 / 18.3 | 72.3 / 18.5 |
| COCO, ViECap captioner | 91.2 / 18.1 | 105.6 / 20.5 | 106.7 / 20.4 | 107.0 / 20.5 |

## Contents

| Path | Content |
|---|---|
| `mbr_train_idf.py`, `rebuild_preds_r1.py` | corrected MBR agent and the rebuild of every selection file |
| `official_matrices.py`, `five_ref_matrix.py`, `perimage_official.py`, `run_r1_*.sh` | official per-candidate and per-image CIDEr-D and SPICE |
| `analyze_r1.py`, `analyze_r1_sweep.py` | paired bootstrap with Holm correction, held-out-reference statistics, k-sensitivity, weight sweep and beam width |
| `chair_coverage.py` | per-image CHAIR and object coverage with the hash-pinned implementation |
| `human_validation/` | validation on THumB 1.0 human ratings: leakage-free oracle, CHAIR, and the FRAMA agents as selectors, with the pre-registration |
| `supervision/` | the control with no COCO-fine-tuned matcher in retrieval or adjudication |
| `cost/` | timing of every stage and configuration on one GPU |
| `factual_reliability/` | factual errors beyond objects: SPICE precision by tuple category, SugarCrepe and VALSE foils inside the beam, and atomic-fact verification with Qwen3-VL-4B validated on THumB; pre-registration in `e6_prereg.md`, outcomes in `e6_results.md` |

The GPU launchers were run on our machine and assume an IFCap checkout at `~/ZeroShotImageCap/IFCap`, set in
variables at the top of each script. THumB 1.0 data: https://github.com/jungokasai/THumB. SugarCrepe and VALSE
sources are listed in `factual_reliability/data/SOURCE.txt`.
