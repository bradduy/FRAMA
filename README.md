# FRAMA: frozen multi-agent adjudication for zero-shot image captioning

Code and released artifacts for **"A Reranking Reality Check for Zero-Shot Image Captioning:
Leakage-Free Oracles and System One Adjudication"**.

FRAMA leaves the captioner frozen and only chooses which of its beam-20 candidates to return.
Six frozen agents score every candidate: LM, the decoder log-likelihood; bigG, the ViT-bigG/14
CLIP cosine; BLIP, the BLIP-ITM match probability; VQA, the yes-probability of the BLIP-2 FlanT5-XL
System One verifier; MBR, a beam-internal consensus; and ground, the mean OWLv2 detection score of
the caption's nouns. Each score is z-normalised within the beam and the six are summed with
equal weights. Nothing is trained or tuned on any evaluation image.

## Revision R1 (October 2026)

The revised manuscript corrects the beam-consensus (MBR) agent. Earlier releases computed its CIDEr-D
document frequencies on the COCO Karpathy-test reference captions, which lets test statistics into the
selection; R1 computes them on the captioner's training captions. The headline numbers become 120.7
CIDEr on COCO with six agents, 82.2 on NoCaps and 72.3 on Flickr30k with five agents. The corrected
scores are `mbr_trainidf_<benchmark>.npy`, the code and the analyses added in the revision are in
[`revision_r1/`](revision_r1/README.md), official metrics in `reference_results/official_metrics_r1/`,
and the prediction files in `predictions_r1/` on [Hugging Face](https://huggingface.co/bradduy/FRAMA).
`inference/frama_infer.py` uses the corrected frequencies. The levels below reproduce the submitted
version byte for byte; `revision_r1/rebuild_preds_r1.py` then rebuilds the R1 files from the same scores.

## Caption your own images

`inference/frama_infer.py` runs the whole pipeline on any image: retrieval, the frozen IFCap
decoder, the six agents and the equal-weight selection. Its assets and model card are on the
Hugging Face Hub at [bradduy/FRAMA](https://huggingface.co/bradduy/FRAMA):

```bash
hf download bradduy/FRAMA --local-dir FRAMA && cd FRAMA
pip install -r requirements.txt
python frama_infer.py --images photo.jpg --out captions.json
```

See `inference/README.md` for its verification against the paper's artifacts.

## What can be reproduced, and how

| Level | Needs | Command |
|---|---|---|
| 1. Every prediction file behind the tables, from the released agent scores | CPU, numpy, scipy | `./reproduce.sh` or `./reproduce.sh all` |
| 2. Official metrics of those files | Java 11, the IFCap evaluation toolkit | `./setup_eval.sh`, `./evaluate_official.sh`, `python3 compare_results.py` |
| 3. Recompute the agent scores from the beam dumps | GPU, COCO / NoCaps / Flickr30k images | scorer scripts below |

Level 1 is exact: on a clean checkout, `./reproduce.sh all` regenerates all 274 prediction files
we scored for the paper byte for byte. Level 2 compares your official scores against `reference_results/official_metrics/`,
the files every table of the paper was filled from.

### Level 1: prediction files (CPU)

```bash
pip install -r requirements.txt
./reproduce.sh        # main tables, about 10 minutes on one CPU core
./reproduce.sh all    # adds verifier ladder, small judges, wider beams, revision controls; about 50 minutes
```

Selected captions are written to `preds/<benchmark>/<rule>__<split>.json` in the IFCap evaluation
format: each record holds `image_name`, the reference `captions` and the selected `prediction`.
The main rules are

| Rule file | Meaning |
|---|---|
| `lm_top1` | captioner's own top-1 |
| `zsum3_lm_bigG_blip` | equal z-sum of LM, bigG and BLIP |
| `stack6_all_equal` | FRAMA, six agents |
| `stack5_noLM_blip_bigG_vqa_mbr_ground` | FRAMA without the LM agent |
| `oracle20_approx` | Oracle@20 selected with the approximate CIDEr of `pilot_headroom.py` |
| `loo_minus_<agent>` | leave-one-out ablation |
| `rel6_spectral`, `rel6_mallows` | label-free reliability weighting |

### Level 2: official metrics

```bash
./setup_eval.sh                       # downloads evaluation.zip of the IFCap v1.0 release into third_party/
./evaluate_official.sh coco nocaps flickr coco_viecap_full
python3 compare_results.py
```

`evaluate_official.sh` scores each prediction file with the IFCap copy of the COCO caption
toolkit and skips files already scored. SPICE needs Java 11. Expected values of the submitted version,
with the test-reference MBR frequencies, from `reference_results/official_metrics/`; the R1 values are in
`revision_r1/README.md`:

| Benchmark | LM top-1 | z-sum of 3 | FRAMA, 6 agents | FRAMA, 5 agents without LM |
|---|---|---|---|---|
| COCO Karpathy test, CIDEr / SPICE | 112.7 / 21.0 | 118.6 / 22.0 | 120.8 / 22.2 | 120.5 / 22.2 |
| NoCaps val overall | 70.3 / 9.8 | 79.9 / 10.9 | 81.3 / 11.0 | 82.3 / 11.0 |
| Flickr30k Karpathy test | 63.7 / 17.0 | 68.2 / 18.1 | 70.5 / 18.2 | 72.1 / 18.5 |
| COCO, ViECap captioner | 91.2 / 18.1 | | 106.7 / 20.4 | 106.8 / 20.5 |

Paired bootstrap and the leakage-free cross-reference oracle use the same toolkit:

```bash
python3 paired_bootstrap_remote.py --base preds/coco/zsum3_lm_bigG_blip__overall.json \
    --cand preds/coco/stack6_all_equal__overall.json --out bootstrap_coco.json
python3 crossref_oracle_official.py --bench coco \
    --beam_dump data/inference_result/coco-indomain/coco-indomain-004_biggFull/coco_beam_dump.json \
    --preds_dir preds/coco --out crossref_coco.json
```

Our outputs are in `reference_results/bootstrap/` and `reference_results/crossref/`.

### Level 3: agent scores (GPU)

```bash
pip install -r requirements-gpu.txt
# LM, bigG and BLIP-ITM, written to one features.npz
python csfn_extract_features.py --beam_dump <dump.json> --image_folder <images> --out_npz features.npz
# System One verifier
python blip2vqa_scorer.py --beam_dump <dump.json> --image_dir <images> --out_jsonl vqascore_coco.jsonl
# open-vocabulary grounding
python detector_ground_scorer.py --beam_dump <dump.json> --image_dir <images> --out_jsonl ground_coco.jsonl \
    --entity_vocab <IFCap>/annotations/vocabulary/vgcocooiobjects_v1_class2ind.json
# beam-internal MBR needs no model: pilot_mbr_consensus.py (submitted version), revision_r1/mbr_train_idf.py (R1)
```

`frozen_scorer.py` for SigLIP2, CLIP-L/14 and BLIP-2 ITM, `smallvlm_yesprob_scorer.py` for SmolVLM
and Qwen2-VL, and the scorers in `revision/` follow the same interface. The shell launchers
`run_*.sh`, `chain_*.sh`, `c5/*.sh` and `revision/run_*.sh` are the ones we ran on our GPU machine;
they assume an IFCap checkout at `~/ZeroShotImageCap/IFCap` with this repository next to it, set in
variables at the top of each script.

The beam-20 dumps themselves come from the released IFCap checkpoints `coco-indomain-004.pt` and
`flickr-indomain-0014.pt`, decoded with beam width 20 after the retrieval stage of the baseline
pipeline: ViT-bigG/14 retrieves nine captions and BLIP-ITM keeps five. The
ViECap dump is produced by `c5/validation_dumpbeam_viecap.py` from the released ViECap COCO
checkpoint. We release the dumps rather than the retrieval code, so every number above starts
from the same candidates we scored.

## Repository layout

| Path | Content |
|---|---|
| `data/inference_result/` | beam-20 dumps and LM, bigG and BLIP features for COCO, NoCaps and Flickr30k; beam-40 and beam-80 dumps in `c4_expand/`, full sets gzipped, plus their 1,000-image subsamples |
| `ground_*.jsonl`, `vqascore_*.jsonl` | grounding and System One verifier scores per candidate |
| `scores_*.jsonl`, `smalljudge_*.jsonl` | verifier ladder and small-judge scores |
| `c5/` | ViECap transfer: beam dumps, features, agent scores, beam-dump script |
| `revision/` | control studies: BLIP-2 captions in `blip2cap/`, the BLIP-2 caption as a 21st candidate in `m4_*`, the System Two judge in `cot_shard*`, the Flickr-tuned BLIP-ITM checkpoint in `itm_flickr_coco.jsonl` |
| `reference_results/` | our official metrics, CHAIR, bootstrap and cross-reference outputs |
| `pilot_*.py`, `analyze_*.py`, `make_*_preds.py` | selection rules and analyses, all CPU |
| `revision_r1/`, `mbr_trainidf_*.npy` | R1 revision: corrected MBR agent, rebuild of every selection file, human validation, supervision control, cost, factual reliability |
| `reference_results/official_metrics_r1/` | official metrics of the R1 prediction files |
| `inference/` | single-image inference, mirrored from the Hugging Face repository, with its verification reports |

Environment variables: `IFCAP_ROOT` is the folder holding `inference_result/`, `data/` by default in
`reproduce.sh`; `COCOEVAL_DIR` is the IFCap evaluation toolkit, `third_party/evaluation` by default;
`PYTHON` is the interpreter.

## License

Code under the MIT license. The beam dumps contain reference captions of COCO, NoCaps and
Flickr30k, which remain under the terms of those datasets.
