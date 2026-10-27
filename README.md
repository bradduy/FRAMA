# FRAMA: frozen multi-agent adjudication for zero-shot image captioning

Code, data and results for **"A Reranking Reality Check for Zero-Shot Image Captioning: Leakage-Free Oracles
and System One Adjudication"**, revised manuscript (R1, October 2026).

FRAMA leaves a zero-shot captioner frozen and only chooses which of its 20 beam candidates to return. Six frozen
agents score every candidate, each score is z-normalised within the beam, and the six are summed with equal
weights. No parameter is trained or tuned on any evaluation image.

| Agent | Model | Signal |
|---|---|---|
| LM | IFCap decoder | length-normalised log-likelihood |
| bigG | OpenCLIP ViT-bigG/14, LAION-2B | image-caption cosine |
| BLIP | `Salesforce/blip-itm-base-coco` | image-text match probability |
| VQA, the System One verifier | `Salesforce/blip2-flan-t5-xl` | P(yes) to "Does this figure show ...?" |
| MBR, the consensus agent | none | mean CIDEr-D of the candidate against the other 19 |
| ground | `google/owlv2-base-patch16-ensemble` | mean detection score of the caption's nouns |

Headline numbers, official COCO caption toolkit, CIDEr / SPICE:

| Benchmark | Captioner top-1 | z-sum of LM, bigG, BLIP | FRAMA, 6 agents | FRAMA, 5 agents without LM |
|---|---|---|---|---|
| COCO Karpathy test | 112.7 / 21.0 | 118.6 / 22.0 | **120.7** / 22.2 | 120.5 / 22.2 |
| NoCaps val overall | 70.3 / 9.8 | 79.9 / 10.9 | 81.2 / 11.0 | **82.2** / 11.0 |
| Flickr30k Karpathy test | 63.7 / 17.0 | 68.2 / 18.1 | 70.7 / 18.3 | **72.3** / 18.5 |
| COCO, ViECap captioner | 91.2 / 18.1 | 105.6 / 20.5 | 106.7 / 20.4 | **107.0** / 20.5 |

## For reviewers

Three checks, from seconds to hours. None needs a GPU.

**1. Every main table from the released results, in seconds.**

```bash
pip install numpy
python3 check_paper.py          # Tables 2, 3, 12, 15, 16 and 17
python3 check_paper.py 2        # one table
```

It reads the official evaluation outputs in `reference_results/`, prints each table, and marks every value ✓ if it
equals the number printed in the paper. The last line reads `All values match the paper.`

**2. Every prediction file of the paper, rebuilt from the released agent scores, in about an hour.**

```bash
pip install -r requirements.txt
IFCAP_ANNOTATIONS=<IFCap checkout>/annotations ./verify_r1.sh
```

It runs every selection rule on the released beams and scores and compares each of the 236 prediction files with
the SHA-256 of the file the paper was scored from (`reference_results/r1/predictions_r1.sha256`). The ten files of
the supervision control in `coco_d4/` come from re-decoded beams; they are released on Hugging Face under
`predictions_r1/coco_d4/` and not rebuilt. Without
`IFCAP_ANNOTATIONS` the eight `m4_*` rules, which need the COCO training captions, are reported as not rebuilt.

**3. Official metrics of those files.** Needs Java 11.

```bash
./setup_eval.sh                                   # IFCap copy of the COCO caption toolkit, into third_party/
PREDS_DIR=preds_r1 RESULTS_DIR=results_r1 ./evaluate_official.sh coco nocaps flickr coco_viecap_full
python3 compare_results.py results_r1 official_metrics_r1
```

`compare_results.py` prints each metric next to its difference from `reference_results/official_metrics_r1/`.
Differences of a few hundredths can come from a different Java build; the prediction files are deterministic.

### Where each number of the paper comes from

| Paper | Released result | Produced by |
|---|---|---|
| Table 1, our rows | `reference_results/official_metrics_r1/`, `official_metrics/` | `evaluate_official.sh` |
| Table 2, main results | same, and `r1/perimage/` for NoCaps splits and Flickr30k | `revision_r1/rebuild_preds_r1.py`, `evaluate_official.sh` |
| Tables 3 and 4, leakage-free oracle | `r1/r1_numbers.json`, key `xref` | `revision_r1/official_matrices.py`, `revision_r1/analyze_r1.py` |
| Table 5, BLIP-2 caption as 21st candidate | `r1/perimage/*m4_*` | `revision/analyze_m4.py`, `revision_r1/rebuild_preds_r1.py` |
| Table 6, COCO supervision | `r1/d4_summary.json`, `r1/d4_bootstrap*.json`, `official_metrics_r1/coco_d4__*` | `revision_r1/supervision/` |
| Table 7, reference-free metrics | `r1/clipscore_r1_*.json`, `r1/pacs_r1_*.json` | `revision/clipscore_eval.py`, `revision/pacs_eval.py` |
| Table 8, verifier ladder | `r1/perimage/*ladder_*` | `analyze_ladder.py`, `make_ladder_preds.py` |
| Table 9, smaller System One models | `r1/perimage/*sj_*`, `*sub1000_*` | `analyze_smalljudge.py` |
| Table 10, System One and System Two | `r1/perimage/*cot500_*` | `revision/cot_judge_scorer.py`, `revision/analyze_cot.py` |
| Table 11, leave-one-out | `r1/perimage/*loo_minus_*` | `analyze_ablation.py` |
| Table 12 and Fig. 4C, CHAIR | `r1/r1_numbers.json`, key `chair`; per image `r1/chair_coverage_r1_coco.json` | `revision_r1/chair_coverage.py` |
| Tables 13 and 14, THumB human ratings | `r1/thumb_results.json`, `r1/thumb_t4_results.json` | `revision_r1/human_validation/` |
| Tables 15 and 16, factual errors beyond objects | `r1/e6_results.json` | `revision_r1/factual_reliability/` |
| Table 17, second captioner | `r1/perimage/viecap__*` | `analyze_c5_full.py`, `revision_r1/rebuild_preds_r1.py` |
| Table 18, cost | `r1/timing_summary.json` | `revision_r1/cost/time_configs.py` |
| Table 19, paired bootstrap | `r1/r1_numbers.json`, keys `family` and `extra` | `revision_r1/analyze_r1.py` |
| Weight sweep and label-free weights, Section VI-A | `r1/r1_sweep.json`, `r1/r1_reliability3.json` | `revision_r1/analyze_r1_sweep.py` |
| Table 20, implementation details | this repository and the scorer scripts below | |

Paths in the middle column are under `reference_results/`. `r1/perimage/<benchmark>__<rule>.json` holds the
official per-image CIDEr-D and SPICE of one prediction file; their means are the table values.

### Correction in the revision

Releases before October 2026 computed the n-gram document frequencies of the consensus (MBR) agent on the COCO
Karpathy-test reference captions, which lets test statistics into the selection. The revision computes them on
the captioner's training captions, 566,747 COCO and 145,000 Flickr30k captions (`revision_r1/mbr_train_idf.py`).
Every number above and in the revised paper uses the corrected agent; the corrected scores are
`mbr_trainidf_<benchmark>.npy`. With the earlier frequencies the six-agent stack scored 120.8 / 81.3 / 70.5 CIDEr
and the five-agent stack 120.5 / 82.3 / 72.1 on COCO / NoCaps / Flickr30k.

## Recompute the agent scores (GPU)

The released score files can be recomputed from the beam dumps and the benchmark images.

```bash
pip install -r requirements-gpu.txt
# LM, bigG and BLIP-ITM, written to one features.npz
python csfn_extract_features.py --beam_dump <dump.json> --image_folder <images> --out_npz features.npz
# System One verifier
python blip2vqa_scorer.py --beam_dump <dump.json> --image_dir <images> --out_jsonl vqascore_coco.jsonl
# open-vocabulary grounding
python detector_ground_scorer.py --beam_dump <dump.json> --image_dir <images> --out_jsonl ground_coco.jsonl \
    --entity_vocab <IFCap>/annotations/vocabulary/vgcocooiobjects_v1_class2ind.json
# consensus agent, CPU only; needs IFCap's annotations/ (training captions) linked into data/
IFCAP_ROOT=data python revision_r1/mbr_train_idf.py
```

The beam-20 dumps come from the released IFCap checkpoints `coco-indomain-004.pt` and `flickr-indomain-0014.pt`
after the retrieval stage of the baseline pipeline: ViT-bigG/14 retrieves nine captions and BLIP-ITM keeps five. The
ViECap dump comes from `c5/validation_dumpbeam_viecap.py` and the released ViECap COCO checkpoint. Every model
identifier, precision and prompt is listed in Table 20 of the paper. The shell launchers `run_*.sh`, `c5/*.sh`,
`revision/run_*.sh` and those in `revision_r1/` are the ones we ran; they assume an IFCap checkout at
`~/ZeroShotImageCap/IFCap`, set in variables at the top of each script.

## Caption your own images

`inference/frama_infer.py` runs the whole pipeline on any image: retrieval, the frozen IFCap decoder, the six
agents and the equal-weight selection. Its assets and model card are at
[huggingface.co/bradduy/FRAMA](https://huggingface.co/bradduy/FRAMA):

```bash
hf download bradduy/FRAMA --local-dir FRAMA_hf && cd FRAMA_hf
pip install -r requirements.txt
python frama_infer.py --images photo.jpg --out captions.json
```

`inference/verify_infer.py` checks it stage by stage against the files behind the paper.

## Repository layout

| Path | Content |
|---|---|
| `check_paper.py`, `verify_r1.sh` | the reviewer checks above |
| `data/inference_result/` | beam-20 dumps and LM, bigG and BLIP features for COCO, NoCaps and Flickr30k; beam-40 and beam-80 dumps in `c4_expand/` |
| `vqascore_*.jsonl`, `ground_*.jsonl` | System One verifier and grounding scores per candidate |
| `mbr_trainidf_*.npy` | consensus-agent scores with training-caption document frequencies |
| `scores_*.jsonl`, `smalljudge_*.jsonl` | verifier ladder and smaller System One models |
| `c5/` | second captioner, ViECap: beam dumps, features, agent scores |
| `revision/` | controls of the first submission: BLIP-2 captions, BLIP-2 caption as a 21st candidate, System Two judge, Flickr30k-tuned BLIP-ITM |
| `revision_r1/` | code added in the revision: corrected consensus agent, rebuild of every prediction file, official matrices and bootstrap, THumB human validation, supervision control, cost, factual reliability; see its README |
| `reference_results/r1/`, `reference_results/official_metrics_r1/` | results of the revised paper |
| `reference_results/official_metrics/`, `bootstrap/`, `crossref/`, `chair/` | results of the first submission, with the earlier MBR frequencies |
| `pilot_*.py`, `analyze_*.py`, `make_*_preds.py` | selection rules and analyses, CPU only |
| `inference/` | single-image inference, mirrored from Hugging Face |

Environment variables: `IFCAP_ROOT` is the folder holding `inference_result/`, `data/` by default; `COCOEVAL_DIR`
is the IFCap evaluation toolkit, `third_party/evaluation` by default; `PYTHON` is the interpreter.

## First-submission files

`./reproduce.sh` regenerates the 274 prediction files of the first submission byte for byte into `preds/`, and
`reference_results/official_metrics/` holds their official scores. They use the earlier MBR frequencies and are kept
so that the correction can be traced; `verify_r1.sh` runs this step first, because the revision reuses the rules
that do not involve the consensus agent.

## License

Code under the MIT license. The beam dumps contain reference captions of COCO, NoCaps and Flickr30k, which remain
under the terms of those datasets. SugarCrepe, VALSE and THumB are used under their own licenses.
