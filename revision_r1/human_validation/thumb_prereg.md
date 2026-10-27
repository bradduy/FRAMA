# Pre-registration: validation against existing human judgements (THumB 1.0)

Written 2026-10-25, before any THumB number was computed. It replaces new annotation, because no annotators are available. The study of `human_study_prereg.md` remains available if annotators are found later.

## Data

- THumB 1.0 (Kasai et al., NAACL 2022), from GitHub `jungokasai/THumB` at commit `fd099b4`.
- 500 COCO Karpathy-test images, all of them in our test set.
- Each image has four machine captions (Up-Down, Unified-VLP, VinVL-base, VinVL-large) and one human COCO caption.
- Every caption is rated by trained annotators: Precision 1–5, Recall 1–5, and a total score = (P+R)/2 minus penalties.
- References: the four THumB references plus the human caption, five in total, which matches the fold structure of the paper.
- Candidates: the four machine captions.
- All metrics are official: PTB tokenizer and pycocoevalcap CIDEr-D, `thumb_official.py`. CHAIR uses the hash-pinned adapter.

## Analyses and expectations

**T1. Does the reference-selected advantage appear in human judgement?**

Per image, define the following picks and baselines:
- `o`, the reference-selected pick: argmax CIDEr against all five references.
- `o_f`, the leakage-free pick for fold f: argmax against the four references other than r_f.
- `m`, the mean over the four candidates, used as the baseline.

Report the following quantities. The CIDEr quantities are averaged over the five folds; the paired bootstrap runs over images with 10,000 resamples, seed 0.
- (a) The CIDEr advantage on the held-out scale:
  - O = C1(o) − C1(m), where `o`'s selection saw r_f;
  - X = C1(o_f) − C1(m).
  - The surviving share is X / O.
- (b) The human-score advantage:
  - HO = H(o) − H(m);
  - HX = H(o_f) − H(m), averaged over folds.
  - The human share is HX / HO.
- (c) H(o) − H(o_f), with its interval.

Expectation (H-T1): most of the extra CIDEr that `o` gains over `o_f` comes from n-grams specific to the held-out reference. If so, the human share exceeds the CIDEr surviving share, and H(o) − H(o_f) is small, with an interval that includes zero or lies within 0.05 score points. If instead H(o) − H(o_f) is clearly positive and of a size comparable to the CIDEr gap, the inflation claim weakens. We will report that outcome as it is.

**T2. Metric agreement with humans at caption level.**

Compute Kendall τ-b between the human score and each of:
- CIDEr against five references;
- CIDEr against four references, averaged over folds;
- CIDEr against one held-out reference, averaged over folds.

These numbers are descriptive and carry no expectation. They show how much a single reference loses at caption level, which bears on how noisy the leakage-free scale is.

**T3. CHAIR against human precision.**

- Per caption, compare the CHAIR flag (at least one hallucinated COCO object) with human precision P.
- Report the mean P of flagged and unflagged captions.
- Report the precision and recall of the CHAIR flag for the event P ≤ 3, meaning the annotators judged the caption clearly inaccurate.
- Repeat the same with P < 5.

Expectation (H-T3): CHAIR flags are mostly correct, so flagged captions have clearly lower P. Recall is low, because many inaccurate captions contain errors outside the 80 object categories. This supports the CHAIR result and the stated limitation together.

**T4. FRAMA agents against human judgement.** Optional; needs a GPU.

- Score the four machine captions of each image with the agents that need no beam: dual encoder, cross-encoder, System One verifier, grounding, and consensus among the four.
- Report the human score of the caption each selector picks: each agent alone, their equal-weight z-sum, CIDEr-5ref (which is the oracle), random choice, and each fixed system.
- Report pairwise agreement with human ordering.

Expectation: the z-sum pick has a higher human score than random and than any single CLIP-family agent.

## Exclusions

None. All 500 images are used.

## Deviations

Record any deviation from this plan here, with its date.
- 2026-10-25: T1–T3 run as planned. T4 not run: both GPUs on GCP2 are occupied by other jobs, and the dual encoder and the System One model do not fit in the free memory. It remains optional.
- 2026-10-26, before any T4 score was computed: T4 is run (`run_thumb_t4.sh`, analysis `thumb_t4.py`). The GPUs are still shared, so the dual encoder and the cross-encoder run in fp32 on the CPU, the precision of the paper, and the System One verifier runs in fp16 sharded over both cards. Scorer code and prompts are unchanged; captions are passed as THumB gives them. Grounding uses the paper's per-caption mean detection score. Added to the planned report, all descriptive: the two- and three-agent sums, the leakage-free CIDEr pick, and the mean human Precision and Recall of every pick. The planned test is unchanged: the five-agent equal-weight sum against random choice and against each of bigG and BLIP-ITM alone, paired bootstrap over images.

## Results

- 2026-10-26, T4 (`thumb_t4.py`, `thumb_t4_results.json`). The five-agent equal-weight pick has human score 4.267, which is +0.167 [+0.137, +0.198] above random choice: supported. Against bigG alone the difference is -0.005 [-0.037, +0.026], and against BLIP-ITM alone it is -0.026 [-0.057, +0.005]: not supported. The expectation is therefore met only in part.
- Descriptive results:
  - Against BLIP-ITM, the five-agent pick has human Precision +0.056 [+0.008, +0.106] and Recall -0.112 [-0.158, -0.068]. Against the bigG + BLIP-ITM sum, Precision is +0.080 [+0.034, +0.128] and Recall -0.110 [-0.156, -0.066].
  - The five-agent pick (4.267) is level with the CIDEr pick on five references (4.258; difference +0.009 [-0.027, +0.045]).
  - Pairwise agreement with the human ordering is 0.759 for the five agents and 0.711 for CIDEr on five references.
  - The detector alone is at the level of random choice (4.097).
