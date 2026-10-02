# FRAMA: frozen multi-agent adjudication, supplementary code

Scripts that produce every number in the paper from frozen beam-20 dumps of the released IFCap
checkpoints (and of ViECap for the transfer study).

- Agents: `csfn_extract_features.py` of the IFCap fork computes the decoder likelihood, the ViT-bigG/14
  cosine and the BLIP-ITM probability; `blip2vqa_scorer.py` the System-1 verifier; `detector_ground_scorer.py`
  OWLv2 grounding; the beam-consensus agent is computed inside the analysis scripts.
- Selection rules and prediction files: `analyze_*.py`, `make_*_preds.py`, `c5/` for ViECap.
- Evaluation: `run_official_eval_remote.sh` (official COCO caption toolkit, Java PTB tokenizer),
  `paired_bootstrap_remote.py` (10,000 resamples of per-image CIDEr), `crossref_oracle_official.py`
  (leakage-free oracle, XRef@N), `per_image_cider_remote.py`.
- Control studies: `revision/`: BLIP-2 direct captioning (`blip2_caption.py`), BLIP-ITM checkpoint swap
  (`itm_swap_scorer.py`), CLIPScore and PAC-S++ (`clipscore_eval.py`, `pacs_eval.py`), leakage-free oracle
  sensitivity and human-agreement baseline (`crossref_sensitivity.py`, `xref_human_ci.py`), System One versus
  System Two judging (`cot_judge_scorer.py`, `analyze_cot.py`), BLIP-2 caption as an extra candidate
  (`run_m4*.sh`, `analyze_m4.py`), within-beam agent correlations (`pairwise_corr.py`).

Python scripts read the IFCap checkout from `IFCAP_ROOT`; the shell launchers assume the layout `~/ZeroShotImageCap/IFCap` and `~/ZeroShotImageCap/macst_v1/v2_upgrade` (this folder), set in the variables at the top of each script. Beam dumps and prediction files are released with the paper.
