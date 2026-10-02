# Inference on your own images

This folder mirrors the code of the Hugging Face model repository
[bradduy/FRAMA](https://huggingface.co/bradduy/FRAMA), which also hosts the 1.5 GB of retrieval
assets. The model card there describes the pipeline, the agents and the output format.

```bash
cd inference
hf download bradduy/FRAMA --include "assets/*" --local-dir .
pip install -r requirements.txt
python frama_infer.py --images photo.jpg --out captions.json
```

Or take everything from the Hub: `hf download bradduy/FRAMA --local-dir FRAMA`.

A 16 GB GPU is enough; 96 COCO images take about 16 minutes on one Tesla T4, model loading
included. The first run downloads about 29 GB of models: BLIP-2 FlanT5-XL, ViT-bigG/14,
BLIP-ITM, OWLv2, CLIP ViT-B/32, GPT-2, and the IFCap COCO checkpoint from the IFCap authors'
GitHub release.

## Verification against the paper

`verify_infer.py` compares every stage with the artifacts of this repository on the first 96 COCO
Karpathy-test images; `verification/` holds the reports from one Tesla T4.

| Stage | Identical to the paper |
|---|---|
| bigG top-9 retrieval | 93 / 96; 4,977 / 5,000 on the full test set |
| five captions kept by BLIP-ITM, hard-prompt entities | 96 / 96 |
| beam of 20 candidates | 79 / 96 |
| agent scores on those beams | bigG, BLIP within 5e-6; VQA, MBR, ground exact; LM within 1.1e-3 |
| selected caption on those beams, six and five agents | 79 / 79 |

`verification/diag_decode.py` locates the 17 differing beams: decoding from the IFCap authors'
stored CLIP ViT-B/32 features gives 93 of 96 identical beams, and recomputed features agree with
the stored ones at cosine 0.999996, so the difference is floating-point noise that beam search
amplifies. The script assumes our GPU machine's layout, set at its top.
