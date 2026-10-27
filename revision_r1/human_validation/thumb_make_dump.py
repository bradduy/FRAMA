"""T4 of thumb_prereg.md: write the four THumB machine captions of every image as a beam dump, so the
unchanged FRAMA agent scorers can score them. Order follows thumb_cider.npz (images sorted by set_id),
so every agent output aligns with C5, C4, C1, H and P. The LM agent is not applicable: the captions were
not produced by our captioner, and its log-probability is set to 0 and never used.

  python thumb_make_dump.py thumb_cider.npz thumb_beams.json
"""
import json, sys
import numpy as np

d = np.load(sys.argv[1], allow_pickle=True)
recs = [{"image_name": str(im), "seg_id": str(s), "split": "thumb",
         "beam": [{"caption": str(c), "score": 0.0} for c in cs]}
        for im, s, cs in zip(d["images"], d["segs"], d["cands"])]
json.dump(recs, open(sys.argv[2], "w"))
print(f"wrote {len(recs)} images x {len(recs[0]['beam'])} captions to {sys.argv[2]}")
