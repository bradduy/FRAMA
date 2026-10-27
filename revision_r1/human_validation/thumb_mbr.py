"""T4: consensus agent over the four THumB captions of each image, with the same code and the same
training-corpus document frequencies as the paper's beam-consensus agent (mbr_train_idf.py): the score
of a caption is its mean CIDEr-D against the other three captions of the image.

  python thumb_mbr.py thumb_beams.json thumb_mbr.npy
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mbr_train_idf as M

vm_corpus = os.path.expanduser("~/ZeroShotImageCap/IFCap/annotations/coco/coco_with_entities.pickle")
if os.path.exists(vm_corpus):  # pilot_headroom.ROOT is the laptop path
    M.CORPUS["coco"] = vm_corpus

beams = json.load(open(sys.argv[1]))
df, logn = M.build_df("coco")
M.init(df, logn)
S = np.array([M.beam_scores([c["caption"] for c in b["beam"]]) for b in beams])
np.save(sys.argv[2], S)
print("wrote", sys.argv[2], S.shape)
