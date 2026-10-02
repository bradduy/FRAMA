import sys, os, numpy as np
FRAMA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FRAMA_DIR)
sys.argv=[sys.argv[0],"1"]
import importlib.util
src=open(os.path.join(FRAMA_DIR, "analyze_ablation.py")).read()
src=src.split("data = {b: load_bench")[0]
src=src.replace("HERE = os.path.dirname(os.path.abspath(__file__))","HERE = " + repr(FRAMA_DIR))
exec(src)
from scipy.stats import spearmanr
names=["LM","G","B","S","M","D"]
for bench in ("coco","nocaps","flickr"):
    beams,M,Z=load_bench(bench)
    N,K,A=Z.shape
    R=np.zeros((A,A)); cnt=np.zeros((A,A))
    for i in range(N):
        for a in range(A):
            for b in range(a+1,A):
                x,y=Z[i,:,a],Z[i,:,b]
                if x.std()<1e-9 or y.std()<1e-9: continue
                r=spearmanr(x,y).correlation
                if np.isnan(r): continue
                R[a,b]+=r; cnt[a,b]+=1
    R=R/np.maximum(cnt,1)
    print(bench)
    for a in range(A):
        print(" ".join(f"{names[a]}-{names[b]}:{R[a,b]:+.2f}" for b in range(a+1,A)))
