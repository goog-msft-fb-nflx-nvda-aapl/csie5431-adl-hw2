export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
$P - <<PY
import sys,numpy as np
sys.path[:0]=["provided"]
from eval_utils import load_products
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
p=list(load_products("data/public/products.jsonl").values())
rng=np.random.RandomState(0); s=[p[i] for i in rng.choice(len(p),3000,replace=False)]
n=np.array([len(x) for x in tok(s)["input_ids"]])
print("passage tokens mean %.0f median %.0f p90 %.0f ; mean after trunc 512 %.0f 384 %.0f 256 %.0f"%(n.mean(),np.median(n),np.percentile(n,90),np.minimum(n,512).mean(),np.minimum(n,384).mean(),np.minimum(n,256).mean()))
PY
for L in 256 384; do
  $P -u code/zs_eval.py --method dense --model models/ftD_hn1_lr2e5_e3 --max-len $L --out results/len.jsonl 2>&1 | grep -E "pool encode|^\{" | cut -c1-260
done
echo ALLDONE
