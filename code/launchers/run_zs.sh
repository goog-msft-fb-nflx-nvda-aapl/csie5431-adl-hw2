export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
$P -u code/zs_eval.py --method bm25
for m in bge-base qwen3-0.6b bge-m3; do
  $P -u code/zs_eval.py --method dense --model $m
  $P -u code/zs_eval.py --method hybrid --model $m
done
echo ALLDONE
