export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
for dc in 600 1000 1400; do
  $P -u code/rerank_eval.py --name qwen3rr_dc$dc --model qwen3rr --depth 30 --doc-chars $dc --bs 32 --emit-depth 30 2>&1 | grep -E "peak|scoring"
done
echo ALLDONE
