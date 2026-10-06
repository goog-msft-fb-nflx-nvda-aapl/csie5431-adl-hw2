export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
export PYTHONPATH=/home/jtan/adl_hw2/provided
$P -u code/rerank_eval.py --name qwen3rr --model qwen3rr --depth 50 --emit-depth 30 --bs 32
for n in bgerr_d30 minilm_d30 qwen25j_d20 qwen3rr_d30; do
  $P -u provided/evaluate.py --retrieval-results results/rr_$n.json --output results/eval_$n.json --batch-size 8 2>&1 | grep -E "graded_|runtime" 
done
echo ALLDONE
