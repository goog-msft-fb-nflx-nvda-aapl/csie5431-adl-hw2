export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
export PYTHONPATH=/home/jtan/adl_hw2:/home/jtan/adl_hw2/provided
S=$(date +%s)
$P -u provided/retrieval.py --output results/rr_bm25_qwen3.json
echo RETRIEVAL_SECONDS $(($(date +%s)-S))
$P -u provided/evaluate.py --retrieval-results results/rr_bm25_qwen3.json --output results/eval_bm25_qwen3.json --batch-size 8
echo ALLDONE
