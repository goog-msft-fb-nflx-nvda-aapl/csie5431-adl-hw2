export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
export PYTHONPATH=/home/jtan/adl_hw2/provided
$P -u code/cands.py --split public --model models/ftB_hn1 --tag _ftB --n 100
$P -u code/rerank_eval.py --name ftB_qwen3rr --model qwen3rr --cands _ftB --depth 30 --bs 16 --emit-depth 30 2>&1 | grep -E "peak|scoring|^\{" | cut -c1-900
$P -u provided/evaluate.py --retrieval-results results/rr_ftB_qwen3rr_d30.json --output results/eval_ftB_qwen3rr_d30.json --batch-size 16 2>&1 | grep -E "graded_|runtime"
echo ALLDONE
