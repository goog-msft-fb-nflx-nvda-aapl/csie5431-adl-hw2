export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
$P -u code/rerank_eval.py --name minilm --model cross-encoder/ms-marco-MiniLM-L6-v2 --depth 50 --emit-depth 30
$P -u code/rerank_eval.py --name bgerr --model BAAI/bge-reranker-v2-m3 --depth 50 --emit-depth 30
$P -u code/rerank_eval.py --name qwen3rr --model Qwen/Qwen3-Reranker-0.6B --depth 50 --emit-depth 30 --bs 32
$P -u code/rerank_eval.py --name qwen25j --model qwen25-judge --depth 20 --bs 16 --emit-depth 20
echo ALLDONE
