export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
runr() { n=$1; shift
  $P -u code/ft_reranker.py --out models/$n "$@" || return
  $P -u code/rerank_eval.py --name ${n}_zsc --model qwen3rr:models/$n --depth 30 --bs 16 2>&1 | grep -E "scoring|^\{" | cut -c1-700
  $P -u code/rerank_eval.py --name ${n}_ftB --model qwen3rr:models/$n --cands _ftB --depth 30 --bs 16 --emit-depth 30 2>&1 | grep -E "scoring|^\{" | cut -c1-700
}
runr rrR5_lr2e5_g16 --alt-prob 0.5 --lr 2e-5 --group 16 --qbs 2
runr rrR6_lr5e6 --alt-prob 0.5 --lr 5e-6
echo ALLDONE
