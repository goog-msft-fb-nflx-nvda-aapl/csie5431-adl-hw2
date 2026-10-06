export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
run() { # name, extra args
  n=$1; shift
  $P -u code/ft_biencoder.py --out models/$n "$@" && \
  $P -u code/zs_eval.py --method dense --model models/$n --out results/ft.jsonl && \
  $P -u code/zs_eval.py --method hybrid --model models/$n --out results/ft.jsonl
}
run ftC_hn3 --n-neg 3
run ftD_hn1_lr2e5_e3 --n-neg 1 --epochs 3 --lr 2e-5
run ftE_hn1_lo0 --n-neg 1 --neg-lo 0 --neg-hi 30
echo ALLDONE
