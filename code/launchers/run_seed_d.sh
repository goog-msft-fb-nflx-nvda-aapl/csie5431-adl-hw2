export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
run() { # name, extra args
  n=$1; shift
  $P -u code/ft_biencoder.py --out models/$n "$@" && \
  $P -u code/zs_eval.py --method dense --model models/$n --out results/ft.jsonl && \
  $P -u code/zs_eval.py --method hybrid --model models/$n --out results/ft.jsonl
}
run ftF_s1 --n-neg 3 --lr 2e-5 --epochs 3 --seed 1
run ftF_s2 --n-neg 3 --lr 2e-5 --epochs 3 --seed 2
echo ALLDONE
