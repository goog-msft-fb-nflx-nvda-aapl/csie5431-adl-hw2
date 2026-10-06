export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
export PYTHONPATH=/home/jtan/adl_hw2:/home/jtan/adl_hw2/provided
export HW2_DENSE=models/ftD_hn1_lr2e5_e3 HW2_RERANK=models/rrR2_alt50 HW2_BM25=1
run() { tag=$1; shift
  S=$(date +%s)
  env "$@" $P -u provided/retrieval.py --output results/e2e_$tag.json 2>&1 | grep -E "retrieval_utils" | tr "\n" " "
  echo "$tag RETRIEVAL_SECONDS $(($(date +%s)-S))"
  $P -u provided/evaluate.py --retrieval-results results/e2e_$tag.json --output results/eval_e2e_$tag.json --batch-size 16 2>&1 | grep -E "graded_" | tr "\n" " "; echo " <- $tag"
}
run L256_d20_c1200 HW2_DENSE_LEN=256 HW2_DEPTH=20 HW2_DOC_CHARS=1200
run L384_d30_c1800 HW2_DENSE_LEN=384 HW2_DEPTH=30 HW2_DOC_CHARS=1800
echo ALLDONE
