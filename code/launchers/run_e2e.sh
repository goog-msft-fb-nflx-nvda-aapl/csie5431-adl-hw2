export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
export PYTHONPATH=/home/jtan/adl_hw2:/home/jtan/adl_hw2/provided
export HW2_DENSE=models/ftD_hn1_lr2e5_e3 HW2_RERANK=models/rrR2_alt50
for bm in 0 1; do
  export HW2_BM25=$bm
  S=$(date +%s)
  $P -u provided/retrieval.py --output results/e2e_bm$bm.json 2>&1 | grep -E "retrieval_utils|Saved"
  echo "bm25=$bm RETRIEVAL_SECONDS $(($(date +%s)-S))"
  $P -u provided/evaluate.py --retrieval-results results/e2e_bm$bm.json --output results/eval_e2e_bm$bm.json --batch-size 16 2>&1 | grep -E "graded_|peak_cuda" | tr "\n" " "; echo
done
echo ALLDONE
