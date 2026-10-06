export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
export PYTHONPATH=/home/jtan/adl_hw2:/home/jtan/adl_hw2/provided
export HW2_DENSE=models/dense_final2 HW2_RERANK=models/reranker_final2 HW2_BM25=1
run() { tag=$1; shift
  S=$(date +%s)
  env "$@" $P -u provided/retrieval.py --output results/e2e_$tag.json 2>&1 | grep -E "retrieval_utils|Error|Trace" | tr "\n" " "
  echo "$tag RETRIEVAL_SECONDS $(($(date +%s)-S))"
  $P -u provided/evaluate.py --retrieval-results results/e2e_$tag.json --output results/eval_e2e_$tag.json --batch-size 16 2>&1 | grep -E "graded_" | tr "\n" " "; echo " <- $tag"
}
run final2_default
echo ALLDONE
