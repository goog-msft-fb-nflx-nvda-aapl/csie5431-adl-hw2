export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 CUBLAS_WORKSPACE_CONFIG=:4096:8
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
export PYTHONPATH=/home/jtan/adl_hw2/provided
for n in "$@"; do
$P -u provided/evaluate.py --retrieval-results results/rr_${n}_d30.json --output results/eval_${n}_d30.json --batch-size 16 2>&1 | grep -E "graded_" | tr "\n" " "; echo " <- $n"
done
echo ALLDONE
