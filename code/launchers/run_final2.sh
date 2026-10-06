export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
P=/home/jtan/miniconda3/envs/adl_hw2_env/bin/python3
cd /home/jtan/adl_hw2
case $1 in
 dense) CUDA_VISIBLE_DEVICES=2 $P -u code/ft_biencoder.py --out models/dense_final2 --n-neg 7 --lr 3e-5 --epochs 4 --dev-frac 0 ;;
 rerank) CUDA_VISIBLE_DEVICES=1 $P -u code/ft_reranker.py --out models/reranker_final2 --alt-prob 0.5 --lr 2e-5 --group 16 --qbs 2 --use-all-train ;;
esac
echo FINISHED $1
