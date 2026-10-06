Raw outputs kept from the GPU experiments (the GPU working directory was deleted):
`zs.jsonl` zero-shot baselines, `ft.jsonl` fine-tuned dense variants, `rerank.jsonl` rerank sweeps,
`len.jsonl` inference-length runs, `eval_*.json` official-judge metrics and judge timing (per-candidate records
stripped), `stages.json` figure input, `q2*.jsonl` Q2 generations, `e2e_final2_default.json` the submitted
pipeline's public ranked lists, `judge_train.npy` judge labels for the top-20 zero-shot candidates of every train
dialogue (int8 [4830, 20]: 1 accepted, 0 rejected, 2 gold; the expensive step 2 of run.sh).
