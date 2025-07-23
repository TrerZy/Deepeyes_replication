
# Script to get scores for VstarBench evaluation

python /cluster/home3/zhaoyutian/code/deepeyes/DeepEyes/eval/judge_result.py \
    --model_name deepeyes \
    --api_url http://0.0.0.0:8000/v1 \
    --vstar_bench_path /cluster/home3/zhaoyutian/datasets/VstarBench \
    --save_path /cluster/home3/zhaoyutian/code/deepeyes/DeepEyes/results/Vstar_result \
    --eval_model_name deepeyes \
    --num_workers 8



