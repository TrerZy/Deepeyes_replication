
CUDA_VISIBLE_DEVICES=1,4

vllm serve ./pretrained_model \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --served-model-name "deepeyes" \
    --trust-remote-code \
    --limit-mm-per-prompt "image=11" \
    --disable-log-requests
