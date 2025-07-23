export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download --local-dir-use-symlinks False craigwu/vstar_bench --local-dir /cluster/home3/zhaoyutian/datasets/VstarBench --repo-type dataset

