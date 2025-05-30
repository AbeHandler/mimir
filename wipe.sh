find . -maxdepth 1 -type f | grep snake | xargs rm
rm -rf ~/.cache/huggingface/datasets/ && mkdir -p ~/.cache/huggingface/datasets/
