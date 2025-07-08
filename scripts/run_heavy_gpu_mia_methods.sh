set -e
export CUDA_VISIBLE_DEVICES=0,1 && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/minhashblocksample_blocks.heavy.json
export CUDA_VISIBLE_DEVICES=0,1 && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/minhashblocksample_noblocks.heavy.json

conda run --live-stream -n analysis python build_output.py --config minhashblocksample_noblocks.heavy
conda run --live-stream -n analysis python build_output.py --config minhashblocksample_blocks.heavy
