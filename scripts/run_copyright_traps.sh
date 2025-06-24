gpu=$1
gpui=$((gpu))
export CUDA_VISIBLE_DEVICES=$gpui && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/copywrite_traps_zeros.json
export CUDA_VISIBLE_DEVICES=$gpui && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/copywrite_traps_blocksbin.json
export CUDA_VISIBLE_DEVICES=$gpui && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/copywrite_traps_noblocksbin.json
conda run --live-stream -n analysis python build_output.py --config copywrite_traps_noblocksbin
conda run --live-stream -n analysis python build_output.py --config copywrite_traps_blocksbin
conda run --live-stream -n analysis python build_output.py --config copywrite_traps_zeros