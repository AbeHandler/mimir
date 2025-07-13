set -e

export CUDA_VISIBLE_DEVICES=0,1 && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/suffixesnoblocksbin.heavy.json
export CUDA_VISIBLE_DEVICES=0,1 && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/suffixesblocksbin.heavy.json

conda run --live-stream -n analysis python build_output.py --config suffixesnoblocksbin.heavy
conda run --live-stream -n analysis python build_output.py --config suffixesblocksbin.heavy
