export CUDA_VISIBLE_DEVICES=1 && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/suffixesblocksbin.zlib.json
export CUDA_VISIBLE_DEVICES=0 && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/suffixesnoblocksbin.zlib.json

conda run --live-stream -n analysis python build_output.py --config suffixesnoblocksbin.zlib
conda run --live-stream -n analysis python build_output.py --config suffixesblocksbin.zlib

mv suffixesblocksbin.zlib.csv csvs
mv suffixesnoblocksbin.zlib.csv csvs