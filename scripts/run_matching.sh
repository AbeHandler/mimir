gpu=0
gpui=$((gpu))
export CUDA_VISIBLE_DEVICES=$gpui && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/nobloxbypublisher.lite.json

conda run --live-stream -n analysis python build_output.py --config nobloxbypublisher.lite