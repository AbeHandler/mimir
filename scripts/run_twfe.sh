set -e
gpu=0
gpui=$((gpu))
 
# I am going to rerun controls b/c it is out of sync locally
# export CUDA_VISIBLE_DEVICES=$gpui && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/twfe_blocks.lite.json
export CUDA_VISIBLE_DEVICES=$gpui && MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache conda run --live-stream -n mimir python run.py --config configs/twfecontrols_blocks.lite.json

#conda run --live-stream -n analysis python build_output.py --config twfe_blocks.lite
#conda run --live-stream -n analysis python build_output.py --config twfecontrols_blocks.lite
