set -e
gpu=0
gpui=$((gpu))


MIMIR_DATA_SOURCE=mimirdata MIMIR_CACHE_PATH=mimrcache CUDA_VISIBLE_DEVICES=0,1 conda run --live-stream -n mimir python run.py --config configs/twfecontrols_noblocks.lite.json

cd ../dolma/which_ci/twfe/
./scripts/gpus/get_perplexity.sh twfecontrols
