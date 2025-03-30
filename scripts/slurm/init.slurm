#!/bin/bash

#SBATCH --partition=aa100
#SBATCH --time=12:00:00
#SBATCH --gres=gpu
#SBATCH --job-name=initmimr
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abha4861@colorado.edu

source /curc/sw/anaconda3/latest
export TMPDIR=/scratch/alpine/abha4861/tmp
export PIP_CACHE_DIR=/scratch/alpine/abha4861/tmp/pip

set -e

rm -rf mimir

git clone git@github.com:AbeHandler/mimir.git

conda env remove -n mimir
conda create -n mimir python=3.10 -y
conda activate mimir

cd mimir
TMPDIR=/scratch/alpine/abha4861/tmp/pip PIP_CACHE_DIR=/scratch/alpine/abha4861/tmp/pip pip install -r requirements_w_versions.txt
export TMP=/scratch/alpine/$USER
export TEMP=/scratch/alpine/$USER
export TMPDIR=/scratch/alpine/$USER
export TEMPDIR=/scratch/alpine/$USER
export HF_HOME='/scratch/alpine/abha4861/RobotEconomist/transformers'

export MIMIR_CACHE_PATH='/scratch/alpine/abha4861/dolma2/dolma/mimr/cache'
export MIMIR_DATA_SOURCE='/scratch/alpine/abha4861/dolma2/dolma/mimr/data'
huggingface-cli login --token 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

python run.py --config configs/mi.json
