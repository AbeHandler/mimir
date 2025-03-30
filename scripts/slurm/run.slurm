#!/bin/bash

#SBATCH --partition=aa100
#SBATCH --time=24:00:00
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

git clone git@github.com:iamgroot42/mimir.git

conda env remove -n mimir
conda create -n mimir python=3.10 -y
conda activate mimir

# You need to comment out line 11 and 25 here
# remove all lines that ref DC_PDDAttack in file b/c that class throws error
# vi mimir/mimir/attacks/utils.py
sed -i -e '11s/^/#/' -e '25s/^/#/' mimir/mimir/attacks/utils.py
sed -i '74s/)/, trust_remote_code=True)/' mimir/mimir/custom_datasets.py

cd mimir
TMPDIR=/scratch/alpine/abha4861/tmp/pip PIP_CACHE_DIR=/scratch/alpine/abha4861/tmp/pip pip install -r requirements.txt
export TMP=/scratch/alpine/$USER
export TEMP=/scratch/alpine/$USER
export TMPDIR=/scratch/alpine/$USER
export TEMPDIR=/scratch/alpine/$USER
export HF_HOME='/scratch/alpine/abha4861/RobotEconomist/transformers'

export MIMIR_CACHE_PATH='/scratch/alpine/abha4861/dolma2/dolma/mimr/cache'
export MIMIR_DATA_SOURCE='/scratch/alpine/abha4861/dolma2/dolma/mimr/data'
huggingface-cli login --token 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

python run.py --config configs/mi.json
