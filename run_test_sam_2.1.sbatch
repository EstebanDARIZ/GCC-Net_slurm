#!/bin/bash
#SBATCH --job-name=GCC-net_test_sam2.1
#SBATCH --partition=gpu-2080ti
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --mail-user=esteban.dreaudariz@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1

set -euo pipefail

# Aller dans le dossier du projet
cd ~/GCC-Net

PYTHON_VERSION=3.8.18
ENVIRONMENT_NAME="gccnet_env"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh

# if the environment does not exist, create it
if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
fi

conda activate ${ENVIRONMENT_NAME}

# Install deps only once
REQ_SHA=$(sha256sum requirements.txt | awk '{print $1}')
MARKER_FILE=".env_installed_${REQ_SHA}.ok"

if [ ! -f "${MARKER_FILE}" ]; then
  python -m pip install -U pip wheel setuptools

  python -m pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 \
    -f https://download.pytorch.org/whl/torch_stable.html

  python -m pip install mmcv-full==1.6.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html

  python -m pip install -r requirements.txt
  python -m pip install -e .

  # nettoyage anciens markers
  rm -f .env_installed_*.ok
  touch "${MARKER_FILE}"
fi

### Évaluation sur le split test de dataset_sam_2.1 (mAP COCO)
python tools/test.py \
  configs/autoassign/autoassign_gcc_sam_2.1.py \
  work_dirs/autoassign_gcc_sam_2.1/epoch_30.pth \
  --eval bbox \
  --show-dir work_dirs/autoassign_gcc_sam_2.1/res_test_sam_2.1 \
  --show-score-thr 0.5
