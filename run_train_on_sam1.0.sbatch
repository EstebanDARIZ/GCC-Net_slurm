#!/bin/bash
#SBATCH --job-name=train_GCC-net_on_sam_1.0
#SBATCH --partition=gpu-2080ti
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --mail-user=esteban.dreaudariz@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:3

set -euo pipefail

# Aller dans le dossier du projet
cd ~/GCC-Net

PYTHON_VERSION=3.8.18
ENVIRONMENT_NAME="gccnet_env"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh

# Si l'environnement n'existe pas, le créer
if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
fi

conda activate ${ENVIRONMENT_NAME}

# Installer les dépendances une seule fois
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

### Entraînement sur dataset_sam_1.0 (8 classes)
python tools/train.py configs/autoassign/autoassign_gcc_sam.py

### Reprise depuis un checkpoint (décommenter si besoin)
#python tools/train.py configs/autoassign/autoassign_gcc_sam.py \
#  --resume-from work_dirs/autoassign_gcc_sam/epoch_15.pth \
#  --cfg-options evaluation.interval=5
