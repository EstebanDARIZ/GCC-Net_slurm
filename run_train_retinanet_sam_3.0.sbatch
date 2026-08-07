#!/bin/bash
#SBATCH --job-name=train_retinanet_on_sam_3.0
#SBATCH --partition=gpu-a40
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4-00:01:00
#SBATCH --mail-user=esteban.dreaudariz@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:2

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
  python -m pip install -U pip wheel
  python -m pip install -r requirements.txt
  python -m pip install -e .
  python -m pip install mlflow

  # nettoyage anciens markers
  rm -f .env_installed_*.ok
  touch "${MARKER_FILE}"
fi

### Entraînement RetinaNet baseline sur dataset_sam_3.0 (9 classes) — 2 GPUs distribué
### samples_per_gpu=8 x 2 GPU = batch 16, identique au papier (8 GPU x 2 img/GPU)
bash tools/dist_train.sh configs/retinanet/retinanet_sam_3.0.py 2

### Reprise depuis un checkpoint (décommenter si besoin)
#bash tools/dist_train.sh configs/retinanet/retinanet_sam_3.0.py 2 \
#  --resume-from work_dirs/retinanet_sam_3.0/epoch_8.pth \
#  --cfg-options evaluation.interval=5
