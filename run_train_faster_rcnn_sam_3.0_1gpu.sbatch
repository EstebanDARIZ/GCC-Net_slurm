#!/bin/bash
#SBATCH --job-name=train_faster_rcnn_on_sam_3.0
#SBATCH --partition=gpu-h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4-00:01:00
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
  python -m pip uninstall -y mmcv-full
  python -m pip install -r requirements.txt
  python -m pip install -e .
  python -m pip install mlflow

  # nettoyage anciens markers
  rm -f .env_installed_*.ok
  touch "${MARKER_FILE}"
fi

### Fine-tuning Faster R-CNN (poids COCO, cf. load_from dans la config) sur
### dataset_sam_3.0 (9 classes) — 1 GPU (H100, contournement du GPU
### defectueux ECC croise a plusieurs reprises sur ce noeud avec 2 GPU).
### samples_per_gpu double a 16 pour preserver le batch total de 16
### (fidele au papier FPN) malgre le passage a 1 seul GPU -- avec dist_train.sh
### a 2 GPU, samples_per_gpu=8/GPU x 2 = 16 deja ; ici il faut compenser
### explicitement puisque mmdetection ne divise pas automatiquement comme le
### fait D-FINE avec total_batch_size.
bash tools/dist_train.sh configs/faster_rcnn/faster_rcnn_sam_3.0.py 1 \
  --cfg-options data.samples_per_gpu=16
