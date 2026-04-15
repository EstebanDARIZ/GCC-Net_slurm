#!/bin/bash
#SBATCH --job-name=GCC-net_fix_datasets
#SBATCH --partition=gpu-2080ti
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-user=esteban.dreaudariz@gmail.com
#SBATCH --mail-type=ALL

set -euo pipefail

cd ~/GCC-Net

PYTHON_VERSION=3.8.18
ENVIRONMENT_NAME="gccnet_env"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh

if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
fi

conda activate ${ENVIRONMENT_NAME}

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

  rm -f .env_installed_*.ok
  touch "${MARKER_FILE}"
fi

# ==============================================================================
# Étape 1 : Corriger le labels_test.json de dataset_test_2.1
#   - Remplace les noms de catégories génériques ('0', '1', ...) par les vrais noms
#   - Décale les IDs de 0-based à 1-based (standard COCO)
#   - Supprime les catégories sans annotation
# ==============================================================================
echo "=== Correction du labels_test.json (dataset_test_2.1) ==="
python tools/fix_coco_categories.py \
    --input  /Utilisateurs/edreau01/datasets/dataset_test_2.1/dataset_test_2.1/labels_test.json \
    --output /Utilisateurs/edreau01/datasets/dataset_test_2.1/dataset_test_2.1/labels_test.json \
    --classes Bait_1_Squid Bait_2_Sardine Ray Sunfish Pilotfish

# ==============================================================================
# Étape 2 : Convertir les labels YOLO de dataset_test_3.0 en COCO JSON
# ==============================================================================
echo "=== Conversion YOLO → COCO JSON (dataset_test_3.0) ==="
python tools/convert_yolo_to_coco.py \
    --img-dir  /Utilisateurs/edreau01/datasets/dataset_test_3.0/images \
    --lbl-dir  /Utilisateurs/edreau01/datasets/dataset_test_3.0/labels \
    --classes  Bait_1_Squid Bait_2_Sardine Ray Sunfish Pilotfish Shark Jellyfish Fish \
    --output   /Utilisateurs/edreau01/datasets/dataset_test_3.0/labels_test.json

echo "=== Terminé ==="
