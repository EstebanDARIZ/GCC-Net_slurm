#!/bin/bash
#SBATCH --job-name=GCC-net_train
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
# You need this to be able to use the 'conda' command. I am not satisfied with this solution, I'll try to find a way so that you won't have to set it in the future.
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh


# if the environment does not exist, create it
if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
fi

conda activate ${ENVIRONMENT_NAME}

# Install python packages, either individually or using a requirements.txt file.
# pip install scipy matplotlib
# or

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

### Command to lunch training from scratch
#python tools/train.py configs/autoassign/autoassign_r50_fpn_8x2_3x_gcc_custom.py 

python tools/train.py configs/autoassign/autoassign_r50_fpn_8x2_3x_gcc_custom.py



### Command to lunch training from a specifique epoch (after a crash) 
#python tools/train.py configs/autoassign/autoassign_r50_fpn_8x2_3x_gcc_custom.py \
#  --resume-from work_dirs/autoassign_r50_fpn_8x2_3x_gcc_custom/epoch_15.pth \
#  --cfg-options evaluation.interval=5
