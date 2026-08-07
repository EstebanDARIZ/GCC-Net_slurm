#!/bin/bash
#SBATCH --job-name=GCC-net_filter_dataset
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --mail-user=esteban.dreaudariz@gmail.com
#SBATCH --mail-type=ALL

set -euo pipefail

cd ~/GCC-Net

ENVIRONMENT_NAME="gccnet_env"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

# Crée dataset_sam_2.2 = dataset_sam_2.0 sans la classe Tuna
# (et sans les images qui la contiennent)
python tools/filter_dataset_class.py \
    --input-root  /Utilisateurs/edreau01/datasets/dataset_sam_2.0/coco_format \
    --output-root /Utilisateurs/edreau01/datasets/dataset_sam_2.2/coco_format \
    --exclude-class Tuna
