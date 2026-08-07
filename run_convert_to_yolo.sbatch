#!/bin/bash
#SBATCH --job-name=GCC-net_coco_to_yolo
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

# dataset_sam_2.0 (coco_format) -> dataset_sam_2.0/yolo_format
python tools/convert_coco_to_yolo.py \
    --coco-root   /Utilisateurs/edreau01/datasets/dataset_sam_2.0/coco_format \
    --output-root /Utilisateurs/edreau01/datasets/dataset_sam_2.0/yolo_format

# dataset_sam_2.2 (coco_format) -> dataset_sam_2.2/yolo_format
python tools/convert_coco_to_yolo.py \
    --coco-root   /Utilisateurs/edreau01/datasets/dataset_sam_2.2/coco_format \
    --output-root /Utilisateurs/edreau01/datasets/dataset_sam_2.2/yolo_format
