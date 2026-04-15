#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-2080ti  # Partition is a queue for jobs
#SBATCH --time=24:00:00         # Time limit for the job
#SBATCH --job-name=train_efficientdet_D0   # Name of your job
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1               # Number of nodes you want to run your process on
#SBATCH --ntasks-per-node=1     # Number of CPU cores
#SBATCH --mem=10GB
#SBATCH --mail-user=esteban.dreaudariz@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:3            # Number of GPUs

# Aller dans le dossier du projet
cd ~/efficientdet

PYTHON_VERSION=3.12
ENVIRONMENT_NAME="my_test_env"

module load Anaconda3
# You need this to be able to use the 'conda' command. I am not satisfied with this solution,>
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh


# if the environment does not exist, create it
if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
fi

conda activate ${ENVIRONMENT_NAME}

# Install python packages, either individually or using a requirements.txt file.
# pip install scipy matplotlib
# or
# Torch CUDA 11.8
pip install --upgrade pip
#pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url 
pip install --no-cache-dir \
  torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118


pip install -r requirements.txt

#python inference_2.py \
#    --model efficientdet_d0 \
#    --checkpoint /Utilisateurs/edreau01/efficientdet/output/train/20251205-170046-tf_efficientdet_d0/model_best.pth.tar \
#    --image /Utilisateurs/edreau01/datasets/datasets_coco/dataset_coco/train2017/R_frame_16430.jpg \
#    --num-classes 5 \
#    --output out.jpg


python inference_2.py \
  --model tf_efficientdet_d0 \
  --checkpoint /Utilisateurs/edreau01/efficientdet/output/res_dataset_sam_0.1/train/20260408-110134-tf_efficientdet_d0/model_best.pth.tar \
  --image-dir /Utilisateurs/edreau01/datasets/dataset_test_3.0/images \
  --num-classes 8 \
  --image-size 512 \
  --output-dir /Utilisateurs/edreau01/efficientdet/output/res_dataset_sam_0.1/train/20260408-110134-tf_efficientdet_d0/res_test_3.0 \
  --device cuda \
  --score-thresh 0.5 \
  --classes "Bait_1_Squid,Bait_2_Sardine,Ray,Sunfish,Pilotfish,Shark,Jellyfish,,Fish"
