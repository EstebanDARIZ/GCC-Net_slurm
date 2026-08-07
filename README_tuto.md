# GCC-Net — Quick Start & Tutorial

This document explains how to train, evaluate and run inference with GCC-Net in this repository.

Pour utiliser le venv du repo GCC-Net, il faut faire :
source /home/esteban-dreau-darizcuren/doctorat/code/detector/GCC-Net/.venv/bin/activate


## Data layout & preparation
Expected layout (example):
```
data/
    annotations/
        instances_val2017.json
        instances_train2017.json
    val/
        images/
    train/
        images/
```

---

## Configuration
You need to update this file with your config : 
```bash
vim configs/autoassign/autoassign_r50_fpn_8x2_3x_gcc_custom.py
```
Key settings to check: 
- samples_per_gpu 
- workers_per_gpu
- lr : value 
- max_epoch 
- num_classes 

This file needs this one to work :
'''bash
nano configs/_base_/datasets/custom_dataset_detection.py
'''
Key settings to check:
- data_root : path to the dataset to train on 
- data_test : path to the dataset test
- classes : the name of the classes
- samples_per_gpu : 
- workers_per_gpu : 
- the corect path to the labels and images

---

## Training
Typical command:
```bash
python train.py --config configs/gcc_net_local.yaml
```
Common options (if supported):
- --config <file>
- --resume <checkpoint.pth>
- --gpus 0,1
- --epochs 50
- --batch-size 16


Training artifacts:
- checkpoints/ (model .pth files)
- logs/ (TensorBoard or plain text logs)

---

## Validation / Testing
Run test to evaluate checkpoints:
```bash
python tools/test.py --config configs/gcc_net_local.yaml --checkpoint checkpoints/epoch_20.pth --out results/
```
Outputs:
- metrics (mAP, precision/recall)
- per-class scores
- optional result files for submission

---



python tools/inference.py \
    /home/esteban-dreau-darizcuren/doctorat/code/detector/GCC-Net_slurm/configs/autoassign/autoassign_gcc_sam.py \
    /home/esteban-dreau-darizcuren/doctorat/code/detector/GCC-Net_slurm/latest.pth\
    /home/esteban-dreau-darizcuren/doctorat/dataset/datasets_test/dataset_test_3.0 \
    /home/esteban-dreau-darizcuren/doctorat/code/detector/GCC-Net_slurm/work_dirs/autoassign_gcc_sam/res_test_3.0 \
    --score-thr 0.5

