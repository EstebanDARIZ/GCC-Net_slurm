# RetinaNet vanilla (ResNet50+FPN, sans fusion GCC/Retinex) sur dataset_sam_3.0 (9 classes)
# Sert de baseline de comparaison face aux configs autoassign_gcc_*.py
_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/datasets/custom_dataset_sam_3.0.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

# Fine-tuning depuis les poids RetinaNet+FPN pré-entraînés sur COCO (comme
# pour faster_rcnn_sam_3.0.py) plutôt que depuis un backbone ImageNet seul.
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'

model = dict(
    bbox_head=dict(num_classes=9))

# Batch total = 16 comme dans le papier (8 GPU x 2 img/GPU), obtenu ici avec
# 2 GPU x 8 img/GPU (cluster gpu-a40, cf. run_train_retinanet_sam_3.0.sbatch).
# Équivalent numériquement au papier : le backbone est en norm_eval=True
# (BatchNorm gelée) et la tête RetinaNet n'a pas de BatchNorm, donc la
# répartition du batch entre GPU n'affecte pas le calcul du gradient.
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4)

# lr=0.01 pour batch=16, comme dans le papier (schedule_1x.py par défaut
# est calé sur un batch plus petit à 0.005).
optimizer = dict(lr=0.01)

# Durée d'entraînement recalée sur le nombre d'itérations du papier plutôt
# que sur son nombre d'epochs.
#
# Le papier (Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017)
# spécifie sa durée d'entraînement en nombre d'itérations : 90 000 itérations
# à batch=16, avec le learning rate divisé par 10 aux itérations 60 000 et
# 80 000. Il ne mentionne aucun nombre d'epochs. Le schedule "1x" de
# mmdetection/Detectron (max_epochs=12, step=[8, 11]) est une conversion de
# ces 90 000 itérations en epochs, mais cette conversion est calée sur la
# taille de COCO trainval35k (~118 000 images) : 90 000 it. x 16 img/it. /
# 118 000 img. ≈ 12,2 epochs. Réutiliser tel quel "12 epochs" sur un dataset
# de taille différente ne préserve donc pas le nombre d'itérations du papier.
#
# Sur dataset_sam_3.0 (48 006 images d'entraînement), reproduire les 90 000
# itérations du papier au même batch=16 donne :
#   90 000 it. x 16 img/it. / 48 006 img. ≈ 30 epochs
# et les paliers de lr (60 000/90 000 = 2/3 et 80 000/90 000 = 8/9 de la
# durée totale) se transposent à :
#   2/3 x 30 ≈ epoch 20 ; 8/9 x 30 ≈ epoch 27
runner = dict(max_epochs=30)
lr_config = dict(step=[20, 27])

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MlflowLoggerHook',
            exp_name='RetinaNet_baseline_sam3.0',
            log_model=False,   # True pour sauvegarder le modèle comme artefact MLflow
            interval=50)
    ])
