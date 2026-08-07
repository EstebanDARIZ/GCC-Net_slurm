# Faster R-CNN vanilla (ResNet50+FPN, sans fusion GCC/Retinex) sur dataset_sam_3.0 (9 classes)
# Sert de baseline de comparaison face aux configs autoassign_gcc_*.py
#
# Hyperparamètres alignés sur le papier FPN (Lin et al., "Feature Pyramid
# Networks for Object Detection", CVPR 2017, https://arxiv.org/abs/1612.03144),
# section "Implementation details" : lr=0.02, batch=16 (8 GPU x 2 img/GPU),
# 80 000 itérations (lr /10 à 60 000 it.), 2000 proposals RPN post-NMS pour
# l'entraînement (1000 pour le test), weight_decay=0.0001, momentum=0.9.
#
# Fine-tuning depuis les poids Faster R-CNN+FPN pré-entraînés sur COCO
# (load_from ci-dessous) plutôt que depuis un backbone ImageNet seul comme
# dans le papier — choix délibéré pour ce dataset. Le papier ne fournit pas
# de recette de fine-tuning (il entraîne from scratch sur COCO), donc au-delà
# du point de départ des poids, on réutilise ici tel quel son schedule
# d'optimisation.
#
# Le nombre d'epochs et le palier de lr sont recalculés pour dataset_sam_3.0
# (48 006 images d'entraînement) à partir des itérations du papier plutôt que
# repris tels quels de la convention "1x" de mmdetection (12 epochs), qui est
# elle-même calée sur la taille de COCO trainval35k (~118 000 images) :
#   80 000 it. x 16 img/it. / 48 006 img. ≈ 26,7 -> 27 epochs
#   60 000 it. (palier lr) : 60 000/80 000 x 27 = 20 epochs
_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/custom_dataset_sam_3.0.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=9)),
    train_cfg=dict(
        # 2000 proposals post-NMS pour l'entraînement (papier), au lieu du
        # défaut mmdet (1000, identique au test).
        rpn_proposal=dict(max_per_img=2000)))

# Batch total = 16 comme dans le papier (8 GPU x 2 img/GPU), obtenu ici avec
# 2 GPU x 8 img/GPU (cluster gpu-a40, cf. run_train_faster_rcnn_sam_3.0.sbatch).
# Équivalent numériquement au papier : le backbone est en norm_eval=True
# (BatchNorm gelée), donc la répartition du batch entre GPU n'affecte pas le
# calcul du gradient.
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4)

# lr=0.02 comme dans le papier pour batch=16 (schedule_1x.py par défaut est
# calé sur un batch plus petit à 0.005). Warmup conservé (absent du papier,
# mais la tête de classification/régression est réinitialisée aléatoirement
# au chargement de load_from — mismatch de forme, 9 classes vs 80 sur COCO —
# donc l'utilité du warmup pour stabiliser ce sous-réseau reste réelle).
optimizer = dict(lr=0.02)

# Un seul palier de lr à 60 000 it. (papier), pas deux comme RetinaNet.
runner = dict(max_epochs=27)
lr_config = dict(step=[20])

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MlflowLoggerHook',
            exp_name='FasterRCNN_baseline_sam3.0',
            log_model=False,   # True pour sauvegarder le modèle comme artefact MLflow
            interval=50)
    ])
