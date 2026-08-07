# RetinaNet vanilla (ResNet50+FPN, sans fusion GCC/Retinex) sur dataset_sam_3.0 (9 classes)
# Sert de baseline de comparaison face aux configs autoassign_gcc_*.py
_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/datasets/custom_dataset_sam_3.0.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
model = dict(
    bbox_head=dict(num_classes=9))

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
