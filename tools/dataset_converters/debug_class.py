# debug_mapping.py
from mmcv import Config
from mmdet.datasets import build_dataset

cfg_path = "work_dirs/autoassign_r50_fpn_8x2_3x_gcc_custom/autoassign_r50_fpn_8x2_3x_gcc_custom.py"
cfg = Config.fromfile(cfg_path)

ds = build_dataset(cfg.data.train)

print("CLASSES:", ds.CLASSES)
print("cat_ids:", ds.cat_ids)          # ordre COCO réellement pris en compte
print("cat2label:", ds.cat2label)      # mapping cat_id -> label index (0..4)

# sanity : compter les labels internes (0..4)
from collections import Counter
cnt = Counter()
for i in range(min(len(ds), 5000)):  # limite pour aller vite
    ann = ds.get_ann_info(i)
    for lab in ann.get("labels", []):
        cnt[int(lab)] += 1
print("label counts (0..4) on first items:", cnt)


print('##############################')
print("label2cat:", {v:k for k,v in ds.cat2label.items()})


import json
from collections import Counter

p="/Utilisateurs/edreau01/datasets/datasets_coco/dataset_coco_gccnet/annotations/instances_train2017.json"
c=json.load(open(p))
cats={x["id"]:x["name"] for x in c["categories"]}
cnt=Counter(a["category_id"] for a in c["annotations"])

print("Counts by category_id:")
for k in sorted(cats):
    print(k, cats[k], cnt.get(k,0))
