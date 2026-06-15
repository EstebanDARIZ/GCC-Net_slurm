# Inférence GCC-Net sur vidéos — `inf_video`

Ce module permet de lancer une inférence GCC-Net sur toutes les vidéos contenues dans un dossier (parcouru récursivement). Pour chaque vidéo, un fichier `.csv` est généré à côté de la vidéo avec les détections.

---

## Fichiers concernés

| Fichier | Rôle |
|---|---|
| `tools/inf_video.py` | Script Python principal d'inférence |
| `run_inf_video.sbatch` | Script SLURM pour lancer l'inférence sur le cluster |

---

## Lancement via SLURM (cluster)

```bash
sbatch run_inf_video.sbatch
```

Le job SLURM est configuré par défaut pour :
- **Partition** : `gpu-2080ti`
- **GPU** : 1
- **CPU** : 8 cœurs
- **RAM** : 48 Go
- **Durée max** : 3 jours 24h
- **Notifications mail** : à toutes les étapes (`ALL`)

Le script charge automatiquement l'environnement conda `gccnet_env`, installe les dépendances si nécessaire, puis lance l'inférence sur le dossier `/Utilisateurs/edreau01/datasets/BORIS`.

Pour changer le dossier cible ou les paramètres, modifier la dernière section du fichier `run_inf_video.sbatch` :

```bash
python tools/inf_video.py \
  configs/autoassign/autoassign_gcc_sam_2.0.py \
  work_dirs/autoassign_gcc_sam_2.0/epoch_30.pth \
  /chemin/vers/vos/videos \
  --score-thr 0.5 --frame-step 15 --skip-existing
```

---

## Lancement manuel (sans SLURM)

```bash
conda activate gccnet_env
cd ~/GCC-Net

python tools/inf_video.py \
  configs/autoassign/autoassign_gcc_sam_2.0.py \
  work_dirs/autoassign_gcc_sam_2.0/epoch_30.pth \
  /chemin/vers/vos/videos \
  --score-thr 0.3 --frame-step 5 --skip-existing
```

---

## Arguments

### Arguments positionnels (obligatoires)

| Argument | Description |
|---|---|
| `config` | Chemin vers le fichier de config du modèle (`.py`) |
| `checkpoint` | Chemin vers le checkpoint entraîné (`.pth`) |
| `input_dir` | Dossier racine contenant les vidéos (parcouru récursivement) |

### Arguments optionnels

| Argument | Défaut | Description |
|---|---|---|
| `--score-thr` | `0.3` | Seuil de confiance minimum pour retenir une détection |
| `--frame-step` | `1` | Analyser 1 frame toutes les N frames (ex: `15` = 1 frame/15) |
| `--gpu-id` | `0` | ID du GPU à utiliser |
| `--device` | *(auto)* | Forcer le device : `cpu`, `cuda:0`, etc. Écrase `--gpu-id` |
| `--skip-existing` | *(désactivé)* | Ignorer les vidéos dont le CSV existe déjà (reprise) |
| `--no-confirm` | *(désactivé)* | Désactiver la confirmation temporelle des détections |
| `--confirm-frames` | `5` | Fenêtre de confirmation : ±N frames autour de la détection |
| `--confirm-thr` | `3` | Nombre minimum de confirmations requises dans la fenêtre |
| `--classes` | *(toutes)* | Ne garder que certaines classes (ex: `--classes 1 5`) |

---

## Formats vidéo supportés

`.mp4`, `.MP4`, `.avi`, `.AVI`, `.mov`, `.MOV`, `.mkv`, `.MKV`

---

## Sortie : format CSV

Un fichier `.csv` est créé à côté de chaque vidéo traitée, avec le même nom de fichier.

**Exemple** : `video_boris_01.mp4` → `video_boris_01.csv`

### Colonnes

| Colonne | Description |
|---|---|
| `frame_idx` | Indice de la frame dans la vidéo |
| `timecode` | Timecode au format `MM:SS.mmm` |
| `class_id` | Identifiant numérique de la classe détectée |
| `class_name` | Nom de la classe détectée |
| `score` | Score de confiance de la détection (0–1) |
| `x1`, `y1`, `x2`, `y2` | Coordonnées de la bounding box (pixels) |
| `inference_time_ms` | Temps d'inférence pour cette frame (ms) |

Des métadonnées de performance sont ajoutées en commentaire en fin de fichier (`# total_time_s`, `# avg_inference_ms`, etc.).

---

## Confirmation temporelle

Par défaut, le script applique une **confirmation temporelle** : une détection n'est retenue que si elle est confirmée par au moins `--confirm-thr` frames dans une fenêtre de ±`--confirm-frames` frames autour. Cela réduit les faux positifs isolés.

Pour désactiver ce filtre :

```bash
python tools/inf_video.py ... --no-confirm
```

---

## Reprise d'une inférence interrompue

Utiliser `--skip-existing` pour ne retraiter que les vidéos sans CSV :

```bash
python tools/inf_video.py ... --skip-existing
```
