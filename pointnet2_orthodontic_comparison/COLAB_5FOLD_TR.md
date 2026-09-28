# PointNet++ Leakage-Free 5-Fold CV

Bu protokol PointNet++'i PAL-Net, DiffusionNet ve AGH-Former vNext ile ayni akademik kosullarda degerlendirir:

- Ortak manifest: `orthodontic_5fold_192_48_60_seed42.json`
- Her fold: `192 train / 48 validation / 60 test`
- Hizalama: yalniz fold-train meshleriyle fit edilen label-free rigid mesh ICP
- Fiziksel olcek: korunur, ALE milimetre olarak raporlanir
- Test politikasi: validation checkpoint kilitlendikten sonra fold testi bir kez degerlendirilir

Hazir notebook: `colab_pointnet2_5fold.ipynb`

## 1. Drive ve kod

Google Drive'da su klasorlerin bulunmasi gerekir:

```text
/content/drive/MyDrive/orthodontic/data/dataset
/content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/publication_cv_stage1_v4_seed42
```

Kod deposunu guncelleyin:

```python
from pathlib import Path
import subprocess

code_root = Path("/content/comparative-study")
repo_url = "https://github.com/eckdev/comparative-study.git"
if not code_root.exists():
    subprocess.run(["git", "clone", repo_url, str(code_root)], check=True)
else:
    subprocess.run(["git", "-C", str(code_root), "pull"], check=True)

subprocess.run(
    ["python", "-m", "pip", "install", "-q", "-r", str(code_root / "pointnet2_orthodontic_comparison/requirements.txt")],
    check=True,
)
```

## 2. Preflight

Egitim yapmadan bes foldun split ve ICP dosyalarini denetler:

```python
%cd /content/comparative-study/pointnet2_orthodontic_comparison
!python -u colab_run_pointnet2_cv.py --preset=preflight --seed=42
```

Her fold icin `Preflight passed ... 192/48/60` gorulmelidir.

## 3. Smoke test

Pipeline'i Fold 1'de 24 ornek ve 2 epoch ile kontrol eder. Smoke ALE bilimsel sonuc degildir.

```python
%cd /content/comparative-study/pointnet2_orthodontic_comparison
!python -u colab_run_pointnet2_cv.py --preset=smoke --seed=42
```

## 4. Ana 5-fold kosu

Colab A100/L4 GPU ile:

```python
%cd /content/comparative-study/pointnet2_orthodontic_comparison
!python -u colab_run_pointnet2_cv.py --preset=cv --seed=42
```

Oturum kesilirse ayni komutu tekrar calistirin. Tamamlanan foldlar atlanir; yarida kalan fold `last_model.pth` dosyasindan devam eder.

Belirli foldlari calistirmak icin:

```python
!python -u colab_run_pointnet2_cv.py --preset=cv --seed=42 --fold-indices=3,4,5
```

## 5. Sonuclar

Ana klasor:

```text
/content/drive/MyDrive/orthodontic/pointnet2_runs/pointnet2_publication_cv_seed42
```

Bes fold tamamlaninca otomatik uretilen ana dosyalar:

```text
summary_metrics.json
summary_fold_metrics.csv
summary_landmark_metrics.csv
summary_class_metrics.csv
summary_gender_metrics.csv
pooled_predictions_test.csv
```

Her fold icinde `metrics_val.json`, `metrics.json`, `predictions_val.csv`, `predictions_test.csv`, `history.json`, `best_model.pth` ve leakage raporu bulunur.

## Dondurulmus ayarlar

Ana protokol `publication_cv_protocol.json` icinde saklanir. PointNet++ kosusu `4096` nokta, XYZ+normal, Gaussian heatmap, AdamW, cosine scheduler ve `topk=20` kullanir. Bu ayarlar outer-test sonucuna bakilarak degistirilmemelidir.
