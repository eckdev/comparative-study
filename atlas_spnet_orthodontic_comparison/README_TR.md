# Atlas-SPNet Ortodontik Karşılaştırma

Atlas-SPNet, 3B yüz landmark lokalizasyonu için leakage-free deney protokolüyle hazırlanan bağımsız bir model hattıdır. Model global yüz formunu, lokal landmark çevresini, anatomik landmark ilişkilerini ve confidence tahminini birlikte kullanır.

Ana güvenlik kararları:

- Varsayılan patient key: `gender_subject`.
- Fold template'i yalnız train fold landmarklarından hesaplanır.
- Normalization yalnız train fold üzerinden fit edilir.
- Procrustes `scale=false`, `reflection=false` mantığıyla rijit hizalama yapar.
- Alignment raporu val/test rigid hizalamanın expert landmark kullandığını açıkça raporlar; bu, template leakage'ini kaldırır ama inference-time label-free registration yerine geçmez.

Colab smoke:

```bash
cd /content/comparative-study/atlas_spnet_orthodontic_comparison
python -u colab_run_atlas_spnet.py --preset smoke
```

A100 ana koşu:

```bash
python -u colab_run_atlas_spnet.py --preset full
```

Manuel örnek:

```bash
python -u run_atlas_spnet.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --output-dir /content/drive/MyDrive/orthodontic/atlas_spnet_runs/full_atlas_spnet \
  --patient-key gender_subject \
  --folds 5 \
  --surface-points 12000 \
  --patch-points 512 \
  --epochs 200 \
  --patience 35 \
  --batch-size 2 \
  --width 192 \
  --device auto \
  --mixed-precision
```

Temel çıktılar:

- `summary_metrics.json`
- `summary_fold_metrics.csv`
- `summary_landmark_metrics.csv`
- `bootstrap_ci.json`
- `outlier_samples.csv`
- `fold_*/metrics.json`
- `fold_*/predictions_test.csv`
- `fold_*/alignment_report.json`
- `fold_*/normalization.json`

