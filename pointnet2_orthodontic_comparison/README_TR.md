# PointNet++ ile Ortodontik 3B Landmark Lokalizasyonu

Bu klasor, mevcut 23 landmarkli ortodontik PLY datasetine uygun PointNet++ tabanli bir karsilastirma modeli kurar.

## Optimal v2 Yaklasimi

Ilk baseline sert 3.5 mm landmark maskesi ve yalnizca XYZ koordinatlariyla egitilmisti. V2 protokolunde model landmark lokalizasyonuna daha uygun hale getirildi:

- Giris: `XYZ + yuzey normali`, yani `[x, y, z, nx, ny, nz]`.
- Hedef: sert 0/1 maske yerine Gaussian landmark heatmap.
- Loss: heatmap loss + normalize koordinat uzerinden Smooth L1 koordinat loss.
- Optimizer: `AdamW`.
- Scheduler: cosine learning-rate schedule.
- Degerlendirme: `topk_softmax` ile aktivasyon agirlikli landmark koordinati.
- Opsiyonel yogun inference: egitimden daha fazla surface point ile test.

Gaussian hedef:

```text
target_j(point) = exp(-distance(point, expert_landmark_j)^2 / (2 * sigma^2))
```

Tahmin:

```text
pred_landmark_j = softmax(logits_j / temperature)^T points
```

Metrik:

```text
ALE = mean(||predicted_landmark_j - expert_landmark_j||_2)
```

## Yerel Hizli Dogrulama

CPU veya dusuk bellekli makinede kodun calistigini hizlica dogrulamak icin:

```bash
cd pointnet2_orthodontic_comparison

../palnet_orthodontic_comparison/.venv/bin/python run_orthodontic_pointnet2.py \
  --data-root ../data/dataset \
  --transformation-dir ../palnet_orthodontic_comparison/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir runs/pointnet2_v2_smoke \
  --surface-points 512 \
  --sa1-points 128 \
  --sa2-points 32 \
  --sa3-points 8 \
  --nsample 24 \
  --epochs 2 \
  --patience 2 \
  --batch-size 2 \
  --target-mode gaussian \
  --heatmap-sigma 3.5 \
  --coord-weight 0.25 \
  --postprocess topk_softmax \
  --topk 10 \
  --device auto
```

## GPU Icin Onerilen Ana Kosu

Colab Pro, CUDA GPU veya baska bir guclu GPU ortaminda asil PointNet++ v2 deneyi:

```bash
cd pointnet2_orthodontic_comparison

../palnet_orthodontic_comparison/.venv/bin/python run_orthodontic_pointnet2.py \
  --data-root ../data/dataset \
  --transformation-dir ../palnet_orthodontic_comparison/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir runs/pointnet2_v2_gaussian_normals_p4096_e200 \
  --surface-points 4096 \
  --eval-surface-points 8192 \
  --sa1-points 1024 \
  --sa2-points 256 \
  --sa3-points 64 \
  --nsample 32 \
  --epochs 200 \
  --patience 40 \
  --batch-size 4 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --optimizer adamw \
  --scheduler cosine \
  --target-mode gaussian \
  --heatmap-sigma 3.5 \
  --use-normals \
  --coord-weight 0.25 \
  --coord-temperature 1.0 \
  --postprocess topk_softmax \
  --topk 10 \
  --temperature 1.0 \
  --device auto
```

Bellek yeterliyse `--surface-points 8192` ve `--eval-surface-points 16384` denenebilir. Bellek yetmezse once `batch-size=2`, sonra `surface-points=4096` sabit tutulmalidir.

## Yeniden Degerlendirme

Egitilmis checkpoint uzerinde yeniden egitmeden farkli test nokta sayisi veya post-processing denemek icin:

```bash
../palnet_orthodontic_comparison/.venv/bin/python run_orthodontic_pointnet2.py \
  --data-root ../data/dataset \
  --transformation-dir ../palnet_orthodontic_comparison/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir runs/pointnet2_v2_gaussian_normals_p4096_e200 \
  --surface-points 4096 \
  --eval-surface-points 16384 \
  --sa1-points 1024 \
  --sa2-points 256 \
  --sa3-points 64 \
  --nsample 32 \
  --batch-size 2 \
  --target-mode gaussian \
  --heatmap-sigma 3.5 \
  --use-normals \
  --coord-weight 0.25 \
  --postprocess topk_softmax \
  --topk 10 \
  --temperature 1.0 \
  --device auto \
  --evaluate-only \
  --model-path runs/pointnet2_v2_gaussian_normals_p4096_e200/best_model.pth
```

## Onceki Baseline Sonucu

Ilk CPU-dostu baseline sonucu:

```text
Run: runs/pointnet2_orthodontic_maskseg_p1024_e60
surface_points: 1024
train/val/test: 180/60/60
postprocess: topk_softmax, topk=10
test ALE: 5.1618
test median: 4.5412
```

V2 protokolunun amaci bu baseline'i daha yogun nokta temsili, normal bilgisi ve koordinat loss ile asagi cekmektir.

## Calistirilan V2 Sonucu

Bu repo icinde calistirilan CPU-dostu v2 kosusu:

```text
Run: runs/pointnet2_v2_gaussian_normals_p1024_e60
surface_points: 1024
target_mode: gaussian
heatmap_sigma: 3.5
use_normals: true
coord_weight: 0.25
optimizer/scheduler: AdamW + cosine
postprocess: topk_softmax, topk=10, temperature=1.0
best validation ALE: 3.9861
test ALE: 3.9706
test median: 3.3463
```

Post-processing kontrolu:

```text
topk=5, temperature=1.0:  ALE 4.0642
topk=10, temperature=1.0: ALE 3.9706 / re-eval 3.9819
topk=20, temperature=1.0: ALE 4.0197
topk=10, temperature=2.0: ALE 4.4271
eval_surface_points=4096: ALE 5.6413
```

Bu nedenle bu checkpoint icin en iyi pratik ayar `surface_points=1024`, `topk=10`, `temperature=1.0` olarak tutuldu. Yogun inference, model 1024 nokta dagilimina alistigi icin bu kosuda sonucu iyilestirmedi.

## Ciktilar

- `metrics.json`: ALE, median, landmark bazli hata ve egitim ayarlari.
- `predictions_test.csv`: uzman ve PointNet++ tahmin koordinatlari.
- `group_metrics_test.csv`: class/cinsiyet bazli ALE.
- `history.json`: epoch bazli train/validation ve learning rate.
- `best_model.pth`: en iyi validation ALE checkpoint.
