# DiffusionNet ile Ortodontik 3B Landmark Lokalizasyonu

Bu klasör, `nmwsharp/diffusion-net` projesini mevcut 23 noktalı ortodontik yüz datasetine uyarlar.

## Yaklaşım

DiffusionNet yüzey veya point cloud üzerinde vertex/point bazlı çıktı üretebilir. Bu adaptasyonda her PLY yüzeyi sabit sayıda point cloud noktasına örneklenir veya `--use-mesh-vertices` ile tüm mesh vertexleri kullanılır. Model her nokta/vertex için 23 landmark kanalında skor üretir.

Paper-benzeri önerilen yaklaşımda her landmark için uzman noktasının 3.5 mm çevresindeki noktalar pozitif maske kabul edilir:

```text
mask_j(point) = 1 if distance(point, expert_landmark_j) <= 3.5 mm else 0
```

Bu nedenle pratik ana kayıp `--loss-mode mask_bce`, ana post-process ise aktivasyon ağırlıklı koordinat hesabı için `--postprocess softmax` olarak önerilir.

Değerlendirme metriği Average Localization Error (ALE):

```text
ALE = mean(||predicted_landmark_j - expert_landmark_j||_2)
```

ALE, normalize edilmiş model koordinatlarında değil, Procrustes hizalanmış gerçek 3B koordinatlarda hesaplanır.

## Çalıştırma

```bash
cd diffusion_net_orthodontic_comparison
../palnet_orthodontic_comparison/.venv/bin/python run_orthodontic_diffusion.py \
  --data-root ../data/dataset \
  --splits-json ../shared_splits/orthodontic_180_60_60_seed42.json \
  --transformation-dir ../palnet_orthodontic_comparison/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir runs/diffusionnet_orthodontic_maskseg_xyz_p2048_k48_w64_b4_e60 \
  --surface-points 2048 \
  --k-eig 48 \
  --epochs 60 \
  --patience 15 \
  --width 64 \
  --blocks 4 \
  --mlp-hidden-dims 128 \
  --loss-mode mask_bce \
  --mask-radius 3.5 \
  --input-features xyz \
  --postprocess softmax \
  --device auto
```

## Google Colab GPU

Colab için hazır notebook:

- `colab_diffusionnet_orthodontic_gpu.ipynb`

Ayrıntılı kullanım:

- `COLAB_TR.md`

## Ortak Split

PAL-Net, DiffusionNet ve PointNet++ karsilastirmasinda ayni 180 egitim, 60 validasyon ve 60 test dosyasini kullanmak icin:

```bash
--splits-json ../shared_splits/orthodontic_180_60_60_seed42.json
```

## Çıktılar

- `metrics.json`: PAL-Net raporlarıyla aynı ana metrik ailesinde ALE özeti.
- `predictions_test.csv`: Test örneği, landmark, uzman koordinatı, DiffusionNet tahmini ve lokalizasyon hatası.
- `group_metrics_test.csv`: Class/cinsiyet bazlı ALE.
- `landmark_metrics_test.csv`: Landmark bazli mean, median, std, max ve PCK degerleri.
- `clinical_thresholds_test.csv`: PCK@2mm, PCK@2.5mm ve PCK@3mm klinik esik analizi.
- `class_metrics_test.csv`: Class I / II / III bazli performans.
- `gender_metrics_test.csv`: Female / male bazli performans.
- `difficult_landmarks_test.csv`: En zor landmark siralamasi.
- `splits.json`: Train/validation/test ayrımı.
- `best_model.pth`: En düşük validation kaybına sahip model.
