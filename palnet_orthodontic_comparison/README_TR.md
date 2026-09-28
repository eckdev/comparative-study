# PAL-Net ile 23 Noktalı Ortodontik Yumuşak Doku Analizi

Bu klasör, `data/dataset` altındaki sınıflandırılmış 3B hasta yüz meshlerini ve uzman ortodontist tarafından işaretlenen 23 landmark noktasını PAL-Net projesine uygular.

## İçerik

- `upstream/`: GitHub'dan indirilen PAL-Net kodu.
- `upstream/src/datasets/orthodontic_dataset.py`: Bu veri setinin `Class*/men|women/*.ply` ve `Class*-Landmark/*.txt` düzenini PAL-Net formatına çeviren dataset adapter'ı.
- `upstream/run_orthodontic.py`: Eğitim, doğrulama, test ve ALE raporlama script'i.
- `colab_run_palnet_cv.py`: Ortak 5-fold manifest ve fold-specific label-free ICP ile Colab çalıştırıcısı.
- `publication_cv_protocol.json`: Makale koşusundan önce dondurulmuş PAL-Net hiperparametreleri ve değerlendirme protokolü.
- `colab_palnet_orthodontic_gpu.ipynb`: Google Colab GPU uzerinde calistirilacak notebook.
- `colab_palnet_5fold.ipynb`: Leakage-free makale 5-fold koşusu için hazır Colab notebook'u.
- `COLAB_TR.md`: Colab Drive yapisi ve kosu presetleri.
- `runs/`: Çalıştırmaların model, metrik ve tahmin çıktıları buraya yazılır.

## Metrik

Sonuç metriği Average Localization Error (ALE) olarak hesaplanır:

```text
ALE = mean(||PAL-Net landmark_i - uzman landmark_i||_2)
```

Hesap 23 landmark ve test örneklerinin tamamı üzerinden yapılır. Script iki PAL-Net sonucu üretir:

- `palnet_raw`: Ağın doğrudan 3B koordinat tahmini.
- `palnet_snapped`: Tahminin en yakın örneklenmiş mesh yüzey noktasına taşınmış hali. Ana karşılaştırma için bunu kullanmak daha tutarlıdır, çünkü landmark yüz üzerinde olmalıdır.

Ek olarak `mean_shape_baseline_*` değerleri de yazılır. Bu değer PAL-Net değildir; eğitim setinin ortalama uzman landmark şablonunu test yüzlerine karşı ölçen basit referanstır.

## Kurulum

```bash
cd palnet_orthodontic_comparison/upstream
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r ../requirements.txt
```

## Çalıştırma

Hızlı bir deneme:

```bash
cd palnet_orthodontic_comparison/upstream
source ../.venv/bin/activate
python run_orthodontic.py --epochs 3 --batch-size 4 --patch-size 250 --surface-points 5000 --splits-json ../../shared_splits/orthodontic_180_60_60_seed42.json
```

Daha gerçekçi eğitim:

```bash
python run_orthodontic.py --epochs 200 --batch-size 8 --patch-size 250 --surface-points 10000 --splits-json ../../shared_splits/orthodontic_180_60_60_seed42.json
```

2 asamali PAL-Net + residual refiner kosusu:

```bash
python run_orthodontic.py \
  --data-root ../../data/dataset \
  --splits-json ../../shared_splits/orthodontic_180_60_60_seed42.json \
  --transformation-dir ../transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir ../runs/palnet_refiner_p1000_surface100k_e200 \
  --epochs 200 \
  --patience 40 \
  --batch-size 2 \
  --patch-size 1000 \
  --surface-points 100000 \
  --template-mode class_gender \
  --train-refiner \
  --refine-center stage1 \
  --residual-target \
  --landmark-weighting val_error \
  --center-jitter-mm 2.0 \
  --point-noise-mm 0.1 \
  --point-dropout 0.05 \
  --refiner-snap-k-candidates 1,3,5
```

Uc seed ensemble icin ayni komutu farkli `--seed` ve `--output-dir` ile calistirdiktan sonra:

```bash
python ensemble_palnet_predictions.py \
  --predictions ../runs/run_seed1/refined_predictions_test.csv ../runs/run_seed2/refined_predictions_test.csv ../runs/run_seed3/refined_predictions_test.csv \
  --output-dir ../runs/palnet_refiner_ensemble
```

## Ortak Split

Makaledeki ana karşılaştırma için PAL-Net, DiffusionNet ve AGH-Former vNext ile aynı outer fold örneklerini kullanır:

```bash
shared_splits/orthodontic_5fold_192_48_60_seed42.json
```

Her fold `192 train / 48 validation / 60 outer-test` içerir. Beş outer-test kümesi birbirinden ayrıdır ve 300 örneğin her biri tam bir kez test edilir. Hizalama, ilgili fold'un yalnız train meshleriyle fit edilen, ölçek değiştirmeyen ve uzman landmark kullanmayan rigid mesh ICP dönüşümleridir.

Eski `orthodontic_180_60_60_seed42.json` koşuları geliştirme/baseline geçmişi olarak korunur; makaledeki 5-fold sonuç tablosuna doğrudan karıştırılmamalıdır.

Colab 5-fold akışı:

```bash
cd /content/comparative-study/palnet_orthodontic_comparison
python -u colab_run_palnet_cv.py --preset preflight --seed 42
python -u colab_run_palnet_cv.py --preset smoke --seed 42
python -u colab_run_palnet_cv.py --preset cv --seed 42
```

`cv` preset'i dondurulmuş PAL-Net ayarlarını kullanır: `patch_size=1000`, `surface_points=100000`, `epochs=220`, `min_epochs=80`, `patience=35`, `batch_size=2`. Checkpoint seçimi validation snapped ALE ile yapılır; outer-test patch'leri checkpoint kilitlenmeden hazırlanmaz.

## Google Colab GPU

Colab icin hazir dosyalar:

- `colab_palnet_orthodontic_gpu.ipynb`
- `COLAB_TR.md`

## Çıktılar

Varsayılan çıktı klasörü:

```text
palnet_orthodontic_comparison/runs/orthodontic_palnet/
```

Önemli dosyalar:

- `metrics.json`: PAL-Net ve uzman doktor karşılaştırması için ALE özeti.
- `metrics_refined.json`: `--train-refiner` aciksa residual refiner ALE, PCK, class/gender ve zor landmark ozetleri.
- `predictions_test.csv`: Her test hastası ve her landmark için uzman koordinatı, PAL-Net koordinatı ve lokalizasyon hatası.
- `stage1_predictions_val.csv`, `stage1_predictions_test.csv`: Refiner oncesi Stage 1 tahminleri.
- `refined_predictions_test.csv`: Residual refiner sonrasi ana test tahminleri.
- `landmark_weights.json`: Validation hatasindan turetilen landmark agirliklari.
- `group_metrics_test.csv`: Class/cinsiyet bazlı ALE.
- `landmark_metrics_test.csv`: Landmark bazli mean, median, std, max ve PCK degerleri.
- `clinical_thresholds_test.csv`: PCK@2mm, PCK@2.5mm ve PCK@3mm klinik esik analizi.
- `class_metrics_test.csv`: Class I / II / III bazli performans.
- `gender_metrics_test.csv`: Female / male bazli performans.
- `difficult_landmarks_test.csv`: En zor landmark siralamasi.
- `splits.json`: Train/validation/test ayrımı ve landmark dosyası eksik olan meshler.
- `best_model.pth`: 5-fold protokolünde en iyi validation snapped ALE'ye sahip PAL-Net ağırlıkları.
- `last_model.pth`: Kesilen fold eğitimini aynı komutla sürdüren optimizer/scheduler/RNG checkpoint'i.
- `metrics_val.json`, `predictions_val.csv`: Checkpoint seçimi sonrasındaki doğrulama çıktıları.
- `split_and_leakage_report.json`: Split çakışması, alignment kaynağı ve test-sealing denetimi.

Tam 5-fold koşunun kök klasöründe ayrıca `summary_metrics.json`, `summary_fold_metrics.csv`, `summary_landmark_metrics.csv`, `summary_class_metrics.csv`, `summary_gender_metrics.csv` ve `pooled_predictions_test.csv` üretilir. Ana PAL-Net sonucu `summary_metrics.json > pooled > ale` değeridir.
