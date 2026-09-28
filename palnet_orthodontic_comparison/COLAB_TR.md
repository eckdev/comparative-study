# PAL-Net Google Colab Pro Kullanimi

Bu dosyalar PAL-Net adaptasyonunu Google Colab GPU uzerinde calistirmak icin hazirlandi. DiffusionNet ve PointNet++ ile adil karsilastirma icin ortak split dosyasi kullanilir.

## Makale Icin 5-Fold Protokol

Ana akademik koşu `colab_run_palnet_cv.py` ile yapılır. Bu çalıştırıcı:

- aynı `orthodontic_5fold_192_48_60_seed42.json` manifestini kullanır,
- her fold için aynı `192/48/60` örneklerini seçer,
- yalnız train meshleriyle fit edilmiş label-free rigid ICP dönüşümlerini doğrular,
- validation snapped ALE ile checkpoint seçer,
- checkpoint kilitlenmeden outer-test patch'lerini veya etiketlerini yüklemez,
- yarım kalan fold'u `last_model.pth` üzerinden sürdürür,
- beş fold bittiğinde pooled ALE, Core20, Hard3, SDR ve bootstrap güven aralığını toplar.

Colab hücreleri:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/comparative-study/palnet_orthodontic_comparison
!python -u colab_run_palnet_cv.py --preset preflight --seed 42
```

Beş fold preflight başarılı olduktan sonra kısa pipeline testi:

```python
%cd /content/comparative-study/palnet_orthodontic_comparison
!python -u colab_run_palnet_cv.py --preset smoke --seed 42
```

A100 ana koşusu:

```python
%cd /content/comparative-study/palnet_orthodontic_comparison
!python -u colab_run_palnet_cv.py --preset cv --seed 42
```

Colab oturumu kesilirse aynı `cv` hücresini yeniden çalıştır. Tamamlanmış fold'lar atlanır; yarım fold son `last_model.pth` checkpoint'inden devam eder. Belirli fold'lar da seçilebilir:

```python
!python -u colab_run_palnet_cv.py --preset cv --seed 42 --fold-indices 3,4,5
```

Varsayılan çıktı:

```text
/content/drive/MyDrive/orthodontic/palnet_runs/palnet_publication_cv_seed42/
```

Bu protokol Stage 1 PAL-Net baseline'ını ölçer. Eski residual refiner, frozen CV protokolüne dahil değildir; `--require-label-free-alignment` ile birlikte bilerek engellenir.

## Dosyalar

- `upstream/run_orthodontic.py`: Yerel/Colab ortak egitim scripti.
- `colab_palnet_orthodontic_gpu.ipynb`: Colab uzerinde hucre hucre calistirilacak notebook.
- `colab_palnet_5fold.ipynb`: Ortak leakage-free 5-fold makale koşusu notebook'u.
- `colab_run_palnet_cv.py`: Preflight, smoke, resume ve fold aggregation çalıştırıcısı.
- `requirements.txt`: Python bagimliliklari.

## Google Drive yapisi

Dataset ve transform klasorleri GitHub'a eklenmedigi icin Google Drive uzerinde tutulmalidir. Onerilen yapi:

```text
MyDrive/
  orthodontic/
    data/
      dataset/
        Class1/
        Class2/
        Class3/
    transforms/
      orthodontic_procrustes_rigid_20260627_143801/
    all23_rgb_geodesic_runs/
      publication_cv_stage1_v4_seed42/
        fold_1/
          alignment/
            mesh_only_transforms.npz
            alignment_report.json
          split_and_leakage_report.json
        fold_2/
        fold_3/
        fold_4/
        fold_5/
    palnet_runs/
```

`dataset` klasoru yereldeki `data/dataset` ile ayni formatta olmalidir. Transform klasoru yoksa notebook'ta `USE_TRANSFORMS = False` yapilabilir; fakat model karsilastirmasinda ayni hizalanmis veri protokolu icin transform kullanilmasi onerilir.

## Ortak Split

Notebook ve komutlar repo icindeki ortak split dosyasini kullanir:

```bash
--splits-json /content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json
```

Bu split 300 hastayi sinif/cinsiyet dengeli olarak 180 egitim, 60 validasyon ve 60 test hastasina ayirir.

Bu `180/60/60` bölüm yalnız eski sabit-split deneyleri içindir. Makale 5-fold koşusu `orthodontic_5fold_192_48_60_seed42.json` kullanır ve her örneği tam bir kez outer-test olarak değerlendirir.

## Smoke Test

Once kodun ve veri yollarinin dogru calistigini gormek icin kisa smoke test calistir. Bu kosu sadece pipeline kontroludur; ALE sonucu bilimsel raporlamada kullanilmaz. Loglar `-u` ve script icindeki `flush=True` ciktilariyla Colab ekranina anlik akar.

```bash
python -u run_orthodontic.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --splits-json /content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json \
  --transformation-dir /content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir /content/drive/MyDrive/orthodontic/palnet_runs/palnet_smoke_fast_colab \
  --epochs 1 \
  --patience 1 \
  --batch-size 8 \
  --patch-size 100 \
  --surface-points 1024 \
  --snap-k 1 \
  --max-train-samples 12 \
  --max-val-samples 6 \
  --max-test-samples 6
```

Smoke testte ekranda sirasiyla `Paired samples`, `Using samples`, `cache train/val/test`, `Device` ve `Epoch` loglarini gormen gerekir. Colab hucre ciktisi yine gecikirse notebook'taki komut hucrelerinde `PYTHONUNBUFFERED=1` ortami zaten ayarlanmistir.

## Colab Pro Pratik Kosu

T4/L4 gibi GPU'larda daha uygulanabilir PAL-Net kosusu:

```bash
python -u run_orthodontic.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --splits-json /content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json \
  --transformation-dir /content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir /content/drive/MyDrive/orthodontic/palnet_runs/palnet_procrustes_p500_surface50k_e120 \
  --epochs 120 \
  --patience 30 \
  --batch-size 4 \
  --patch-size 500 \
  --surface-points 50000 \
  --lr 0.001 \
  --snap-k 1 \
  --model PALNET \
  --loss combined
```

## A100 / Yuksek VRAM Buyuk Kosu

Paper'a daha yakin yogun ayar ve residual refiner icin:

```bash
python -u run_orthodontic.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --splits-json /content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json \
  --transformation-dir /content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir /content/drive/MyDrive/orthodontic/palnet_runs/palnet_refiner_p1000_surface100k_e200 \
  --epochs 200 \
  --patience 40 \
  --batch-size 2 \
  --patch-size 1000 \
  --refiner-patch-size 500 \
  --surface-points 100000 \
  --lr 0.001 \
  --snap-k 1 \
  --model PALNET \
  --loss combined \
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

`--patch-size` Stage 1 PAL-Net mimarisidir; var olan `p1000` checkpoint yuklenecekse 1000 kalmalidir. Stage 2 lokal refiner boyutu icin `--refiner-patch-size` kullanilir.

Bellek hatasi alirsan sirasiyla `--batch-size 1`, `--surface-points 50000`, `--refiner-patch-size 500` deneyebilirsin.

Tek refiner 2 mm ustunde kalirsa ayni hucreyi `--seed 43`, `--seed 44`, `--seed 45` ve farkli `--output-dir` ile calistirip ensemble al:

```bash
python -u ensemble_palnet_predictions.py \
  --predictions /content/drive/MyDrive/orthodontic/palnet_runs/run_seed43/refined_predictions_test.csv /content/drive/MyDrive/orthodontic/palnet_runs/run_seed44/refined_predictions_test.csv /content/drive/MyDrive/orthodontic/palnet_runs/run_seed45/refined_predictions_test.csv \
  --output-dir /content/drive/MyDrive/orthodontic/palnet_runs/palnet_refiner_ensemble
```

## Beklenen Ciktilar

Her run klasorunde:

- `metrics.json`: `palnet_raw`, `palnet_snapped` ve baseline ALE ozetleri.
- `metrics_refined.json`: residual refiner aciksa ana iyilestirilmis ALE/PCK ozetleri.
- `history.json`: epoch bazli train/validation loss.
- `refiner_history.json`: residual refiner train/validation loss ve validation ALE.
- `predictions_test.csv`: uzman ve PAL-Net tahmin koordinatlari.
- `stage1_predictions_val.csv`, `stage1_predictions_test.csv`, `refined_predictions_test.csv`: iki asamali tahmin dosyalari.
- `landmark_weights.json`: zor landmark agirliklari.
- `group_metrics_test.csv`: class/cinsiyet bazli ALE.
- `splits.json`: ortak split kaynagini gosteren run-local split kaydi.
- `best_model.pth`: en iyi validation loss checkpoint.
