# All-23 Anatomy-Guided Cascade

Bu klasör, 23 yumuşak doku landmarkını tek nihai çıktı olarak üreten deneysel cascade hattını içerir. Ana amaç yalnız kolay landmarkları raporlamak değil, mevcut en güçlü all-23 tahminini korurken Trichion (`LM0`) ve bilateral Gonion (`LM21`, `LM22`) için yeni yüzey kanıtı üretmektir.

## Model akışı

```text
AGH-Former Stage2 (23 landmark)
  -> all-23 shape-prior/stacker kalibrasyonu
  -> LM0/21/22 full-surface heatmap branch
  -> validation-selected confidence fusion
  -> tek bir final 23-landmark tahmini
```

Hard3 dalı, önceki tahminleri yalnız yeniden birleştirmez. AGH koşusunun gerçek yüzey noktalarını, normalleri, yoğunluk/eğrilik özelliklerini, 23 başlangıç landmarkına uzaklıkları, class ve gender bilgisini kullanarak yüzey adaylarına olasılık verir. LM0 için daha geniş, Gonion noktaları için landmark-specific Gaussian hedef kullanılır. Eğitimde hard3 başlangıç koordinatlarına jitter uygulanarak train tahminleri ile gerçek inference tahminleri arasındaki dağılım farkı azaltılır.

Model ve postprocess seçimi yalnız validation setinde yapılır. Test uzman koordinatları, checkpoint ile `top-k`, temperature ve fusion ayarları kilitlendikten sonra değerlendirmeye alınır. `blend=0` adayının sweep içinde bulunması, yeni dal validation üzerinde yararlı değilse mevcut all-23 sonucu otomatik olarak korur.

## Colab hazırlığı

Google Drive altında beklenen temel dosyalar:

```text
/content/drive/MyDrive/orthodontic/
  data/dataset/
  transforms/orthodontic_procrustes_rigid_20260627_143801/
  diffusion_runs/
    aghformer_v6_stage2_raw_fine_refiner_p12000/
    shape_prior_stacker/
```

`shape_prior_stacker` yerine daha iyi bir all-23 tahmin klasörü varsa `--initial-prediction-dir` ile verilebilir. Klasörde `predictions_val.csv` ve `predictions_test.csv` bulunmalıdır.

## Smoke test

Colab hücresinde:

```python
%cd /content/comparative-study/all23_anatomy_guided_cascade
!python -u colab_run_all23_cascade.py --preset smoke
```

Smoke test yalnız kod ve veri akışını doğrular; bilimsel performans sonucu değildir.

## A100 ana koşu

```python
%cd /content/comparative-study/all23_anatomy_guided_cascade
!python -u colab_run_all23_cascade.py --preset a100
```

Farklı bir stacker klasörü kullanmak için:

```python
%cd /content/comparative-study/all23_anatomy_guided_cascade
!python -u colab_run_all23_cascade.py \
    --preset a100 \
    --initial-prediction-dir /content/drive/MyDrive/orthodontic/diffusion_runs/shape_prior_stacker
```

Runner, `refined_predictions_train.csv` eksikse mevcut Stage2 checkpoint üzerinden bir kez üretir. Bu işlem ilk çalıştırmada zaman alabilir.

## Temel çıktılar

```text
best_hard3_surface_refiner.pth
history.json
postprocess_sweep_val.csv
predictions_val.csv
predictions_test.csv
landmark_metrics_val.csv
landmark_metrics_test.csv
group_metrics_test.csv
metrics_all23_cascade.json
config_all23_cascade.json
```

`metrics_all23_cascade.json` içinde şunlar birlikte raporlanır:

- all-23 ALE, median, standart sapma ve percentile değerleri,
- SDR@2/2.5/3/4 mm,
- Core20 ve Hard3 tanısal metrikleri,
- LM0/21/22 aday yüzey oracle ALE,
- bootstrap %95 güven aralığı,
- seçilen top-k, temperature ve confidence-fusion politikası.

## Sonucu yorumlama

- `candidate_oracle_test_hard3` düşük, `final_test_hard3` yüksekse sorun aday noktaların yokluğu değil, heatmap sıralamasıdır.
- Oracle da yüksekse `candidate-points` veya lokal/global aday oranı yetersizdir; önce yüzey kapsaması düzeltilmelidir.
- Seçilen `blend=0` ise hard3 dalı validation üzerinde güvenilir kazanç üretmemiştir ve final çıktı başlangıç all-23 tahminini korur.
- Test ALE tek başına model seçmek için kullanılmamalıdır; tüm seçim `postprocess_sweep_val.csv` üzerinden validation ile yapılır.
