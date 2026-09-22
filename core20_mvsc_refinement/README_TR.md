# Core20-MVSC Stage 4

Bu modül AGH-Former vNext'in `LM0`, `LM21` ve `LM22` çıktısını değiştirmeden
`LM1-LM20` için aday seçimini yeniden öğrenir. MVSC, *Multi-View Spatial
Configuration* ifadesinin kısaltmasıdır.

## Mimari

- Her landmark için mevcut 1024 noktalı dinamik geodezik ROI kullanılır.
- Frontal, lokal tangent ve sagittal/taraf-uyumlu oblik olmak üzere üç adet
  `96x96` görünüm oluşturulur.
- Raster kanalları RGB, depth, normal, üç ölçekli normal variation/curvature,
  üç ölçekli RGB gradient, görüntü koordinatı ve occupancy bilgisini içerir.
- GroupNorm tabanlı küçük FPN, görüntü özelliklerini ROI vertexlerine geri örnekler.
- Altı anatomik landmark grubu ayrı aday head'leri kullanır.
- Her landmark, diğer 19 Core20 noktadan tahmin edilen train-only conditional
  Gaussian prior ile puanlanır. Ridge katsayısı inner OOF üzerinde seçilir;
  eğitimde de uzman bağlamı yerine held-out ridge modeli ile jitter uygulanmış
  OOF tahmin bağlamı kullanılır.
- Top-8 soft expectation sürekli koordinat üretir. Entropy, peak margin ve
  lokal/prior uyuşmazlığı confidence ve base/proposal gate girdileridir.

Train merkezleri yüzde 50 Stage1-OOF residual, yüzde 35 `N(0,1.5 mm)` ve yüzde
15 `N(0,3 mm)` karışımıyla üretilir; düzeltme normu 6 mm ile sınırlıdır. Outer
validation yalnız altı grup için blend politikasını ve pahalı CV kabul kapısını
kilitler. Test etiketi bu seçimlerde kullanılmaz.

## Colab sırası

```python
%cd /content/comparative-study/agh_former_vnext_orthodontic_comparison
!python -u colab_run_aghformer_vnext.py --preset core20_preflight --seed 42
!python -u colab_run_aghformer_vnext.py --preset core20_fold1 --seed 42
```

`core20_stage4_decision.json` içinde `run_full_cv=true` oluşursa:

```python
!python -u colab_run_aghformer_vnext.py --preset core20_cv --seed 42
```

`core20_cv`, Fold 1'de Core20 `<=1.70 mm`, bootstrap iyileşme olasılığı
`>=0.95` ve hiçbir anatomik grupta `0.10 mm` üzerinde regresyon yoksa açılır.
Preset, Hard3 için en iyi gözlenen H3-RSCR V10 checkpoint sözleşmesini
(`crossfit_set_context`) kilitler.

## Çıktılar

```text
fold_*/core20_mvsc_v1/best_model.pth
fold_*/core20_mvsc_v1/last_model.pth
fold_*/core20_mvsc_v1/history.json
fold_*/core20_mvsc_v1/spatial_prior.json
fold_*/core20_mvsc_v1/training_report.json
fold_*/core20_mvsc_v1/predictions_val.csv
fold_*/core20_mvsc_v1/landmark_metrics_val.csv
fold_*/core20_mvsc_v1/anatomical_group_metrics_val.csv
fold_*/core20_mvsc_v1/anatomical_group_metrics_test.csv
fold_*/core20_mvsc_v1/core20_metrics_val.json
fold_*/core20_mvsc_v1/core20_metrics_test.json
fold_*/core20_mvsc_v1/core20_feasibility.json
fold_*/core20_mvsc_v1/core20_stage4_decision.json
```

`last_model.pth`, tamamlanmış son epoch ile optimizer, scheduler, AMP ve RNG
durumlarını saklar. Kesilen bir Fold 1 koşusu aynı komutla yeniden başlatıldığında
Stage 4 bu epoch'tan devam eder.

`C1-C4` ablationları sırasıyla çok ölçekli 3B head, RGB-D çoklu görünüm,
conditional spatial prior ve confidence gate bileşenlerini kademeli olarak açar.
