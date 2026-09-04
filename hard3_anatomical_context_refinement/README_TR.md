# Hard3 Anatomical Context Refinement

Bu deney, AGH-Former vNext'in güçlü Core20 tahminlerini değiştirmeden yalnız
`LM0=Trichion`, `LM21=Gonion left` ve `LM22=Gonion right` noktalarını yeniden
lokalize eder. Eski pointwise Hard3 ranker ile aynı çıktı klasörünü kullanmaz;
sonuçlar her fold altında `hard3_dual_view/` dizinine yazılır.

## Neden farklı bir model?

- Trichion saç çizgisi ile yüz orta hattının kesişimidir; RGB geçişi geometri kadar
  önemlidir.
- Gonion tek bir lokal tepe değildir. Posterior ve inferior alt-yüz konturlarının
  teğetleriyle tanımlanan bilateral bir kontur landmarkıdır.
- Bu nedenle model, her ROI'yi frontal ve profil yönlerinden 2.5D rastera çevirir.
  Yirmi kanal; RGB, lokal renk kontrastı, canonical normal, derinlik, eğrilik,
  yoğunluk, görünüm kimliği, CoordConv `u/v`, signed silhouette distance,
  depth-gradient ve occupancy bilgisini içerir.
- Trichion ve Gonion için ayrı küçük U-Net'ler kullanılır; iki Gonion aynı ağırlığı
  paylaşır ve pair-midpoint/width kaybıyla birlikte öğrenilir.
- Adaptive Wing'e ek olarak ICCV 2025 PossLoss kullanılır. Böylece heatmap peak
  kayması büyük olan zor örnekler eğitimde kendiliğinden daha yüksek ağırlık alır.
- Core20 konfigürasyonuna göre yalnız outer-train şekillerinden yerel atlas priorı
  üretilir. Neural heatmap, atlas ve surface candidate birleşimi yalnız validation
  fold'unda seçilir; testte politika değiştirilmez.

## Colab Fold 1 geliştirme koşusu

```python
%cd /content/comparative-study/agh_former_vnext_orthodontic_comparison
!python -u colab_run_aghformer_vnext.py --preset hard3_fold1 --seed 42
```

Bu preset varsayılan olarak `--hard3-refiner-mode dual_view` kullanır. Daha önce
tamamlanan vNext Stage 1/Stage 2 checkpointleri aynı run klasöründe ise yeniden
eğitilmez; yeni dosyalar `fold_1/hard3_dual_view/` altında oluşturulur.

Eski ranker'ı ablation olarak çalıştırmak için doğrudan ana script'e
`--hard3-refiner-mode structured` verilebilir.

## Karar kuralı

Fold 1 yalnız geliştirme kapısıdır. Beş fold'a ancak aşağıdaki koşullarla geçilir:

```text
validation Hard3 ALE < 4.00 mm
validation Core20 değişimi = 0.00 mm
overall ALE kazancı >= 0.03 mm
Hard3 kazancı >= 0.20 mm
bootstrap P(improved) >= 0.90
p95 regresyonu <= 0.10 mm
```

Ana dosyalar:

```text
hard3_dual_view/hard3_dual_view_model.pth
hard3_dual_view/hard3_dual_view_training_report.json
hard3_dual_view/hard3_blend_selection.json
hard3_dual_view/metrics_val.json
hard3_stage3_decision.json
validation_only_summary.json
```

`target_reached_on_validation=false` ise test açılmamalı ve tam 5-fold koşusuna
geçilmemelidir. Colab `cv` preset'i bu kararı otomatik okur ve başarısız kapıda
çalışmayı durdurur. Bu mekanizma 4 mm hedefini raporlama sonrasında değil, deneyden
önce tanımlanmış bir kabul eşiği olarak uygular.
