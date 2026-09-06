# Hard3 Anatomical Context Refinement (H3-DVAR v3)

Bu deney, AGH-Former vNext'in güçlü Core20 tahminlerini değiştirmeden yalnız
`LM0=Trichion`, `LM21=Gonion left` ve `LM22=Gonion right` noktalarını yeniden
lokalize eder. Eski pointwise Hard3 ranker ile aynı çıktı klasörünü kullanmaz;
sonuçlar her fold altında `hard3_dual_view_v3/` dizinine yazılır.

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

## Fold 1 denetimi ve v3 değişikliği

İlk dual-view koşusu Hard3 ALE'yi `5.2358 -> 4.6143 mm` düşürmüştür. Landmark
bazında `LM0=2.9303`, `LM21=5.6220`, `LM22=5.2905 mm` ölçülmüştür. Buna göre
Trichion ana darboğaz olmaktan çıkmış, bilateral Gonion seçimi sınırlayıcı hale
gelmiştir. V1'de iki Gonion birbirinden bağımsız decode ediliyor, pair loss yalnız
son koordinatlara uygulanıyor ve frontal/profil ağırlıkları tüm örnekler için sabit
kalıyordu.

V2 ortak bilateral sıralamayı eklemesine rağmen OOF top-32 proposal recall değeri
`LM21=%57.3`, `LM22=%57.8` düzeyinde kalmıştır. Joint pair başlığının Gonion kazancı
yalnız `0.024 mm` olmuştur. Ayrıca inner-fold en iyi epoch medyanı `10` iken final
refit `min_epochs=30` nedeniyle 30 epocha zorlanmış ve bazı foldlarda belirgin biçimde
overfit olmuştur.

V3 bu ölçülmüş darboğazları doğrudan değiştirir:

- Her mesh adayı, mirrored canonical konumu ve leakage-free `LM10/11/12` alt-orta
  hat anchor geometrisiyle temsil edilir.
- İki görünümün heatmap skoru ile canonical/LM10-LM12 anchor geometrisi, normal,
  eğrilik ve yoğunluğu birleştiren 26 boyutlu shared 3B proposal head, tüm ROI
  adaylarını top-k öncesinde puanlar.
- Learned geometry, fused heatmap, frontal ve profil kaynaklarının rank-union'ından
  96 proposal seçilir. Bunlar `96 x 96` ortak bilateral dağılımda puanlanır.
- Pair ranker mesafe tabanlı soft-listwise hedef ve pair hard-negative mining ile
  eğitilir. Uzman-nearest teacher forcing ilk beş epochta doğrusal olarak sıfıra iner;
  sonraki eğitim ile validation/test aynı proposal dağılımını kullanır.
- Frontal/profil füzyonu heatmap kalitesi ve U-Net bağlamına göre örnek bazında
  değişir.
- Gonion eğitiminde RGB/kontrast kanalları rastgele düşürülerek asimetrik ışık ve
  gölge kestirmelerine bağımlılık azaltılır; Trichion'un RGB yolu korunur.
- Validation blend katsayıları LM21 ve LM22 için ayrı seçilebilir. Core20 yine
  değiştirilemez.
- Doğrudan atlas Gonion adayı seçimden çıkarılmıştır; atlas yalnız zayıf bir logit
  düzenleyicisi olarak denenebilir.
- Aynı raster pikseline düşen ön/arka yüzeyler artık ortalanmaz; dış yüzeyi koruyan
  z-buffer görünürlüğü kullanılır.
- Final çıkarım varsayılan olarak beş inner-fold best checkpoint ensemble'ıdır.
  Alternatif full-train refit gerçek medyan best epocha kadar eğitilir.

## Colab Fold 1 geliştirme koşusu

```python
%cd /content/comparative-study/agh_former_vnext_orthodontic_comparison
!python -u colab_run_aghformer_vnext.py --preset hard3_fold1 --seed 42
```

Bu preset varsayılan olarak `--hard3-refiner-mode dual_view` kullanır. Daha önce
tamamlanan vNext Stage 1/Stage 2 checkpointleri aynı run klasöründe ise yeniden
eğitilmez. Yalnız `fold_1/hard3_dual_view_v3/` yeniden eğitilir; v1/v2 çıktıları
değiştirilmez.

Yeni eğitim raporunda aşağıdaki tanılar ayrıca bulunur:

```text
oof.proposal_diagnostics.at_k.16/32/64/96
oof.gonion_pair_topk_recall.lm21/lm22/both
oof.mean_dynamic_view_weights
oof.std_dynamic_view_weights
coordinate_policy.gonion_pair
candidate_metrics.joint_soft/joint_argmax/joint_snapped
selected.alpha_gonion_left/right
```

Tam CV için LM21 ve LM22 `proposal@96 recall >= %90` ve shortlist Gonion oracle
`<=1.50 mm` olmalıdır. Recall yüksek olduğu halde Hard3 hedefe ulaşmazsa darboğazın
proposal değil ortak aday sıralaması olduğu ölçülebilir biçimde gösterilmiş olur.

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
OOF proposal@96 recall (LM21 ve LM22) >= %90
OOF shortlist Gonion oracle ALE <= 1.50 mm
```

Ana dosyalar:

```text
hard3_dual_view_v3/hard3_dual_view_model.pth
hard3_dual_view_v3/hard3_dual_view_training_report.json
hard3_dual_view_v3/hard3_blend_selection.json
hard3_dual_view_v3/metrics_val.json
hard3_stage3_decision.json
validation_only_summary.json
```

`target_reached_on_validation=false` ise test açılmamalı ve tam 5-fold koşusuna
geçilmemelidir. Colab `cv` preset'i bu kararı otomatik okur ve başarısız kapıda
çalışmayı durdurur. Bu mekanizma 4 mm hedefini raporlama sonrasında değil, deneyden
önce tanımlanmış bir kabul eşiği olarak uygular.
