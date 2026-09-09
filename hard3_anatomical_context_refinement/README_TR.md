# Hard3 Anatomical Context Refinement (H3-DVAR v7)

Bu deney, AGH-Former vNext'in güçlü Core20 tahminlerini değiştirmeden yalnız
`LM0=Trichion`, `LM21=Gonion left` ve `LM22=Gonion right` noktalarını yeniden
lokalize eder. Eski pointwise Hard3 ranker ile aynı çıktı klasörünü kullanmaz;
sonuçlar her fold altında `hard3_dual_view_v7/` dizinine yazılır.

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

## Fold 1 denetimi ve v4-v7 geliştirmeleri

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

V3 Fold 1 sonucunda proposal@96 değerleri `LM21=%80.2`, `LM22=%80.7`, shortlist
oracle ise `1.225 mm` olmuştur. Buna rağmen seçilmiş Hard3 sonucu yalnız
`5.2643 -> 4.7045 mm` düzeyine inmiştir. İyi adayların mevcut olduğu, fakat
`96 x 96` ortak decoder'ın bunları doğru sıralayamadığı görülmüştür.

V4 seçim darboğazını proposal ve pair eğitimine ayırmıştır:

- Her adayın canonical konumu, `LM10/11/12` anchor geometrisi, normal, eğrilik,
  yoğunluk, RGB, lokal kontrast, intensity ve chroma özellikleri kullanılır.
- Geodezik ROI içindeki her aday için 12 komşulu sabit lokal yüzey grafı çıkarılır.
  EdgeConv-benzeri context encoder noktanın kendisini, komşu farklarını ve lokal
  koordinat değişimini birlikte işler.
- Proposal aşaması tüm 1024 aday üzerinde mesafe tabanlı listwise hedefle eğitilir;
  bilateral pair başlığı bu sırada dondurulur.
- En yüksek learned proposal skoruna sahip 24 sol ve 24 sağ aday seçilir. Proposal
  ağı dondurulduktan sonra pair ranker yalnız bu `24 x 24` dağılım üzerinde eğitilir.
- Pair aşamasında expert-nearest teacher forcing kullanılmaz; eğitim ve çıkarım aynı
  shortlist dağılımını görür.
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
- Final çıkarım inner-fold en iyi checkpointlerinden oluşan beş model ensemble'ıdır.

V4 Fold 1'de Hard3 ALE'yi `5.2593 -> 4.6402 mm`, overall ALE'yi `2.2810 ->
2.2002 mm` düşürmüştür. Ancak top-24 shortlist oracle ALE `2.455 mm`, p95
`7.400 mm` ve SDR@2 `%63.5` iken aynı proposal'ın top-96 değerleri sırasıyla
`1.097 mm`, `2.603 mm` ve `%92.4` olmuştur. Pair-soft decoder unary Gonion
tahminine yalnız `0.046 mm` kazandırmıştır. Sonuç, iyi adayın geniş listede mevcut
olduğunu; doğrudan `1024 -> 24` sıkıştırma ve geniş pair hedefinin sınırlayıcı
olduğunu göstermiştir.

V5 bu nedenle üç ayrı, leakage-safe eğitim aşaması kullanır:

- Broad proposal tüm 1024 aday üzerinde v4 yüzey bağlamını öğrenir ve en iyi 96
  adayı korur.
- Proposal dondurulur. Sharp unary reranker canonical/RGB/yüzey özellikleri ile
  dört proposal kaynağını örnek-bazlı set context içinde işler ve `96 -> 32`
  sıralaması üretir.
- Reranker hedefi `sigma=2 mm` soft-listwise loss, doğrudan beklenen mesafe,
  Smooth L1 koordinat ve ordinal hard-negative loss birleşimidir.
- Proposal ve reranker dondurulduktan sonra `32 x 32` bilateral pair decoder ayrı
  eğitilir. Pair hedefi de `sigma=2 mm` ve beklenen ALE terimi kullanır.
- Reranker düzeltmesi sınırlıdır; shortlist dışındaki broad logit varyasyonu
  korunur. Böylece beş model ensemble'ında kalibrasyon bozulmaz.
- Expert-nearest teacher forcing hiçbir aşamada kullanılmaz.
- Proposal, reranker ve pair için inner-fold en iyi epochlar ayrı saklanır.

V5 OOF sonucu, sorunun bağımsız daraltma olduğunu doğrulamıştır. Broad top-96
Gonion oracle ALE `1.0909 mm`, p95 `2.7098 mm` ve SDR@2 `%93.0` iken sharp
top-32 çıktısı sırasıyla `2.0888 mm`, `6.6509 mm` ve `%70.6` olmuştur. Dahası,
broad top-32 oracle `2.0923 mm` olduğundan reranker ölçülebilir bir sıralama
kazancı üretmemiştir. En iyi reranker epochlarının `27/5/1/3/6` olması küçük veri
üzerinde kararsızlığı, pair-soft sonucunun unary sonuçtan kötü olması da ikinci
decoder'ın yararlı sinyali kullanamadığını göstermiştir.

V6 bu nedenle `96 -> 32` hard pruning aşamasını kaldırır:

- Broad proposal dondurulduktan sonra sol ve sağ top-96 listeler korunur.
- Hafif candidate encoder ve bilateral cross-attention, her tarafı karşı tarafın
  bütün aday dağılımıyla koşullandırır.
- Mirrored canonical geometri zayıf, öğrenilebilir bir simetri priorı sağlar.
- `96 x 96 = 9216` çiftin tamamı tek ortak dağılım içinde skorlanır.
- Pair loss; `sigma=1.5 mm` listwise hedef, unilateral listwise denetim,
  `<=2 mm` klinik pozitif kütle, beklenen çift mesafesi ve `>4 mm` yüksek skorlu
  hard-negative margin terimlerinden oluşur.
- Pair checkpoint seçimi değişmeyen Trichion yerine doğrudan inner-validation
  Gonion ALE üzerinden yapılır.
- Eğitim ve çıkarım aynı broad top-96 aday uzayını görür; expert-nearest aday
  enjeksiyonu yapılmaz.

V6 Fold 1'de broad top-96 oracle `1.089 mm` olmasına karşın Hard3 sonucu
`4.6994 mm`de kalmıştır. Kod denetimi iki temel neden göstermiştir: eski çift hedefi
matematiksel olarak iki bağımsız landmark dağılımına ayrışmakta ve Hard3 eğitim
ROI'leri Stage 1 merkezlerinden, validation ROI'leri ise Stage 2 + shape-prior
merkezlerinden üretilmekteydi.

V7 bu seçim ve dağılım uyuşmazlığını hedefler:

- Outer-train örneklerinde donmuş Stage 2, gate ve shape-prior cascade'i yeniden
  çalıştırılır; Hard3 eğitim merkezleri bu çıktılardan alınır.
- Geodezik/hybrid ROI cache'inde hem koordinat çerçevesi hem aday vertex havuzu bu
  merkezler çevresinde yeniden kurulur. Validation/test aynı yolu kullanır.
- Adaylara iki görünümden signed silhouette distance, depth-gradient ve occupancy
  örnekleri eklenir.
- Tüm 23 cascade tahmini canonical shape context olarak bilateral ranker'a verilir.
- Ranker, Gonion çiftini ortak `mean + asymmetry` contour state'i olarak tahmin eder;
  bu state doğrudan uzman koordinatlarıyla denetlenir.
- Mean ve asymmetry enerjileri farklı öğrenilebilir ölçekler kullanır. Böylece joint
  skor, V6'daki gibi iki bağımsız unary skora cebirsel olarak ayrılamaz.
- OOF raporu selector regret ve `x/y/z` hata bileşenlerini ayrıca kaydeder.

## Colab Fold 1 geliştirme koşusu

```python
%cd /content/comparative-study/agh_former_vnext_orthodontic_comparison
!python -u colab_run_aghformer_vnext.py --preset hard3_fold1 --seed 42
```

Bu preset varsayılan olarak `--hard3-refiner-mode dual_view` kullanır. Daha önce
tamamlanan vNext Stage 1/Stage 2 checkpointleri aynı run klasöründe ise yeniden
eğitilmez. Yalnız `fold_1/hard3_dual_view_v7/` yeniden eğitilir; eski sürüm
çıktıları değiştirilmez.

Yeni eğitim raporunda aşağıdaki tanılar ayrıca bulunur:

```text
oof.proposal_diagnostics.at_k.32/48/96
oof.proposal_diagnostics.diverse_rank_union_at_k.32/48/96
oof.gonion_pair_topk_recall.lm21/lm22/both
oof.gonion_pair_topk_recall.oracle_ale/oracle_p95/oracle_sdr_at_2mm
oof.gonion_pair_topk_recall.clinical_both_coverage
oof.selection_diagnostics.selector_regret_mm
oof.selection_diagnostics.axis_mae_xyz
oof.mean_dynamic_view_weights
oof.std_dynamic_view_weights
coordinate_policy.gonion_pair
candidate_metrics.joint_soft/joint_argmax/joint_snapped
selected.alpha_gonion_left/right
```

`proposal_diagnostics` broad aşamayı, `gonion_pair_topk_recall` ise V7'de doğrudan
contour-pair arama uzayını ölçer. Exact-nearest vertex recall yalnız tanısaldır. Tam
CV için arama uzayının Gonion oracle ALE değeri `<=1.50 mm`, p95 değeri
`<=3.50 mm` ve SDR@2mm değeri `>=%75` olmalıdır. Bu ölçüler yoğun mesh üzerinde
komşu iki vertex arasındaki önemsiz indeks değişimlerinden etkilenmez.

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
OOF full-pair search Gonion oracle ALE <= 1.50 mm
OOF full-pair search Gonion oracle p95 <= 3.50 mm
OOF full-pair search Gonion oracle SDR@2mm >= %75
```

Ana dosyalar:

```text
hard3_dual_view_v7/hard3_dual_view_model.pth
hard3_dual_view_v7/hard3_dual_view_training_report.json
hard3_dual_view_v7/hard3_blend_selection.json
hard3_dual_view_v7/metrics_val.json
hard3_stage3_decision.json
validation_only_summary.json
```

`target_reached_on_validation=false` ise test açılmamalı ve tam 5-fold koşusuna
geçilmemelidir. Colab `cv` preset'i bu kararı otomatik okur ve başarısız kapıda
çalışmayı durdurur. Bu mekanizma 4 mm hedefini raporlama sonrasında değil, deneyden
önce tanımlanmış bir kabul eşiği olarak uygular.
