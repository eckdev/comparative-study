# AGH-Former vNext

Bu klasör, önceki AGH-Former sonucunu yayın açısından güvenli bir protokole taşıyan ve modelin
tespit edilen mimari darboğazlarını düzelten ayrı deney hattıdır. Eski AGH çıktıları ve
checkpoint'leri değiştirilmez.

## Uygulanan geliştirmeler

- Doğrulanmış anatomi: orta hat `LM0-LM12`; simetri çiftleri `(13,16)`, `(14,15)`,
  `(17,18)`, `(19,20)`, `(21,22)`.
- Label-free hizalama: outer-train meshlerinden çoklu atlas ve ölçeği koruyan mesh-ICP.
- Train-only normalizasyon ve outer-train örnekleri için nested OOF Stage 1 tahminleri.
- Gerçek mesh adjacency üzerinde sparse Point Transformer yüzey encoder'ı.
- Landmark tokenlarının yüzey cross-attention ve anatomik graph attention ile güncellenmesi.
- Stage 1 merkezi, global heatmap ve sınırlı residual arasında landmark/sample-specific fusion.
- Dinamik geodezik ROI ve texture/contour/generic uzman refinement head'leri.
- LM0 için RGB/texture-gradient; LM21/22 için LM10-LM12 koşullu bilateral pair ranking.
- Refiner dondurulduktan sonra ayrı confidence gate eğitimi.
- Outer-train uzman şekillerinden fit edilen PCA + Core20-to-Hard3 conditional shape-prior.
  Shape-prior hiperparametreleri yalnız validation'da seçilir; test etiketi kullanılmaz.
- Donmuş vNext + shape-prior çıktısı üzerinde H3-DVAR v7:
  - `LM0` için frontal/profil RGB-depth appearance U-Net,
  - `LM21/LM22` için RGB, normal, curvature ve landmark-anchor özellikli lokal
    surface-context proposal ranker,
  - geodezik ROI içindeki 12-komşulu aday grafından `1024 -> 96` broad proposal,
  - broad top-96 adayları kaybetmeden iki tarafı koşullandıran cross-attention,
  - tüm `96 x 96` aday uzayını değerlendiren shape-conditioned contour-state
    bilateral ranker,
  - `sigma=1.5 mm` listwise hedef, SDR@2 pozitif kütle, expected-distance ve
    pair hard-negative loss,
  - eğitim ve çıkarımda aynı Stage 2 + shape-prior merkezlerinden yeniden kurulan
    dinamik geodezik ROI,
  - all-23 canonical shape context ve doğrudan mean/asymmetry state supervision,
  - nested OOF best-checkpoint ensemble ve inference ile aynı teachersız pair eğitimi,
  - görünür dış konturu koruyan z-buffer rasterizasyonu,
  - `LM0=12 mm`, Gonion=`15 mm` düzeltme sınırı ve validation-kilitli blend.
- Önceki pointwise ranker `--hard3-refiner-mode structured` ile ablation olarak korunur.
- TTA, confidence calibration, bootstrap CI, landmark/sınıf/cinsiyet sonuçları.

## Bilimsel protokol

Ana yayın sonucu patient-level aynı beş fold üzerinde raporlanmalıdır. Her outer fold:

```text
192 train / 48 validation / 60 test
```

Test landmarkları checkpoint, gate, shape-prior veya confidence seçimi tamamlanmadan
okunmaz. Sabit `180/60/60` split yalnız geliştirme ve diğer modellerle aynı-split ablation
için kullanılabilir.

## Yerel smoke test

```bash
python -u agh_former_vnext_orthodontic_comparison/run_aghformer_vnext.py \
  --data-root data/dataset \
  --output-dir /tmp/agh_vnext_smoke \
  --protocol fixed \
  --splits-json shared_splits/orthodontic_180_60_60_seed42.json \
  --coarse-source train_template \
  --train-center-mode template \
  --max-samples 24 \
  --icp-points 512 \
  --icp-iterations 3 \
  --atlas-size 2 \
  --atlas-iterations 1 \
  --registration-candidates 1 \
  --registration-restarts 1 \
  --roi-points 128 \
  --width 32 \
  --global-blocks 1 \
  --token-blocks 1 \
  --token-surface-points 512 \
  --epochs 2 \
  --min-epochs 1 \
  --patience 1 \
  --gate-stage-epochs 1 \
  --gate-stage-min-epochs 1 \
  --gate-stage-patience 1 \
  --skip-oracle-gate \
  --max-stage2-val-ale 200 \
  --no-shape-prior \
  --hard3-refiner-mode dual_view \
  --hard3-dual-view-folds 2 \
  --hard3-dual-view-epochs 2 \
  --hard3-dual-view-min-epochs 1 \
  --hard3-dual-view-patience 1 \
  --hard3-dual-view-image-size 32 \
  --hard3-dual-view-width 8 \
  --hard3-dual-view-decoder-mode contour_coordinate \
  --hard3-dual-view-proposal-topk 16 \
  --hard3-dual-view-pair-topk 16 \
  --hard3-dual-view-proposal-neighbors 4 \
  --hard3-dual-view-pair-stage-epochs 1 \
  --hard3-dual-view-pair-stage-min-epochs 1 \
  --hard3-dual-view-pair-stage-patience 1 \
  --hard3-dual-view-final-members 1 \
  --hard3-dual-view-final-policy median_best_refit \
  --no-tta \
  --no-tta-validation \
  --device cpu
```

## Google Colab Pro

Notebook: `colab_aghformer_vnext_tr.ipynb`

Kısa smoke:

```python
%cd /content/comparative-study/agh_former_vnext_orthodontic_comparison
!python -u colab_run_aghformer_vnext.py --preset smoke --seed 42
```

Önce yalnız Fold 1 ve validation:

```python
!python -u colab_run_aghformer_vnext.py --preset dev_fold1 --seed 42
```

Tamamlanmış Fold 1 checkpoint'ini değiştirmeden yalnız yeni Hard3 aşamasını denemek için:

```python
!python -u colab_run_aghformer_vnext.py --preset hard3_fold1 --seed 42
```

Bu komut aynı `publication_cv_seed42/fold_1` klasörünü kullanır. Stage 2 ve ayrı gate
checkpoint imzaları eşleşiyorsa yeniden eğitilmez; yalnız yeni
`hard3_dual_view_v7/` modeli eğitilir. Önceki H3-DVAR çıktıları korunur; aynı komut
tekrar çalıştırılırsa v7 model cache'den yüklenir.

Beş-fold preprocessing kontrolü:

```python
!python -u colab_run_aghformer_vnext.py --preset cv_preflight --seed 42
```

Fold 1 kabul kapısını geçerse yayın koşusu:

```python
!python -u colab_run_aghformer_vnext.py --preset cv --seed 42
```

`cv` preset'i önce aynı run klasöründeki
`fold_1/hard3_stage3_decision.json` dosyasını okur. `hard3_fold1` kapısı başarıyla
geçilmemişse pahalı beş-fold koşuyu bilinçli olarak durdurur; bu bir çalışma hatası
değil, test setini ve hesaplama bütçesini koruyan deney protokolüdür.

Kesilen koşu aynı komutla yeniden başlatılabilir. `last_model.pth`, `best_model.pth` ve
Stage 1 cache imzaları uyuşuyorsa tamamlanan epochlar yeniden eğitilmez. Belirli foldlar:

```python
!python -u colab_run_aghformer_vnext.py --preset cv --seed 42 --fold-indices 3,4,5
```

## Kabul kapısı

Pahalı beş-fold koşudan önce Fold 1 validation sonucu aynı fold baseline'ına göre tek
bir H3-DVAR kapısından geçmelidir:

- Hard3 ALE `<4.00 mm` ve overall ALE `<=2.25 mm` olmalı,
- Core20 koordinatları tam olarak değişmeden kalmalı,
- Hard3 kazancı en az `0.20 mm`, overall kazanç en az `0.03 mm` olmalı,
- bootstrap iyileşme olasılığı en az `0.90` olmalı,
- overall p95 değeri `0.10 mm`den fazla kötüleşmemeli.
- OOF full-pair Gonion oracle ALE en fazla `1.50 mm` olmalı.
- OOF full-pair Gonion oracle p95 en fazla `3.50 mm` olmalı,
- OOF full-pair Gonion SDR@2mm en az `%75` olmalı.

Kapı geçmezse fusion alpha otomatik olarak sıfırlanır ve tam CV başlatılmaz. Bu durumda
model büyütmek yerine `gonion_pair_topk_recall`, shortlist oracle kuyruğu,
örnek-bazlı görünüş ağırlıkları ve joint-pair aday sonuçları incelenmelidir. Fold 1
baseline Hard3 değeri `5.2358 mm`, v1 sonucu `4.6143 mm`, v2 sonucu `4.8017 mm`,
v3 sonucu `4.7045 mm`, v4 sonucu `4.6402 mm`dir.
V5 sonucu `4.7139 mm` olmuş; broad top-96 oracle `1.0909 mm` iken bağımsız
top-32 reranker oracle değeri `2.0888 mm`ye yükselmiştir. V6 bu nedenle hard
pruning uygulamaz ve aynı `<4.00 mm` kabul eşiğini kullanır.
V6'nın `4.6994 mm` sonucu ve `1.089 mm` broad oracle değeri, kalan problemin aday
bulmak değil seçmek olduğunu doğrulamıştır. V7, gerçek bilateral mean/asymmetry
state'i, all-23 shape context'i ve train/inference uyumlu cascade ROI'leriyle bu
darboğazı hedefler.
`hard3_stage3_decision.json` bu kapıları, mevcut Core20 sabitken 2 mm overall hedefi için
gereken Hard3 ALE bütçesini ve `run_full_cv` kararını otomatik hesaplar.

## Ana çıktılar

```text
fold_*/best_model.pth
fold_*/history.json
fold_*/metrics_val.json
fold_*/metrics_test.json
fold_*/landmark_metrics_*.csv
fold_*/group_metrics_*.csv
fold_*/predictions_*.csv
fold_*/shape_prior_selection.json
fold_*/shape_prior_only/metrics_val.json
fold_*/hard3_dual_view_v7/hard3_dual_view_model.pth
fold_*/hard3_dual_view_v7/hard3_dual_view_training_report.json
fold_*/hard3_dual_view_v7/hard3_blend_selection.json
fold_*/hard3_dual_view_v7/metrics_val.json
fold_*/hard3_stage3_decision.json
fold_*/split_and_leakage_report.json
summary_fold_metrics.csv
summary_metrics.json
```

`neural_only/` shape-prior öncesi AGH vNext sonucunu, `shape_prior_only/` mevcut
`2.2818 mm` hattına karşılık gelen Stage 3 öncesi sonucu saklar.
`hard3_dual_view_v7/` ve ana fold dosyaları validation'da kilitlenen H3-DVAR v7
dahil nihai sonucu içerir.
