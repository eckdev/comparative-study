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
- Donmuş vNext + shape-prior çıktısı üzerinde son aşama, düşük kapasiteli bir Hard3
  structured ranker:
  - `LM0` için illumination-normalized RGB, mesh texture contrast ve yüz-kanonik geometri,
  - `LM21/LM22` için karşı tarafın kaba Gonion tahminini kullanmayan ortak bilateral candidate-pair head,
  - yalnız outer-train içinde nested OOF eğitim ve inner-fold model ensemble,
  - inner-fold ensemble anlaşmazlığından örnek bazlı güven,
  - `LM0=12 mm`, Gonion=`15 mm` düzeltme sınırı ve validation'da grup bazlı güvenli blend.
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
checkpoint imzaları eşleşiyorsa yeniden eğitilmez; yalnız `hard3_structured/` altındaki küçük
OOF ranker eğitilir. Aynı komut ikinci kez çalıştırıldığında ranker da cache'den yüklenir.

Beş-fold preprocessing kontrolü:

```python
!python -u colab_run_aghformer_vnext.py --preset cv_preflight --seed 42
```

Fold 1 kabul kapısını geçerse yayın koşusu:

```python
!python -u colab_run_aghformer_vnext.py --preset cv --seed 42
```

Kesilen koşu aynı komutla yeniden başlatılabilir. `last_model.pth`, `best_model.pth` ve
Stage 1 cache imzaları uyuşuyorsa tamamlanan epochlar yeniden eğitilmez. Belirli foldlar:

```python
!python -u colab_run_aghformer_vnext.py --preset cv --seed 42 --fold-indices 3,4,5
```

## Kabul kapısı

Pahalı beş-fold koşudan önce Fold 1 validation sonucu aynı fold baseline'ına göre:

- overall ALE en az `0.15 mm` düşmeli,
- Core20 regresyonu `0.03 mm`yi aşmamalı,
- Hard3 en az `0.75 mm` düşmeli,
- p95 kötüleşmemeli.

Bu kapı geçmezse model büyütülmemeli; fusion alpha, candidate ranking ve shape-prior
raporları incelenmelidir.

Hard3 structured aşaması ayrıca validation üzerinde en az `0.15 mm` Hard3 ve `0.03 mm`
overall kazanç, en az `0.90` bootstrap iyileşme olasılığı göstermediğinde veya p95 değeri
`0.10 mm`den fazla kötüleştiğinde otomatik olarak `alpha=0` seçer. Bu durumda ana tahminler
değişmez. Bu aşama için pahalı beş-fold kararı verilmeden önce hedef, Fold 1 Hard3 değerini
`5.2565 mm`den en az `4.5 mm` altına indirmek ve overall ALE'yi `2.25 mm` altına taşımaktır.
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
fold_*/hard3_structured/hard3_structured_model.pth
fold_*/hard3_structured/hard3_training_report.json
fold_*/hard3_structured/hard3_blend_selection.json
fold_*/hard3_structured/metrics_val.json
fold_*/hard3_stage3_decision.json
fold_*/split_and_leakage_report.json
summary_fold_metrics.csv
summary_metrics.json
```

`neural_only/` shape-prior öncesi AGH vNext sonucunu, `shape_prior_only/` mevcut
`2.2818 mm` hattına karşılık gelen Stage 3 öncesi sonucu saklar. `hard3_structured/` ve ana
fold dosyaları validation'da kilitlenen structured ranker dahil nihai sonucu içerir.
