# AGH-Former ile Ortodontik 3B Yumuşak Doku Landmark Lokalizasyonu

Bu klasör, mevcut `data/dataset` içindeki 300 ortodontik 3B yüz meshini ve 23 uzman landmarkını kullanarak özgün bir karşılaştırma modeli kurar: **Anatomy-Aware Geodesic Heatmap Transformer (AGH-Former)**.

## Yaklaşım

AGH-Former, PAL-Net, DiffusionNet ve PointNet++ modellerinden farklı olarak landmarkları bağımsız koordinat tahmini olarak değil, anatomik olarak ilişkili bir 23 noktalı konfigürasyon olarak ele alır.

- Girdi: hizalanmış yüzey noktaları.
- Özellikler: `XYZ`, yüzey normali, lokal yoğunluk ve lokal eğrilik proxy değeri.
- Global encoder: yüzey noktalarından nokta bazlı morfolojik temsil çıkarır.
- Landmark tokenları: 23 anatomik landmark için öğrenilebilir token kullanır.
- Template prior: yalnızca train split landmarklarından hesaplanan class-gender anatomik template kullanılır.
- Cross-attention: landmark tokenları tüm yüzeyden bilgi toplar.
- Anatomik graph bias: landmark tokenları arasında bölge, simetri ve orta hat ilişkilerini temsil eder.
- Heatmap hedefi: her landmark için Gaussian yüzey aktivasyonu.
- Koordinat hedefi: train-template + modelin öğrendiği residual düzeltme.
- Kayıp: pozitif ağırlıklı heatmap, koordinat, anatomik mesafe, simetri, klinik eşik ve belirsizlik bileşenleri.

Ana değerlendirme metriği:

```text
ALE = mean(||AGH-Former landmark_i - uzman landmark_i||_2)
```

## Hızlı Smoke Test

```bash
cd agh_former_orthodontic_comparison
python run_orthodontic_aghformer.py \
  --data-root ../data/dataset \
  --splits-json ../shared_splits/orthodontic_180_60_60_seed42.json \
  --transformation-dir ../palnet_orthodontic_comparison/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir runs/aghformer_v2_template_smoke \
  --surface-points 512 \
  --epochs 2 \
  --patience 2 \
  --batch-size 2 \
  --width 64 \
  --blocks 1 \
  --heads 4 \
  --template-mode class_gender \
  --prediction-mode direct \
  --selection-metric raw \
  --coord-weight 1.0 \
  --heatmap-positive-weight 20 \
  --heatmap-ce-weight 0.05 \
  --topk 10 \
  --device auto
```

## A100 / Colab Pro Ana Koşu

```bash
cd agh_former_orthodontic_comparison
python -u run_orthodontic_aghformer.py \
  --data-root ../data/dataset \
  --splits-json ../shared_splits/orthodontic_180_60_60_seed42.json \
  --transformation-dir ../palnet_orthodontic_comparison/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir runs/aghformer_v2_template_p12000_w192_b4_e220 \
  --surface-points 12000 \
  --eval-surface-points 12000 \
  --epochs 220 \
  --patience 35 \
  --batch-size 2 \
  --lr 0.0008 \
  --weight-decay 0.0001 \
  --width 192 \
  --blocks 4 \
  --heads 6 \
  --mlp-ratio 2.0 \
  --heatmap-sigma-start 5.0 \
  --heatmap-sigma-end 2.5 \
  --heatmap-loss weighted_mse \
  --heatmap-positive-weight 20 \
  --heatmap-ce-weight 0.05 \
  --template-mode class_gender \
  --prediction-mode direct \
  --selection-metric raw \
  --residual-scale 0.18 \
  --coord-weight 1.0 \
  --structure-weight 0.08 \
  --symmetry-weight 0.02 \
  --clinical-weight 0.05 \
  --uncertainty-weight 0.02 \
  --rotation-aug-deg 2.0 \
  --point-jitter-std 0.001 \
  --feature-dropout 0.05 \
  --topk 30 \
  --temperature 1.0 \
  --device auto
```

Bellek yetmezse sırasıyla `--batch-size 1`, sonra `--surface-points 8192`, sonra `--width 128` denenmelidir.

## Stage 2 Lokal Refiner

Stage 2, tamamlanmış AGH-Former v2 checkpoint'ini sabit Stage 1 olarak kullanır. Her landmark için Stage 1 snapped tahmini çevresinden lokal patch çıkarır. v4 sürümünde refiner, residual düzeltmeye ek olarak patch içinde lokal heatmap yardımcı görevi öğrenir ve tek nokta snap yerine top-k surface projection kullanır:

```text
stage2_prediction = stage1_prediction + predicted_delta
```

Google Colab runner ile:

```bash
cd /content/comparative-study/agh_former_orthodontic_comparison
python -u colab_run_aghformer_shared_metrics.py --preset stage2
```

Manuel çalıştırma:

```bash
python -u run_aghformer_stage2_refiner.py \
  --data-root ../data/dataset \
  --splits-json ../shared_splits/orthodontic_180_60_60_seed42.json \
  --transformation-dir ../palnet_orthodontic_comparison/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --stage1-run-dir runs/aghformer_v2_template_p12000_w192_b4_e220 \
  --output-dir runs/aghformer_v4_stage2_heatmap_refiner_p12000 \
  --surface-points 12000 \
  --patch-points 1024 \
  --patch-radius-mm 18 \
  --patch-heatmap-sigma-mm 3.0 \
  --stage1-center snapped \
  --epochs 160 \
  --patience 30 \
  --batch-size 256 \
  --refiner-width 192 \
  --landmark-embedding-dim 48 \
  --final-mode center_delta \
  --patch-heatmap-weight 0.25 \
  --patch-heatmap-positive-weight 20 \
  --patch-heatmap-ce-weight 0.05 \
  --projection-mode topk_distance \
  --projection-topk 5 \
  --center-jitter-mm 1.5 \
  --point-noise-mm 0.1 \
  --point-dropout 0.05 \
  --device auto
```

Stage 2 smoke test:

```bash
python -u colab_run_aghformer_shared_metrics.py --preset stage2_smoke
```

## Evaluate-only

```bash
python -u run_orthodontic_aghformer.py \
  --data-root ../data/dataset \
  --splits-json ../shared_splits/orthodontic_180_60_60_seed42.json \
  --transformation-dir ../palnet_orthodontic_comparison/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir runs/aghformer_v2_template_p12000_w192_b4_e220 \
  --surface-points 12000 \
  --width 192 \
  --blocks 4 \
  --heads 6 \
  --evaluate-only \
  --model-path runs/aghformer_v2_template_p12000_w192_b4_e220/best_model.pth \
  --template-mode class_gender \
  --prediction-mode direct \
  --selection-metric raw \
  --topk 30 \
  --device auto
```

## Çıktılar

- `metrics.json`: ana ALE, median, PCK ve gelişmiş analizler.
- `predictions_test.csv`: uzman ve AGH-Former koordinatları.
- `template_landmarks.json`: yalnızca train split üzerinden hesaplanan template prior.
- `landmark_metrics_test.csv`: landmark bazlı mean, median, std, max ve PCK.
- `clinical_thresholds_test.csv`: PCK@2mm, PCK@2.5mm, PCK@3mm.
- `class_metrics_test.csv`: Class I / II / III performansı.
- `gender_metrics_test.csv`: kadın / erkek performansı.
- `difficult_landmarks_test.csv`: zor landmark sıralaması.
- `structure_metrics_test.csv`: anatomik yapı tutarlılığı.
- `uncertainty_metrics_test.csv`: belirsizlik-hata korelasyonu.
- `history.json`: epoch bazlı eğitim geçmişi.
- `best_model.pth`: en iyi validasyon ALE checkpoint.
- `stage1_predictions_train.csv`, `stage1_predictions_val.csv`, `stage1_predictions_test.csv`: Stage 2 için kullanılan sabit Stage 1 tahminleri.
- `best_refiner.pth`: Stage 2 lokal refiner checkpoint.
- `refined_predictions_test.csv`: Stage 2 nihai test tahminleri.
- `metrics_refined.json`: Stage 2 raw/snapped ALE, PCK ve detaylı analizler.
