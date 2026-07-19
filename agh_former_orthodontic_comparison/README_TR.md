# AGH-Former ile Ortodontik 3B Yumuşak Doku Landmark Lokalizasyonu

Bu klasör, mevcut `data/dataset` içindeki 300 ortodontik 3B yüz meshini ve 23 uzman landmarkını kullanarak özgün bir karşılaştırma modeli kurar: **Anatomy-Aware Geodesic Heatmap Transformer (AGH-Former)**.

## Yaklaşım

AGH-Former, PAL-Net, DiffusionNet ve PointNet++ modellerinden farklı olarak landmarkları bağımsız koordinat tahmini olarak değil, anatomik olarak ilişkili bir 23 noktalı konfigürasyon olarak ele alır.

- Girdi: hizalanmış yüzey noktaları.
- Özellikler: `XYZ`, yüzey normali, lokal yoğunluk ve lokal eğrilik proxy değeri.
- Global encoder: yüzey noktalarından nokta bazlı morfolojik temsil çıkarır.
- Landmark tokenları: 23 anatomik landmark için öğrenilebilir token kullanır.
- Cross-attention: landmark tokenları tüm yüzeyden bilgi toplar.
- Anatomik graph bias: landmark tokenları arasında bölge, simetri ve orta hat ilişkilerini temsil eder.
- Heatmap hedefi: her landmark için Gaussian yüzey aktivasyonu.
- Koordinat hedefi: heatmap ağırlıklı koordinat + residual düzeltme.
- Kayıp: heatmap, koordinat, anatomik mesafe, simetri, klinik eşik ve belirsizlik bileşenleri.

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
  --output-dir runs/aghformer_smoke \
  --surface-points 512 \
  --epochs 2 \
  --patience 2 \
  --batch-size 2 \
  --width 64 \
  --blocks 1 \
  --heads 4 \
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
  --output-dir runs/aghformer_p12000_w192_b4_e220 \
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
  --coord-weight 0.45 \
  --structure-weight 0.08 \
  --symmetry-weight 0.02 \
  --clinical-weight 0.05 \
  --uncertainty-weight 0.02 \
  --rotation-aug-deg 4.0 \
  --point-jitter-std 0.003 \
  --feature-dropout 0.05 \
  --topk 30 \
  --temperature 1.0 \
  --device auto
```

Bellek yetmezse sırasıyla `--batch-size 1`, sonra `--surface-points 8192`, sonra `--width 128` denenmelidir.

## Evaluate-only

```bash
python -u run_orthodontic_aghformer.py \
  --data-root ../data/dataset \
  --splits-json ../shared_splits/orthodontic_180_60_60_seed42.json \
  --transformation-dir ../palnet_orthodontic_comparison/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir runs/aghformer_p12000_w192_b4_e220 \
  --surface-points 12000 \
  --width 192 \
  --blocks 4 \
  --heads 6 \
  --evaluate-only \
  --model-path runs/aghformer_p12000_w192_b4_e220/best_model.pth \
  --topk 30 \
  --device auto
```

## Çıktılar

- `metrics.json`: ana ALE, median, PCK ve gelişmiş analizler.
- `predictions_test.csv`: uzman ve AGH-Former koordinatları.
- `landmark_metrics_test.csv`: landmark bazlı mean, median, std, max ve PCK.
- `clinical_thresholds_test.csv`: PCK@2mm, PCK@2.5mm, PCK@3mm.
- `class_metrics_test.csv`: Class I / II / III performansı.
- `gender_metrics_test.csv`: kadın / erkek performansı.
- `difficult_landmarks_test.csv`: zor landmark sıralaması.
- `structure_metrics_test.csv`: anatomik yapı tutarlılığı.
- `uncertainty_metrics_test.csv`: belirsizlik-hata korelasyonu.
- `history.json`: epoch bazlı eğitim geçmişi.
- `best_model.pth`: en iyi validasyon ALE checkpoint.
