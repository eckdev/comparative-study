# Core20 2.5D Local Heatmap Refiner

Bu deney `LM0`, `LM21`, `LM22` dışındaki landmarkları iyileştirmek için AGH-Former v6 tahminleri etrafında lokal 2.5D patch üretir.
Model nokta bulutu yerine tangent-plane raster patch görür ve heatmap + 2D soft-argmax ile lokal düzeltme öğrenir.
Patch görüntüsüne landmark kimliği ve hizalanmış merkez koordinatları da sabit context kanalları olarak eklenir; böylece model benzer lokal yüzeyleri anatomik konuma göre ayırabilir.

Varsayılan hedef landmarklar:

```text
1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
```

Colab çalıştırma:

```bash
cd /content/comparative-study/core20_2p5d_refinement
python -u colab_run_2p5d.py --preset smoke
python -u colab_run_2p5d.py --preset a100
```

Runner base merkezleri otomatik olarak şu sırayla arar:

```text
aghformer_v12_stage3_core20_refiner_v6
aghformer_v11_stage3_mid_refiner_v6
aghformer_v6_stage2_raw_fine_refiner_p12000
```

Bu nedenle v12 klasörü Drive'da varsa train/val/test Stage2 merkezleri oradan kullanılır.

Yerel/manuel çalıştırma:

```bash
python -u run_core20_2p5d_refiner.py \
  --data-root ../data/dataset \
  --splits-json ../shared_splits/orthodontic_180_60_60_seed42.json \
  --base-run-dir ../agh_former_orthodontic_comparison/aghformer_v6_stage2_raw_fine_refiner_p12000 \
  --output-dir runs/core20_2p5d_smoke \
  --epochs 2 \
  --max-samples 24
```

Ana çıktılar:

- `metrics_2p5d.json`: base, all-target ve validation-gated sonuçlar.
- `predictions_test.csv`: base/2.5D/final koordinatlar ve hatalar.
- `landmark_metrics_test.csv`: landmark bazlı sonuç.
- `history_2p5d.json`: epoch geçmişi.
