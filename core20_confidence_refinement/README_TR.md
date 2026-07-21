# Core20 Confidence Refinement

Bu deney AGH-Former v6 ve Stage3 çıktılarından öğrenilen bir confidence gate kullanır.
Amaç yeni koordinat regresyonu yapmak değil, Stage3 düzeltmesine hangi örnek-landmarklarda güvenileceğini seçmektir.

Varsayılan girişler:

- `agh_former_orthodontic_comparison/aghformer_v11_stage3_mid_refiner_v6/stage3_predictions_val.csv`
- `agh_former_orthodontic_comparison/aghformer_v11_stage3_mid_refiner_v6/stage3_predictions_test.csv`

Çalıştırma:

```bash
python -u run_core20_confidence_refinement.py \
  --stage3-run-dir ../agh_former_orthodontic_comparison/aghformer_v11_stage3_mid_refiner_v6 \
  --output-dir runs/core20_gate_v11
```

Colab:

```bash
python -u colab_run_core20_gate.py
```

Core20 Stage3 çıktısı için:

```bash
python -u colab_run_core20_gate.py \
  --stage3-run-dir /content/drive/MyDrive/orthodontic/diffusion_runs/aghformer_v12_stage3_core20_refiner_v6 \
  --output-dir /content/drive/MyDrive/orthodontic/diffusion_runs/core20_gate_v12 \
  --target-landmarks 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
```

Yerel smoke sonucu v11 üzerinde gate şu yönde çalışmıştır:

```text
Base test ALE: 2.5049
Stage3 all test ALE: 2.5011
Gate test ALE: 2.4960
```

Çıktılar:

- `metrics_gate.json`: base, Stage3, gated sonuçları.
- `gated_predictions_test.csv`: final seçimler.
- `gate_thresholds.json`: validation üzerinde seçilen eşikler.
- `feature_importance.json`: kullanılan gate modelinin katsayıları/önem sırası.
