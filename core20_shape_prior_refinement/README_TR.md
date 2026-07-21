# Core20 Shape-Prior Residual Refiner

Bu deney 2.5D lokal patch yaklaşımından farklıdır. Mesh veya lokal yüzey görüntüsü kullanmaz; AGH-Former Stage2/V6 tahminlerinin oluşturduğu 23 noktalı anatomik konfigürasyondan sistematik residual düzeltme öğrenir.

Ana fikir:

- Girdi: her hastadaki 23 predicted landmark koordinatı, class ve gender bilgisi.
- Hedef: uzman landmarklarına göre residual vektörleri.
- Model: kapalı-form ridge regression + validation ile seçilen shrinkage.
- Seçim: validation setinde iyileşen landmarklar testte gated olarak aktif edilir.

Bu yöntem küçük veri setlerinde derin lokal refiner yerine daha kontrollüdür; overfit riski ridge katsayısı ve shrinkage ile sınırlanır.

Colab çalıştırma:

```bash
cd /content/comparative-study/core20_shape_prior_refinement
python -u colab_run_shape_prior.py
```

Yalnızca core20 düzeltmesi için:

```bash
python -u colab_run_shape_prior.py \
  --target-landmarks core20 \
  --gate-landmarks core20 \
  --selection-metric core20 \
  --output-dir /content/drive/MyDrive/orthodontic/diffusion_runs/shape_prior_core20_only
```

Runner base prediction kaynağını şu sırayla arar:

```text
aghformer_v12_stage3_core20_refiner_v6
aghformer_v11_stage3_mid_refiner_v6
aghformer_v6_stage2_raw_fine_refiner_p12000
```

Ana çıktılar:

- `metrics_shape_prior.json`
- `predictions_val.csv`
- `predictions_test.csv`
- `landmark_metrics_test.csv`
- `delta_analysis_shape_prior.csv`
- `config_shape_prior.json`

Yerel v11 prediction CSV'leriyle doğrulama sonucu:

```text
Base ALE: 2.5049
Shape-prior all-target ALE: 2.3684
Shape-prior gated ALE: 2.3789
Core20 base/gated ALE: 2.1143 -> 2.0620
```

Core20-only koşu:

```text
Base ALE: 2.5049
Shape-prior all-target ALE: 2.4489
Shape-prior gated ALE: 2.4594
Core20 base/gated ALE: 2.1143 -> 2.0620
```
