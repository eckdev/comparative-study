# Core20 Shape-Prior Residual Refiner

Bu deney 2.5D lokal patch yaklaşımından farklıdır. Mesh veya lokal yüzey görüntüsü kullanmaz; AGH-Former Stage2/V6 tahminlerinin oluşturduğu 23 noktalı anatomik konfigürasyondan sistematik residual düzeltme öğrenir.

Ana fikir:

- Girdi: her hastadaki 23 predicted landmark koordinatı, class ve gender bilgisi.
- Hedef: uzman landmarklarına göre residual vektörleri.
- Model: kapalı-form ridge regression + validation ile seçilen shrinkage.
- Seçim: validation setinde iyileşen landmarklar testte gated olarak aktif edilir.
- Varsayılan final protokolü: `shape_prior`. Yani tüm anatomik residual kalibrasyonu uygulanır; `gated` sonuç ayrıca raporlanır.

Bu yöntem küçük veri setlerinde derin lokal refiner yerine daha kontrollüdür; overfit riski ridge katsayısı ve shrinkage ile sınırlanır.

Colab çalıştırma:

```bash
cd /content/comparative-study/core20_shape_prior_refinement
python -m pip install -q -r requirements.txt
python -u colab_run_shape_prior.py
```

Runner seçilen AGH prediction klasöründe `refined_predictions_train.csv` bulamazsa önce bunu otomatik üretir. Bu, mevcut AGH Stage2 `best_refiner.pth` checkpoint'i ile train split inference yapar; yeniden model eğitimi başlatmaz. Bu davranışı kapatmak için:

```bash
python -u colab_run_shape_prior.py --skip-refined-train-export
```

Yalnızca core20 düzeltmesi için:

```bash
python -u colab_run_shape_prior.py \
  --target-landmarks core20 \
  --gate-landmarks core20 \
  --selection-metric core20 \
  --output-dir /content/drive/MyDrive/orthodontic/diffusion_runs/shape_prior_core20_only
```

Per-landmark flat+metadata varyantı için:

```bash
python -u colab_run_shape_prior.py \
  --preset per_landmark_flat_meta \
  --prediction-dir /content/drive/MyDrive/orthodontic/diffusion_runs/aghformer_v6_stage2_raw_fine_refiner_p12000 \
  --output-dir /content/drive/MyDrive/orthodontic/diffusion_runs/shape_prior_local_per_landmark_flat_meta
```

Klasor adinin `per_landmark` olmasi tek basina yeterli degildir; bu varyant icin `--calibration-mode per_landmark` argumani mutlaka verilmelidir.
Guncel runner'da bunu daha guvenli yapmak icin `--preset per_landmark_flat_meta` kullanilabilir.

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
- `landmark_metrics_val.csv`
- `delta_analysis_shape_prior.csv`
- `bootstrap_metrics.json`
- `group_metrics_val.csv`
- `group_metrics_test.csv`
- `config_shape_prior.json`

## Shape-Prior Stacker

HardNet sonuclarinda LM0/21/22 icin kazanc cok sinirli kaldiginda, shape-prior varyantlarini validasyon uzerinden residual stacking ile birlestirmek daha kontrollu bir sonraki denemedir.

Colab:

```bash
cd /content/comparative-study/core20_shape_prior_refinement
python -m pip install -q -r requirements.txt
python -u colab_run_shape_prior_stacker.py
```

Beklenen cikti klasoru:

```text
/content/drive/MyDrive/orthodontic/diffusion_runs/shape_prior_stacker
```

Bu kosu yeni derin model egitmez. Daha once uretilmis shape-prior prediction dosyalarindan validation-selected residual kombinasyonu ogrenir ve test setine kilitli uygular.

Sağlamlaştırma çıktıları:

- Hasta-bazlı bootstrap CI: `bootstrap_metrics.json`
- Landmark bazlı base / shape-prior / gated / selected karşılaştırması
- Class I/II/III ve kadın/erkek grup metrikleri
- Validation sweep: `sweep_validation.csv`

Yerel v11 prediction CSV'leriyle doğrulama sonucu:

```text
Base ALE: 2.5049
Shape-prior all-target ALE: 2.3684
Shape-prior gated ALE: 2.3789
Selected policy: shape_prior
Selected median: 1.9124
Core20 base/selected ALE: 2.1143 -> 2.0499
All23 ALE delta CI95: [-0.1668, -0.1075]
```

Core20-only koşu:

```text
Base ALE: 2.5049
Shape-prior all-target ALE: 2.4489
Shape-prior gated ALE: 2.4594
Core20 base/gated ALE: 2.1143 -> 2.0620
```
