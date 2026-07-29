# AGH-HardNet Refiner

Bu alt proje, AGH-Former'in genel 23 landmark tahminini koruyup yalnızca zor ve belirsiz karakterli landmarklar olan `LM0` Trichion, `LM21` Gonion Left ve `LM22` Gonion Right noktalarını iyileştirmek için hazırlanmıştır.

Ana fikir:

```text
AGH tahmini
-> LM0/21/22 etrafinda genis yuzey patch'i
-> aday yuzey noktalarini skorlama
-> top-k weighted coordinate + kucuk residual
-> final 23 landmark dosyasi
```

## Neden Ayrı Model?

Mevcut AGH/Atlas sonuçlarında Core20 noktaları yaklaşık 2 mm bandına yaklaşırken LM0/21/22 yaklaşık 5 mm seviyesinde kalmaktadır. Bu üç nokta yüzey üzerinde burun ucu gibi geometrik olarak keskin tekil noktalar değildir. Bu nedenle klasik koordinat regresyonu yerine aday seçimi ve anatomik bağlam daha uygundur.

## Varsayılan Girdiler

Colab çalıştırıcı, v6 AGH klasörü için şu dosyaları bekler:

```text
aghformer_v6_stage2_raw_fine_refiner_p12000/stage1_predictions_train.csv
aghformer_v6_stage2_raw_fine_refiner_p12000/refined_predictions_val.csv
aghformer_v6_stage2_raw_fine_refiner_p12000/refined_predictions_test.csv
aghformer_v6_stage2_raw_fine_refiner_p12000/stage1_point_cache/
```

Not: v6 klasöründe `refined_predictions_train.csv` olmadığı için train splitinde stage1 prediction kullanılır. Val/test tarafında stage2 refined prediction kullanılır. Daha adil bir koşu için ileride stage2 train prediction üretmek daha doğru olur.

## Hızlı Çalıştırma

```bash
cd /content/comparative-study/agh_hardnet_refiner
python -u colab_run_agh_hardnet.py --preset smoke
```

Ana koşu:

```bash
python -u colab_run_agh_hardnet.py --preset full
```

## Çıktılar

```text
oracle_report.json
oracle_val.csv / oracle_test.csv
best_model.pth
history.json
metrics_hardnet.json
predictions_val.csv
predictions_test.csv
landmark_metrics_test.csv
```

Önce `oracle_report.json` yorumlanmalıdır. Eğer LM0/21/22 patchleri içinde uzman noktasına 2 mm'den yakın aday yüzey noktası sık bulunmuyorsa, problem modelden çok landmark/mesh/registration tanımı olabilir.

## İlk Bulguyu Nasıl Yorumlamalı?

Yerel tam oracle analizi, v6 val/test patchlerinde doğru yüzey adayının çoğu örnekte bulunduğunu gösterdi:

```text
Test hard3 base ALE:   ~5.11 mm
Test hard3 oracle ALE: ~1.19 mm
Test oracle PCK@2mm:   ~90%
```

Bu, sorunun mesh içinde doğru adayın olmaması değil, doğru adayın seçilememesi olduğunu gösterir. Bu nedenle specialist candidate-scoring yaklaşımı mühendislik olarak anlamlıdır.

Mini CPU smoke koşusu yalnız teknik doğrulama içindir. v6 klasöründe `refined_predictions_train.csv` bulunmadığı için default train kaynağı `stage1_predictions_train.csv`, val/test kaynağı ise stage2 refined prediction dosyalarıdır. Bu dağılım farkı HardNet kazancını sınırlayabilir.

En doğru ana deney için önerilen sıra:

```text
1. Oracle preset'i çalıştır ve candidate coverage'i doğrula.
2. Mümkünse AGH Stage2 train predictions üret.
3. Train/val/test için aynı prediction seviyesini kullanarak full preset'i çalıştır.
4. Final 23 sonucu Core20 sabit + LM0/21/22 HardNet olarak raporla.
```
