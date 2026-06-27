# PAL-Net ile 23 Noktalı Ortodontik Yumuşak Doku Analizi

Bu klasör, `data/dataset` altındaki sınıflandırılmış 3B hasta yüz meshlerini ve uzman ortodontist tarafından işaretlenen 23 landmark noktasını PAL-Net projesine uygular.

## İçerik

- `upstream/`: GitHub'dan indirilen PAL-Net kodu.
- `upstream/src/datasets/orthodontic_dataset.py`: Bu veri setinin `Class*/men|women/*.ply` ve `Class*-Landmark/*.txt` düzenini PAL-Net formatına çeviren dataset adapter'ı.
- `upstream/run_orthodontic.py`: Eğitim, doğrulama, test ve ALE raporlama script'i.
- `runs/`: Çalıştırmaların model, metrik ve tahmin çıktıları buraya yazılır.

## Metrik

Sonuç metriği Average Localization Error (ALE) olarak hesaplanır:

```text
ALE = mean(||PAL-Net landmark_i - uzman landmark_i||_2)
```

Hesap 23 landmark ve test örneklerinin tamamı üzerinden yapılır. Script iki PAL-Net sonucu üretir:

- `palnet_raw`: Ağın doğrudan 3B koordinat tahmini.
- `palnet_snapped`: Tahminin en yakın örneklenmiş mesh yüzey noktasına taşınmış hali. Ana karşılaştırma için bunu kullanmak daha tutarlıdır, çünkü landmark yüz üzerinde olmalıdır.

Ek olarak `mean_shape_baseline_*` değerleri de yazılır. Bu değer PAL-Net değildir; eğitim setinin ortalama uzman landmark şablonunu test yüzlerine karşı ölçen basit referanstır.

## Kurulum

```bash
cd palnet_orthodontic_comparison/upstream
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r ../requirements.txt
```

## Çalıştırma

Hızlı bir deneme:

```bash
cd palnet_orthodontic_comparison/upstream
source ../.venv/bin/activate
python run_orthodontic.py --epochs 3 --batch-size 4 --patch-size 250 --surface-points 5000
```

Daha gerçekçi eğitim:

```bash
python run_orthodontic.py --epochs 200 --batch-size 8 --patch-size 250 --surface-points 10000
```

## Çıktılar

Varsayılan çıktı klasörü:

```text
palnet_orthodontic_comparison/runs/orthodontic_palnet/
```

Önemli dosyalar:

- `metrics.json`: PAL-Net ve uzman doktor karşılaştırması için ALE özeti.
- `predictions_test.csv`: Her test hastası ve her landmark için uzman koordinatı, PAL-Net koordinatı ve lokalizasyon hatası.
- `group_metrics_test.csv`: Class/cinsiyet bazlı ALE.
- `splits.json`: Train/validation/test ayrımı ve landmark dosyası eksik olan meshler.
- `best_model.pth`: En iyi validation kaybına sahip PAL-Net ağırlıkları.
