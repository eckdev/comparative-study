# Curve-Supervised Hard3 Refinement

Bu modül AGH-Former vNext'in `LM0=Trichion`, `LM21=Gonion Left` ve
`LM22=Gonion Right` tahminlerini ayrı nokta sınıflandırıcılarıyla değil, önce
anatomik destek eğrisini öğrenerek düzeltir. Core20 tahminleri dondurulur ve
nihai model yine 23 landmarkı birlikte raporlar.

## Neden Yeni Bir Hedef?

V8-V13 deneylerinde Gonion aday havuzunun oracle hatası yaklaşık `1.0-1.1 mm`
iken seçilen adayların hatası `4.3-5.7 mm` bandında kaldı. Bu, aday erişiminden
çok anatomik tanım belirsizliğine işaret eder. Curve-H3 bu nedenle:

1. Trichion için saç çizgisi desteğini,
2. sol ve sağ Gonion için mandibular alt-arka konturu,
3. bu eğri üzerinde uzman landmarkının olasılık dağılımını

birlikte öğrenir.

Model; iki RGB-depth görünüm, gerçek mesh komşulukları, normal/eğrilik/yoğunluk
özellikleri, yüz şekli bağlamı ve bilateral Gonion bağlamını kullanır. Kayıp;
curve BCE + Dice, soft-listwise landmark ranking, Smooth L1 koordinat,
bilateral geometri, klinik mesafe ve confidence-aware NLL bileşenlerinden oluşur.

## Leakage Protokolü

- Curve-H3 yalnız dış fold'un **train** örnekleriyle fit edilir.
- Train tahminleri iç fold OOF eğitimiyle üretilir.
- Eğriler `raw_mesh_mm` koordinatındadır ve örneğin train-only ICP dönüşümüyle
  aynı uzaya taşınır.
- Validation eğrileri model eğitiminde kullanılmaz; yalnız uzman noktaları
  blend seçimi ve metrik için kullanılır.
- Test eğrileri ve test noktaları konfigürasyon seçimine girmez.
- Curve checkpoint'i `postprocess_version=15` ile eski Hard3 sürümlerinden
  ayrılır.

## Anotasyon Şeması

Boş manifest üretimi:

```python
%cd /content/comparative-study
!python -u curve_supervised_hard3_refinement/prepare_annotations.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --output /content/drive/MyDrive/orthodontic/annotations/hard3_curves_v1.json
```

Her polylineda ham PLY koordinat sisteminde sıralı XYZ noktaları bulunur:

```json
{
  "version": 1,
  "coordinate_space": "raw_mesh_mm",
  "samples": {
    "Class1_F1": {
      "curves": {
        "hairline": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "jaw_left": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "jaw_right": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
      },
      "repeat_landmarks": {
        "0": [[0.0, 0.0, 0.0]],
        "21": [[0.0, 0.0, 0.0]],
        "22": [[0.0, 0.0, 0.0]]
      }
    }
  }
}
```

Eğri başına en az iki nokta gerekir. Curve üzerindeki örnekleme aralığının
yaklaşık `2-3 mm` olması önerilir. Tekrarlı landmark işaretleri isteğe bağlıdır;
varsa target tek bir yapay ortalama yerine en yakın tekrar anotasyonuna göre
hesaplanır.

## Colab Çalıştırma

Mimari ve I/O smoke testi:

```python
%cd /content/comparative-study
!python -u curve_supervised_hard3_refinement/colab_run_curve_hard3.py \
  --preset pseudo_smoke --seed 42
```

Tam pseudo-curve ablation:

```python
!python -u curve_supervised_hard3_refinement/colab_run_curve_hard3.py \
  --preset pseudo_fold1 --seed 42
```

Gerçek eğri anotasyonlu Fold-1 karar deneyi:

```python
!python -u curve_supervised_hard3_refinement/colab_run_curve_hard3.py \
  --preset annotated_fold1 --seed 42 \
  --annotation-manifest /content/drive/MyDrive/orthodontic/annotations/hard3_curves_v1.json \
  --minimum-annotated-samples 60
```

`pseudo_smoke` ve `pseudo_fold1` yayın sonucu değildir. Gerçek deneyde en az 60
dış-train örneğinde üç eğrinin de bulunması zorunludur. Eksik train örnekleri
düşük ağırlıklı pseudo hedefle yarı gözetimli olarak kullanılabilir.

## Çıktılar

```text
fold_1/curve_supervised_hard3_v1/
  curve_hard3_model.pth
  curve_hard3_training_report.json
  hard3_blend_selection.json
  metrics_val.json
  predictions_val.csv
```

`curve_hard3_training_report.json`; anotasyon kapsamını, manifest SHA256
değerini, OOF Hard3/landmark metriklerini, iç-fold sample ID'lerini, parametre
sayısını ve süreyi kaydeder.

## Fold-1 Karar Kuralı

Tam 5-fold CV yalnız aşağıdakilerin tamamı sağlanırsa anlamlıdır:

- Overall validation ALE `<= 2.25 mm`
- Hard3 validation ALE `< 4.00 mm`
- Core20 değişimi `0`
- Candidate oracle ALE `<= 1.50 mm`, p95 `<= 3.50 mm`, SDR@2 `>= 0.75`
- Blend bootstrap ve p95 güvenlik kontrolleri geçer
- Gerçek curve anotasyon eşiği sağlanır

Bu gate geçmezse daha fazla fold eğitmek yerine anotasyon kapsamı, eğri tanımı
ve OOF landmark bazlı hata incelenmelidir.
