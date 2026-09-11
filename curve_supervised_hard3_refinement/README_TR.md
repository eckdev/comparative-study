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

V1 pseudo deneyi `overall=2.2086 mm`, `Core20=1.8338 mm` ve
`Hard3=4.7077 mm` üretmiştir. ROI oracle `0.78 mm` olduğu halde OOF Hard3
`5.27 mm` kalmıştır. V2 bu nedenle iki RGB-depth görünüm ve gerçek mesh
komşuluklarını iki aşamada işler:

1. Curve pretraining yalnız destek eğrisini öğrenir.
2. Curve-conditioned ikinci surface-graph geçişi landmarkı destek üzerinde seçer.

Gerçek eğri bulunan örnekler source-balanced batchlerle oversample edilir ve
inner foldlara dengeli dağıtılır. Kayıp; curve BCE + Dice + beklenen curve
mesafesi, soft-listwise landmark ranking, Smooth L1 koordinat, bilateral
geometri, klinik mesafe ve confidence-aware NLL bileşenlerinden oluşur.
İç-fold checkpoint seçimi Hard3 ALE yanında yalnız gerçek validation eğrilerinde
ölçülen beklenen destek-eğrisi mesafesini düşük ağırlıkla kullanır. Böylece
curve-only ön eğitim joint aşamada tamamen unutulmaz; pseudo eğriler checkpoint
seçimine ek sinyal olarak girmez.

## Leakage Protokolü

- Curve-H3 yalnız dış fold'un **train** örnekleriyle fit edilir.
- Train tahminleri iç fold OOF eğitimiyle üretilir.
- Eğriler `raw_mesh_mm` koordinatındadır ve örneğin train-only ICP dönüşümüyle
  aynı uzaya taşınır.
- Validation eğrileri model eğitiminde kullanılmaz; yalnız uzman noktaları
  blend seçimi ve metrik için kullanılır.
- Test eğrileri ve test noktaları konfigürasyon seçimine girmez.
- Curve checkpoint'i `postprocess_version=16` ile eski Hard3 sürümlerinden
  ayrılır.
- Manifest cache hash'i yalnız outer-train anotasyonlarından hesaplanır.

## Anotasyon Şeması

Fold-1 train içinden sınıf/cinsiyet dengeli 24 örneklik pilot manifest üretimi:

```python
%cd /content/comparative-study
!python -u curve_supervised_hard3_refinement/prepare_annotations.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --split-report /content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/publication_cv_stage1_v4_seed42/fold_1/split_and_leakage_report.json \
  --split-name train --sample-count 24 --seed 42 \
  --output /content/drive/MyDrive/orthodontic/annotations/hard3_curves_pilot_v2.json
```

Komut ayrıca anotasyon sırasını ve PLY yollarını içeren
`hard3_curves_pilot_v2_tasks.csv` dosyasını üretir. Var olan manifest varsayılan
olarak ezilmez.

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

Anotasyon QA'sını eğitimden bağımsız çalıştırmak için:

```python
!python -u curve_supervised_hard3_refinement/validate_annotations.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --manifest /content/drive/MyDrive/orthodontic/annotations/hard3_curves_pilot_v2.json \
  --minimum-annotated-samples 24
```

Bu kontrol curve noktalarının ham mesh yüzeyine uzaklığını, ilgili uzman
landmarkının eğriye uzaklığını ve polyline örnekleme aralığını raporlar.
`annotated_pilot` ve `annotated_fold1` presetleri bu QA'yı eğitimden önce
otomatik çalıştırır; koordinat sistemi hatasında pahalı koşu başlamaz.

## Colab Çalıştırma

Mimari ve I/O smoke testi:

```python
%cd /content/comparative-study
!python -u curve_supervised_hard3_refinement/colab_run_curve_hard3.py \
  --preset pseudo_smoke --seed 42
```

Tam pseudo-curve ablation yalnız V2 regresyon karşılaştırması içindir:

```python
!python -u curve_supervised_hard3_refinement/colab_run_curve_hard3.py \
  --preset pseudo_fold1 --seed 42
```

24 gerçek eğri anotasyonlu, yayın dışı pilot:

```python
!python -u curve_supervised_hard3_refinement/colab_run_curve_hard3.py \
  --preset annotated_pilot --seed 42 \
  --annotation-manifest /content/drive/MyDrive/orthodontic/annotations/hard3_curves_pilot_v2.json \
  --pilot-annotated-samples 24
```

Pilot sinyal verdiğinde gerçek eğri anotasyonlu Fold-1 yayın kapısı:

```python
!python -u curve_supervised_hard3_refinement/colab_run_curve_hard3.py \
  --preset annotated_fold1 --seed 42 \
  --annotation-manifest /content/drive/MyDrive/orthodontic/annotations/hard3_curves_publication_v2.json \
  --minimum-annotated-samples 60
```

`pseudo_*` ve `annotated_pilot` koşulları daima `publication_ready=False`
kalır. CLI ile düşük bir eşik verilse bile yalnız `publication` modu ve en az 60
dış-train örneğinde üç eğri bulunması yayın hazır durumunu açar. Eksik train
örnekleri düşük ağırlıklı pseudo hedefle yarı gözetimli kullanılabilir.

## Çıktılar

```text
fold_1/curve_supervised_hard3_v2/
  annotation_preflight.json
  curve_hard3_model.pth
  curve_hard3_training_report.json
  hard3_blend_selection.json
  metrics_val.json
  predictions_val.csv
```

`curve_hard3_training_report.json`; anotasyon kapsamını, train-only manifest
SHA256 değerini, source-aware fold dağılımını, curve-support Dice/mesafesini,
OOF Hard3/landmark metriklerini, hata-belirsizlik korelasyonunu, parametre
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
