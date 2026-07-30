# All-23 RGB-Geodesic Cascade

Bu klasör, AGH-Former/shape-prior tahminlerini coarse başlangıç olarak kullanabilen; ham PLY
topolojisi, RGB, yüzey normali ve geodezik ROI'lerle 23 landmarkın tamamını birlikte refine eden
yeni deney hattıdır.

## Neyi Düzeltiyor?

- Landmark sözleşmesi sabittir: orta hat `LM0..LM12`, bilateral çiftler
  `(13,16), (14,15), (17,18), (19,20), (21,22)`.
- Varsayılan kayıt `mesh_icp` seçeneğidir. Sadece train meshlerinden medoid seçer; PCA ve
  point-to-plane ICP sırasında hiçbir expert landmark kullanmaz ve ölçek değiştirmez.
- RGB atılmaz. Girdi sırası `XYZ, RGB, lokal RGB kontrastı, normal, yoğunluk, eğrilik`tir.
- Global encoder mesh edge'leri üzerinde seyrek Point Transformer uygular.
- Her landmark için anatomik yarıçaplı geodezik ROI ve ortak/specialized refinement head vardır.
- Point dropout vertexleri edge, softmax ve loss dışında bırakır.
- Checkpoint seçimi doğrudan all-23 validation ALE ile yapılır.
- Test seti checkpoint, postprocess ve confidence calibration kilitlendikten sonra okunur.
- Eğitim başlamadan validation candidate-oracle ALE hesaplanır; varsayılan `1.5 mm` sınırı
  aşılırsa yetersiz ROI kapsamıyla pahalı koşu başlatılmaz.

## Colab Kurulumu

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/eckdev/comparative-study.git /content/comparative-study
!pip install -q -r /content/comparative-study/all23_rgb_geodesic_cascade/requirements.txt
```

Smoke test:

```python
%cd /content/comparative-study/all23_rgb_geodesic_cascade
!python -u colab_run_all23_rgb_geodesic.py --preset smoke
```

İlk tam koşudan önce veri denetimi:

```python
!python -u audit_dataset.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --splits-json /content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json \
  --output-dir /content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/dataset_audit
```

A100 sabit-split geliştirme deneyi:

```python
%cd /content/comparative-study/all23_rgb_geodesic_cascade
!python -u colab_run_all23_rgb_geodesic.py --preset a100 --seed 42
```

Leakage-free 5-fold koşusundan önce tüm foldların ROI kapsamını doğrulayın. Bu komut eğitim
başlatmaz; train-only template, mesh-ICP, normalizasyon ve geodezik ROI cache'lerini hazırlar:

```python
%cd /content/comparative-study/all23_rgb_geodesic_cascade
!python -u colab_run_all23_rgb_geodesic.py --preset cv_preflight --seed 42
```

`preflight_oracle_summary.csv` içinde beş foldun da `validation_oracle_ale <= 1.5 mm` ve
`validation_hard3_oracle_ale <= 2.5 mm` koşullarını sağlaması gerekir. Kontrol geçtikten sonra
leakage-free 5-fold yayın koşusu:

```python
%cd /content/comparative-study/all23_rgb_geodesic_cascade
!python -u colab_run_all23_rgb_geodesic.py --preset cv --seed 42
```

Pipeline kilitlendikten sonraki tekrarlı 5-fold nihai koşu:

```python
!python -u colab_run_all23_rgb_geodesic.py --preset cv_repeated --seed 42
```

`cv` presetinde train/val/test coarse center yalnız ilgili foldun train landmark ortalamasından
üretilir. Böylece refiner eğitim ve değerlendirmede aynı coarse-center dağılımını görür. Eğitimde
template merkezlerine yalnız `1 mm` jitter uygulanır; ROI noktası `1024`, anatomik ROI yarıçapı
çarpanı `1.5` olarak kullanılır. Tüm loss hesapları AMP dışında float32 yapılır ve sonlu olmayan
batch oranı `%1` değerini aşarsa koşu açık bir hata ile durur. Sabit
split `a100` koşusunda val/test için mevcut stacker coarse tahminleri kullanılır; train merkezleri
expert noktalarına yalnız train içinde deterministik sentetik coarse hata eklenerek üretilir. Böylece
in-sample prediction dağılımı kaldırılır. Bununla birlikte mevcut AGH/stacker tahminleri ilk
üretilirken expert-Procrustes kullanıldığı için sabit-split sonuç keşif/ablation niteliğindedir;
makalenin ana sonucu `cv` presetinden alınmalıdır.

Eski `publication_cv_seed*` sonuçlarında train merkezi `expert + synthetic error`, val/test merkezi
ise train template idi. Bu dağılım kayması ve bazı foldlarda görülen `NaN` nedeniyle eski CV klasörü
bilimsel sonuç olarak kullanılmamalıdır. Düzeltilmiş preset sonuçları `publication_cv_v2_seed*`
altına yazar.

## Kontrollü Ablation

```python
!python -u colab_run_all23_rgb_geodesic.py --preset e0
!python -u colab_run_all23_rgb_geodesic.py --preset e1
!python -u colab_run_all23_rgb_geodesic.py --preset e2
!python -u colab_run_all23_rgb_geodesic.py --preset e3
!python -u colab_run_all23_rgb_geodesic.py --preset e5
!python -u colab_run_all23_rgb_geodesic.py --preset e6
!python -u colab_run_all23_rgb_geodesic.py --preset e7
```

- `E1`: düzeltilmiş anatomi/veri hattı, geometry-only.
- `E2`: RGB ve lokal renk kontrastı.
- `E3/E4`: global mesh encoder ve all-23 geodezik local refiner.
- `E5`: anatomik graph/simetri loss.
- `E6`: texture/contour/generic specialized heads.
- `E7`: MSE-over-mesh ve validation/test TTA.
- `E8`: yalnız validation ALE `<=2.05` ise üç seed ensemble.

Üç seed tamamlandıktan sonra:

```python
!python -u ensemble_runs.py \
  --run-dirs \
  /content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/full_fixed_seed42 \
  /content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/full_fixed_seed43 \
  /content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/full_fixed_seed44 \
  --output-dir /content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/ensemble
```

## Çıktılar

Her fold/run aşağıdakileri üretir:

```text
best_model.pth
history.json
normalization.json
alignment/alignment_report.json
split_and_leakage_report.json
metrics_val.json
metrics_test.json
landmark_metrics_{val,test}.csv
group_metrics_{val,test}.csv
predictions_{val,test}.csv
outlier_samples_{val,test}.csv
run_summary.json
```

Ana kıyas `metrics_test.json -> overall.ale` değeridir. `core20`, `hard3`, SDR@2/3/4,
percentile, bootstrap CI ve subgroup sonuçları da aynı dosyalarda raporlanır.

## Önemli Bilimsel Not

`--alignment legacy`, expert landmarklarla üretilmiş Procrustes matrislerini kullanır ve yalnız
eski sonuçlarla mühendislik karşılaştırması içindir. Makalenin ana tablosunda yalnız
`--alignment mesh_icp` ve patient-level CV sonucu kullanılmalıdır. `<2 mm` bir hedef ve kabul
kriteridir; kod bu sonucu önceden garanti etmez.
