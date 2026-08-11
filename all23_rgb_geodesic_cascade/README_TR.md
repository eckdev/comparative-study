# All-23 RGB-Geodesic Cascade

Bu klasör, AGH-Former/shape-prior tahminlerini coarse başlangıç olarak kullanabilen; ham PLY
topolojisi, RGB, yüzey normali ve geodezik ROI'lerle 23 landmarkın tamamını birlikte refine eden
yeni deney hattıdır.

## Neyi Düzeltiyor?

- Landmark sözleşmesi sabittir: orta hat `LM0..LM12`, bilateral çiftler
  `(13,16), (14,15), (17,18), (19,20), (21,22)`.
- Varsayılan kayıt `mesh_icp` seçeneğidir. Yalnız train meshlerinden seçilen sekiz merkezi yüzün
  robust karşılık medyanıyla atlas kurar. Her hasta birleşik atlas ve en uygun atlas üyelerine
  birden fazla rijit başlangıçla kaydedilir; seçim simetrik kırpılmış yüzey residualıyla yapılır.
  PCA/ICP sırasında expert landmark kullanılmaz ve ölçek değiştirilmez.
- RGB atılmaz. Girdi sırası `XYZ, RGB, lokal RGB kontrastı, normal, yoğunluk, eğrilik`tir.
- Global encoder mesh edge'leri üzerinde seyrek Point Transformer uygular.
- Ayrı Stage 1 global coarse ağı, outer-train için sabit-epoch 5-fold OOF tahmin üretir. Her OOF
  model holdout dışındaki tüm outer-train örneklerini görür; holdout etiketi checkpoint veya blend
  seçiminde kullanılmaz. Validation/test merkezleri yalnız outer-train modelinden gelir.
- Her landmark için çoklu-seed geodezik ve Euclidean adayları birleştiren hybrid ROI vardır.
  Train-only registration residualı yüksek örneklerde ROI etiketsiz biçimde genişler.
- Trichion alın/orta-hat tokenlarıyla; iki Gonion birbirleri ve alt yüz orta-hat tokenlarıyla ortak
  bağlamda modellenir. Confidence-aware gate, coarse ve lokal tahmin arasında train üzerinde
  öğrenilen landmark/örnek bazlı bir katsayı üretir.
- Point dropout vertexleri edge, softmax ve loss dışında bırakır.
- Checkpoint seçimi doğrudan all-23 validation ALE ile yapılır.
- Test seti checkpoint, postprocess ve confidence calibration kilitlendikten sonra okunur.
- Stage 2 başlamadan overall/Hard3/p95/max/sample-level candidate-oracle kapıları uygulanır.

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

Leakage-free 5-fold koşusundan önce tüm foldların Stage 1 ve ROI kapsamını doğrulayın. Bu komut
Stage 2 eğitimini başlatmaz; train-only çoklu atlası, fixed-epoch OOF Stage 1 modellerini ve hybrid
ROI cache'lerini hazırlar. İlk kullanım uzun sürebilir; ana koşu aynı Stage 1 checkpoint/cache'lerini
yeniden kullanır. Beş outer foldun her birinde beş OOF model ve bir outer-train model olmak
üzere toplam `30` küçük Stage 1 modeli eğitilir; bu nedenle preflight birkaç saat sürebilir:

```python
%cd /content/comparative-study/all23_rgb_geodesic_cascade
!python -u colab_run_all23_rgb_geodesic.py --preset cv_preflight --seed 42
```

`preflight_oracle_summary.csv` içinde beş foldun da aşağıdaki koşulları sağlaması gerekir:

```text
validation_oracle_ale <= 1.5 mm
validation_hard3_oracle_ale <= 2.5 mm
validation_oracle_p95 <= 2.0 mm
validation_oracle_max <= 15.0 mm
validation_sample_oracle_max <= 2.0 mm
```

Aynı tabloda `stage1_train_oof_ale`, `stage1_oof_p95`, `stage1_validation_ale` ve
mutlak `stage1_oof_validation_gap` ile yönü gösteren
`stage1_oof_validation_signed_gap` bulunur. Bu değerler coarse
model kalitesini, oracle sütunları ise dinamik ROI'nin uzman noktasını gerçekten kapsayıp
kapsamadığını ayırarak gösterir. Stage 1 tamamlandıktan sonra kalite kapısı başarısız olsa bile
checkpoint ve tahminler silinmez; eşik/ROI ayarı düzeltilerek yeniden çağrıldığında pahalı Stage 1
eğitimi cache'den yüklenir.
Preflight bağlantısı bir outer foldun ortasında kesilirse tamamlanmış inner Stage 1 modelleri de
`training_complete.json` imzaları üzerinden yeniden kullanılır; yalnız yarım kalan inner model
baştan eğitilir.

Kontrol geçtikten sonra leakage-free 5-fold yayın koşusu:

```python
%cd /content/comparative-study/all23_rgb_geodesic_cascade
!python -u colab_run_all23_rgb_geodesic.py --preset cv --seed 42
```

Mevcut E8 sonucunun detayli denetimi
[`ENGINEERING_AUDIT_E8_TR.md`](ENGINEERING_AUDIT_E8_TR.md) dosyasindadir. Yeni E9
candidate-ranker icin preflight ve Stage 1 tekrar calistirilmaz. Ilk olarak yalniz Fold 1,
`publication_cv_stage1_v4_seed42` altindaki alignment, OOF Stage 1 ve ROI cache'leriyle
calistirilir:

```python
%cd /content/comparative-study/all23_rgb_geodesic_cascade
!python -u colab_run_all23_rgb_geodesic.py --preset e9_dev --seed 42
```

Fold 1 kabul kapisini otomatik hesaplayin:

```python
!python -u evaluate_e9_gate.py \
  --baseline-root /content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/publication_cv_stage1_v4_seed42 \
  --candidate-root /content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/publication_e9_cv_seed42 \
  --fold 1
```

Fold 1 validation sonucu rapordaki kabul kapilarini gecerse ayni E9 tum foldlara acilir:

```python
!python -u colab_run_all23_rgb_geodesic.py --preset e9_cv --seed 42
```

Iki preset de `publication_e9_cv_seed42` klasorunu kullanir. `e9_dev` validation-only
calisir ve test etiketlerini tuketmez. Tam kosu Fold 1'in tamamlanmis Stage 2 checkpointini
yukler, validation kalibrasyonunu kilitler ve yalniz final test degerlendirmesini yapar;
Fold 1 yeniden egitilmez.

E10, pahali Stage 1/registration/ROI cache'lerini yine `publication_cv_stage1_v4_seed42`
klasorunden kullanir. Once iki epochluk mimari ve iki-fazli egitim smoke testi yapilir:

```python
%cd /content/comparative-study/all23_rgb_geodesic_cascade
!python -u colab_run_all23_rgb_geodesic.py --preset e10_smoke --seed 42
```

Ardindan test etiketlerini acmadan yalniz Fold 1 validation deneyi calistirilir:

```python
!python -u colab_run_all23_rgb_geodesic.py --preset e10_dev --seed 42
```

E10'un Fold 1 validation sonucu E9'a gore anlamli kazanc saglarsa bes fold acilir:

```python
!python -u colab_run_all23_rgb_geodesic.py --preset e10_cv --seed 42
```

E10'da LM0, RGB ve lokal texture-gradient ozelliklerini kullanan bagimsiz bir candidate
ranker ile cozulur. LM21/22, contralateral coarse anchor kullanmadan refine edilmis LM10-12
alt-orta-hat noktalarina gore ortak bilateral candidate-pair ranker ile puanlanir. Candidate
hedefi tek en yakin vertex degil, uzmana geodezik/Euclidean yakinliga gore yumusayan listwise
dagilimdir; yuksek puanli uzak adaylar hard-negative olarak ek cezalandirilir. Refiner fazinda
gate parametreleri donuktur. En iyi refiner checkpointi kilitlendikten sonra yalniz sample-specific
gate acilir ve ayri checkpoint/early stopping ile egitilir.

Her iki komut da kesintiden sonra aynen yeniden calistirilabilir. Tamamlanan foldlar konfigürasyon
imzasi dogrulandiktan sonra atlanir; yarim kalan fold `last_model.pth` icindeki model, optimizer,
scheduler, AMP scaler ve RNG durumundan devam eder. Hiperparametre degistirildiyse yeni bir output
klasoru kullanin. Bilerek bastan baslatmak icin dogrudan ana scripte
`--force-stage2-retrain` verilebilir.

Pipeline kilitlendikten sonraki tekrarlı 5-fold nihai koşu:

```python
!python -u colab_run_all23_rgb_geodesic.py --preset cv_repeated --seed 42
```

`cv` presetinde outer-train merkezleri beş inner modelin gerçek OOF tahminlerinden üretilir. Her OOF
örneği model fitinden dışlanır. Inner modeller `120` sabit epoch çalışır ve ayrıca validation
ayırmadığı için yaklaşık `153-154` örnek kullanır. Tek global template katsayısı outer-train OOF
tahminleri üzerinde kalibre edilir ve aynı katsayı outer validation/test tahminlerine uygulanır;
outer validation/test etiketleri bu seçimde kullanılmaz. Sabit-epoch eğitim öğrenme oranını ilk
yüzde 80 boyunca korur ve yalnız son yüzde 20'de cosine decay uygular. Outer validation/test
merkezleri yalnız outer-train modelinden gelir.
ROI noktası `1024`, anatomik ROI yarıçapı çarpanı `1.5` olarak kullanılır. Tüm loss hesapları AMP
dışında float32 yapılır ve sonlu olmayan
batch oranı `%1` değerini aşarsa koşu açık bir hata ile durur. A100 koşuları attention geri
yayılımındaki FP16 taşmalarını önlemek için BF16 AMP kullanır. FP16 fallback durumunda dinamik
loss-scaler taşmaları gerçek `NaN` batchlerden ayrı kaydedilir. Sabit
split `a100` koşusunda val/test için mevcut stacker coarse tahminleri kullanılır; train merkezleri
expert noktalarına yalnız train içinde deterministik sentetik coarse hata eklenerek üretilir. Böylece
in-sample prediction dağılımı kaldırılır. Bununla birlikte mevcut AGH/stacker tahminleri ilk
üretilirken expert-Procrustes kullanıldığı için sabit-split sonuç keşif/ablation niteliğindedir;
makalenin ana sonucu `cv` presetinden alınmalıdır. Stage 2 en az `80` epoch çalışır; scheduler
epoch `60` öncesinde devreye girmez ve ilk `5` epoch lineer LR warmup kullanır.

Mesh veya landmark dosyasının boyutu/değişiklik zamanı, kayıt cache kimliğine dahildir. Bir veri
dosyası değiştiğinde ilgili mesh kaydı, train-only atlas, normalizasyon ve Stage 1 tahminleri eski
cache ile karışmadan yeniden oluşturulur.

Eski `publication_cv_seed*` sonuçlarında train merkezi `expert + synthetic error`, val/test merkezi
ise train template idi. Bu dağılım kayması ve bazı foldlarda görülen `NaN` nedeniyle eski CV klasörü
bilimsel sonuç olarak kullanılmamalıdır. Train-template kullanan `publication_cv_v2_seed*` koşusu
da coarse merkez ablation'ıdır. Yeni fixed-epoch OOF + hybrid ROI sonuçları
`publication_cv_stage1_v4_seed*` altına yazar. `v3` klasörleri geriye dönük inceleme içindir.

Log başlangıcında `Precision: AMP bfloat16` görülmelidir. `Precision: AMP float16` görülüyorsa
çalışılan GPU BF16 desteklemiyordur; preset FP16 için düşük başlangıç scale'i (`1024`) ve ayrı
overflow denetimi kullanır.

## Kontrollü Ablation

```python
!python -u colab_run_all23_rgb_geodesic.py --preset e1
!python -u colab_run_all23_rgb_geodesic.py --preset e2
!python -u colab_run_all23_rgb_geodesic.py --preset e3
!python -u colab_run_all23_rgb_geodesic.py --preset e5
!python -u colab_run_all23_rgb_geodesic.py --preset e6
!python -u colab_run_all23_rgb_geodesic.py --preset e7
!python -u colab_run_all23_rgb_geodesic.py --preset e8
!python -u colab_run_all23_rgb_geodesic.py --preset e9
!python -u colab_run_all23_rgb_geodesic.py --preset e10
```

- `E1`: düzeltilmiş anatomi/veri hattı, geometry-only.
- `E2`: RGB ve lokal renk kontrastı.
- `E3/E4`: global mesh encoder ve all-23 geodezik local refiner.
- `E5`: anatomik graph/simetri loss.
- `E6`: texture/contour/generic specialized heads.
- `E7`: MSE-over-mesh ve validation/test TTA.
- `E8`: confidence-aware coarse/refined gate ve ilk Hard3 bağlam modülleri.
- `E9/FULL`: anatomik anchor kosullu Hard3 surface-candidate ranker, Hard3'e ozel gate,
  candidate CE loss, dengeli checkpoint secimi ve validation-only grup alpha kalibrasyonu.
- `E10`: bagimsiz LM0 RGB/texture-gradient ranker, refine LM10-12 kosullu ortak bilateral
  LM21/22 pair ranker, soft-listwise distance loss, hard-negative mining ve refiner sonrasi
  ayri sample-specific gate egitimi. Geriye donuk yeniden uretilebilirlik icin `FULL`, `E9`
  ile ayni kalir; E10 acikca `--experiment E10` ile secilir.

Üç seed ensemble, E8 model seçimi tamamlandıktan sonra ayrı `ensemble_runs.py` komutuyla uygulanır.

Üç seed tamamlandıktan sonra:

```python
!python -u ensemble_runs.py \
  --run-dirs \
  /content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/full_fixed_v4_seed42 \
  /content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/full_fixed_v4_seed43 \
  /content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/full_fixed_v4_seed44 \
  --output-dir /content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/ensemble
```

## Çıktılar

Her fold/run aşağıdakileri üretir:

```text
best_model.pth
last_model.pth
history.json
best_gate.pth                 # E10
gate_history.json             # E10
gate_training.json            # E10
normalization.json
alignment/alignment_report.json
alignment/train_multimesh_atlas.npz
split_and_leakage_report.json
preprocessing_source.json
refinement_calibration.json
candidate_oracle_samples_pretrain.csv
stage1_global_coarse/complete.json
stage1_global_coarse/metrics_train_val.json
stage1_global_coarse/oof_predictions_train.csv
stage1_global_coarse/predictions_{val,test}.csv
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
`ungated_refined_*` alanları gate öncesi lokal sonucu, `refinement_gate` alanı alpha özetini içerir.
Tahmin CSV'lerinde coarse/refined/final koordinatları ve `refinement_alpha` birlikte saklanır.

## Önemli Bilimsel Not

`--alignment legacy`, expert landmarklarla üretilmiş Procrustes matrislerini kullanır ve yalnız
eski sonuçlarla mühendislik karşılaştırması içindir. Makalenin ana tablosunda yalnız
`--alignment mesh_icp` ve patient-level CV sonucu kullanılmalıdır. `<2 mm` bir hedef ve kabul
kriteridir; kod bu sonucu önceden garanti etmez.
