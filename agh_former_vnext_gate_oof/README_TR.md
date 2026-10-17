# AGH-Former vNext Gate-OOF

Bu deney, mevcut bes-fold AGH-Former vNext checkpoint'lerini degistirmeden Stage 2
yerel duzeltme gate'inin gercek katkisini olcer. Makalenin `q1_manuscript/`
klasoru bu kod surumunun Git push kapsaminda degildir.

## Calistirma (Colab A100)

Drive'i baglayin ve mevcut AGH-vNext CV ile all23 Stage 1 preprocessing
klasorlerinin yukaridaki Colab varsayilan yollarinda bulundugunu dogrulayin.
Repository'de veri yoktur; Drive'daki `orthodontic/data/dataset` kullanilir.

```python
%cd /content/comparative-study/agh_former_vnext_gate_oof
!python -u colab_run_gate_oof.py --preset gate_oof_preflight --seed 42
!python -u colab_run_gate_oof.py --preset gate_oof_fold1 --seed 42
```

`fold_1/selection_policy.json` icindeki `run_full_cv` true ise:

```python
!python -u colab_run_gate_oof.py --preset gate_oof_cv --seed 42
```

Ayni komutla yeniden baslatmak tamamlanan inner fold'lari atlar; yarida
kalan Stage 1/Stage 2 egitimi kendi checkpoint'lerinden devam eder. Kaynak
run klasorleri salt okunur; yeni dosyalar Drive'daki
`agh_vnext_gate_oof_runs/publication_gate_oof_seed42/` altina yazilir.

## Bilimsel yorum

`historical_gate_audit.json`, eski bes-fold test tahminlerini yeniden olcer;
bu yeni bir test ya da bagimsiz dogrulama degildir. Inner fold'da Stage 1
merkezleri yalniz inner-train'den, Stage 2 modeli inner-train/validation'dan
ogrenilir. Inner holdout uzman etiketleri ancak tahmin dosyasi yazildiktan
sonra okunur. OOF gate yalniz bes tanisal sinyal ve landmark kimligini kullanir;
ayri inner modellerin token vektorleri birbiriyle karistirilmaz.

Fold 1 validation pilotunda Hard3 en az 0.10 mm iyilesmezse bes-fold test
kosusu bloke edilir. Bes-fold sonucunda Hard3 kazanci en az 0.20 mm ve
eslestirilmis guven araligi pozitif degilse gate ana mekanizma olarak
sunulmaz. Bu protokol hala *internal development cross-validation*'dir;
bagimsiz dis kohort degerlendirmesi degildir. Yeni gate sadece Stage 2
ciktilarinda sinanir. Sonraki shape prior/Hard3/Core20 zincirine entegrasyon
ayri bir validation kilidi ve yeni test gerektirir.
