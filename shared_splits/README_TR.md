# Ortak Dataset Split'i

Adil model karsilastirmasi icin PAL-Net, DiffusionNet ve PointNet++ ayni hasta dosyalariyla calistirilmalidir. Bu klasordeki JSON dosyasi tek ortak split kaynagidir:

- Egitim: 180 hasta
- Validasyon: 60 hasta
- Test: 60 hasta
- Stratifikasyon: `class_name + gender`

Split dosyasini yeniden uretmek icin:

```bash
python shared_splits/create_common_splits.py \
  --data-root data/dataset \
  --output shared_splits/orthodontic_180_60_60_seed42.json \
  --seed 42
```

Model egitimlerinde ayni dosyayi kullan:

```bash
--splits-json shared_splits/orthodontic_180_60_60_seed42.json
```

PAL-Net scripti `palnet_orthodontic_comparison/upstream` klasorunden calistirilirse goreli yol:

```bash
--splits-json ../../shared_splits/orthodontic_180_60_60_seed42.json
```

## Makale Icin Ana 5-Fold Protokolu

Ana akademik karsilastirma `orthodontic_5fold_192_48_60_seed42.json`
manifestini kullanir. Bu manifest AGH-Former vNext kosusundaki dis foldlarla
birebir aynidir:

- Her fold: 192 train, 48 validation, 60 test.
- Her test folduna her class-cinsiyet tabakasindan 10 ornek girer.
- Her ornek bes dis test foldunun tam olarak birinde bulunur.
- Split anahtari `sample_id`'dir.

Manifesti ayni algoritmayla yeniden olusturmak icin:

```bash
python -u shared_splits/create_cv_splits.py \
  --data-root data/dataset \
  --output shared_splits/orthodontic_5fold_192_48_60_seed42.json \
  --seed 42
```

Makalede PAL-Net, DiffusionNet, PointNet++ ve AGH-Former vNext bu manifestteki
ayni kimliklerle calistirilmalidir. Eski 180/60/60 kosusu yalniz tamamlayici
deney olarak raporlanmalidir.
