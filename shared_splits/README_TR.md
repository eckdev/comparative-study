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
