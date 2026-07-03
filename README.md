# comparative-study

PAL-Net, DiffusionNet ve PointNet++ tabanli 3B ortodontik landmark lokalizasyonu modellerinin uzman ortodontist isaretlemeleriyle Average Localization Error (ALE) karsilastirmasi.

Bu repo kodlari, egitim/evaluasyon scriptlerini ve rapor uretim araclarini icerir. Hasta dataseti, model ciktilari, PDF raporlari, run klasorleri ve transform matrisleri GitHub'a dahil edilmez.

## Icerik

- `palnet_orthodontic_comparison/README_TR.md`: Proje akisi ve kullanim notlari.
- `diffusion_net_orthodontic_comparison/README_TR.md`: DiffusionNet adaptasyonu ve Colab GPU kullanim notlari.
- `pointnet2_orthodontic_comparison/README_TR.md`: PointNet++ adaptasyonu ve Colab GPU kullanim notlari.
- `shared_splits/`: Uc model icin ortak 180 egitim, 60 validasyon, 60 test hasta listesi.
- `palnet_orthodontic_comparison/requirements.txt`: Python bagimliliklari.
- `palnet_orthodontic_comparison/upstream/`: PAL-Net kaynak kodu ve ortodontik veri seti icin eklenen custom scriptler.

## Ortak Deney Split'i

Adil karsilastirma icin tum modeller ayni dosya listeleriyle calistirilmalidir:

```bash
--splits-json shared_splits/orthodontic_180_60_60_seed42.json
```

Bu split 300 hastayi sinif/cinsiyet dengeli olarak 180 egitim, 60 validasyon ve 60 test hastasina ayirir.

## Disarida Birakilanlar

- `data/`: Hasta meshleri ve landmark dosyalari.
- `output/`: PDF ve render ciktilari.
- `*/runs/`: Egitim sonuclari, model agirliklari ve tahmin dosyalari.
- `palnet_orthodontic_comparison/transforms/`: Uretilmis Procrustes transform matrisleri.
