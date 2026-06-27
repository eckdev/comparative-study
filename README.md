# comparative-study

PAL-Net tabanli 3B ortodontik landmark lokalizasyonu ve uzman ortodontist isaretlemeleriyle Average Localization Error (ALE) karsilastirmasi.

Bu repo kodlari, egitim/evaluasyon scriptlerini ve rapor uretim araclarini icerir. Hasta dataseti, model ciktilari, PDF raporlari, run klasorleri ve transform matrisleri GitHub'a dahil edilmez.

## Icerik

- `palnet_orthodontic_comparison/README_TR.md`: Proje akisi ve kullanim notlari.
- `palnet_orthodontic_comparison/requirements.txt`: Python bagimliliklari.
- `palnet_orthodontic_comparison/upstream/`: PAL-Net kaynak kodu ve ortodontik veri seti icin eklenen custom scriptler.

## Disarida Birakilanlar

- `data/`: Hasta meshleri ve landmark dosyalari.
- `output/`: PDF ve render ciktilari.
- `palnet_orthodontic_comparison/runs/`: Egitim sonuclari, model agirliklari ve tahmin dosyalari.
- `palnet_orthodontic_comparison/transforms/`: Uretilmis Procrustes transform matrisleri.

