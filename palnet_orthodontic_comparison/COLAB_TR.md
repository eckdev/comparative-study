# PAL-Net Google Colab Pro Kullanimi

Bu dosyalar PAL-Net adaptasyonunu Google Colab GPU uzerinde calistirmak icin hazirlandi. DiffusionNet ve PointNet++ ile adil karsilastirma icin ortak split dosyasi kullanilir.

## Dosyalar

- `upstream/run_orthodontic.py`: Yerel/Colab ortak egitim scripti.
- `colab_palnet_orthodontic_gpu.ipynb`: Colab uzerinde hucre hucre calistirilacak notebook.
- `requirements.txt`: Python bagimliliklari.

## Google Drive yapisi

Dataset ve transform klasorleri GitHub'a eklenmedigi icin Google Drive uzerinde tutulmalidir. Onerilen yapi:

```text
MyDrive/
  orthodontic/
    data/
      dataset/
        Class1/
        Class2/
        Class3/
    transforms/
      orthodontic_procrustes_rigid_20260627_143801/
    palnet_runs/
```

`dataset` klasoru yereldeki `data/dataset` ile ayni formatta olmalidir. Transform klasoru yoksa notebook'ta `USE_TRANSFORMS = False` yapilabilir; fakat model karsilastirmasinda ayni hizalanmis veri protokolu icin transform kullanilmasi onerilir.

## Ortak Split

Notebook ve komutlar repo icindeki ortak split dosyasini kullanir:

```bash
--splits-json /content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json
```

Bu split 300 hastayi sinif/cinsiyet dengeli olarak 180 egitim, 60 validasyon ve 60 test hastasina ayirir.

## Smoke Test

Once kodun ve veri yollarinin dogru calistigini gormek icin:

```bash
python -u run_orthodontic.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --splits-json /content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json \
  --transformation-dir /content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir /content/drive/MyDrive/orthodontic/palnet_runs/palnet_smoke_colab \
  --epochs 2 \
  --patience 2 \
  --batch-size 2 \
  --patch-size 100 \
  --surface-points 5000 \
  --snap-k 1
```

## Colab Pro Pratik Kosu

T4/L4 gibi GPU'larda daha uygulanabilir PAL-Net kosusu:

```bash
python -u run_orthodontic.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --splits-json /content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json \
  --transformation-dir /content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir /content/drive/MyDrive/orthodontic/palnet_runs/palnet_procrustes_p500_surface50k_e120 \
  --epochs 120 \
  --patience 30 \
  --batch-size 4 \
  --patch-size 500 \
  --surface-points 50000 \
  --lr 0.001 \
  --snap-k 1 \
  --model PALNET \
  --loss combined
```

## A100 / Yuksek VRAM Buyuk Kosu

Paper'a daha yakin yogun ayar icin:

```bash
python -u run_orthodontic.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --splits-json /content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json \
  --transformation-dir /content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir /content/drive/MyDrive/orthodontic/palnet_runs/palnet_procrustes_p1000_surface100k_e200 \
  --epochs 200 \
  --patience 40 \
  --batch-size 2 \
  --patch-size 1000 \
  --surface-points 100000 \
  --lr 0.001 \
  --snap-k 1 \
  --model PALNET \
  --loss combined
```

Bellek hatasi alirsan sirasiyla `--batch-size 1`, `--surface-points 50000`, `--patch-size 500` deneyebilirsin.

## Beklenen Ciktilar

Her run klasorunde:

- `metrics.json`: `palnet_raw`, `palnet_snapped` ve baseline ALE ozetleri.
- `history.json`: epoch bazli train/validation loss.
- `predictions_test.csv`: uzman ve PAL-Net tahmin koordinatlari.
- `group_metrics_test.csv`: class/cinsiyet bazli ALE.
- `splits.json`: ortak split kaynagini gosteren run-local split kaydi.
- `best_model.pth`: en iyi validation loss checkpoint.
