# Google Colab GPU Kullanimi

Bu klasor DiffusionNet landmark lokalizasyonunu Google Colab GPU uzerinde calistirmak icin hazirlandi.

## Makale Icin 5-Fold CV

Kod ve Drive baglantisi hazirlandiktan sonra once fold ve label-free hizalama
provenance kontrolunu calistirin:

```python
%cd /content/comparative-study/diffusion_net_orthodontic_comparison
!python -u colab_run_diffusionnet_cv.py --preset preflight --seed 42
```

Kisa teknik test:

```python
!python -u colab_run_diffusionnet_cv.py --preset smoke --seed 42
```

Ana A100 kosusu:

```python
!python -u colab_run_diffusionnet_cv.py --preset cv --seed 42
```

Colab oturumu kesilirse ayni ana kosu komutunu tekrar calistirin. Tamamlanan
foldlar atlanir; yarim kalan fold son tamamlanan epochtan devam eder. Foldlari
ayri oturumlarda calistirmak icin `--fold-indices 1`, daha sonra
`--fold-indices 2` seklinde ilerlenebilir.

Ana ozet dosyasi:

```text
/content/drive/MyDrive/orthodontic/diffusion_runs/
  diffusionnet_publication_cv_seed42/summary_metrics.json
```

## Dosyalar

- `run_orthodontic_diffusion.py`: Colab/yerel ortak egitim scripti.
- `colab_diffusionnet_orthodontic_gpu.ipynb`: Colab uzerinde hucre hucre calistirilacak notebook.
- `requirements.txt`: Python bagimliliklari.

## Colab veri yerlesimi

Dataset ve ciktilar GitHub'a eklenmedigi icin Google Drive uzerinde tutulmalidir. Onerilen yapi:

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
    diffusion_runs/
```

`dataset` klasoru mevcut yerel `data/dataset` klasoru ile ayni formatta olmalidir.

## Onerilen Colab calistirma

Notebook icindeki `Pratik GPU kosusu` hucreleri standart Colab T4/L4 icin daha uygundur:

```bash
python run_orthodontic_diffusion.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --splits-json /content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json \
  --transformation-dir /content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir /content/drive/MyDrive/orthodontic/diffusion_runs/diffusionnet_maskseg_xyz_p4096_k64_w128_b6_e120 \
  --surface-points 4096 \
  --k-eig 64 \
  --epochs 120 \
  --patience 25 \
  --width 128 \
  --blocks 6 \
  --mlp-hidden-dims 256 \
  --loss-mode mask_bce \
  --mask-radius 3.5 \
  --input-features xyz \
  --postprocess softmax \
  --device auto
```

Paper'a daha yakin tam ayar icin `use_mesh_vertices`, `width=384`, `blocks=12`, `mlp_hidden_dims=768`, `epochs=200` kullanilir. Bu ayar 24 GB ve ustu GPU bellegi icin daha uygundur.

## Ortak Split

Notebook ve komutlar `SPLITS_JSON = /content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json` dosyasini kullanir. Bu dosya PAL-Net, DiffusionNet ve PointNet++ icin ayni 180/60/60 hasta listesini sabitler.

## Beklenen fark

Yerelde CPU ile pratik paper-inspired kosuda 2048 nokta ile ALE yaklasik `3.7640` bulundu. Colab GPU'da daha fazla nokta, daha genis model ve daha uzun egitim ile bunun iyilesmesi beklenir; ancak tam paper ayarlari veri boyutu, VRAM ve Colab oturum suresine baglidir.
