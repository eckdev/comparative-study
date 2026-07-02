# PointNet++ Google Colab Pro Kullanimi

Bu dosya mevcut PointNet++ modelini degistirmeden Google Colab Pro GPU uzerinde calistirmak icin hazirlandi. DiffusionNet sonuclari sabit tutulacaksa, bu kosular PointNet++ karsilastirma sonucunu GPU ortaminda yeniden uretmek icindir.

## Dosyalar

- `run_orthodontic_pointnet2.py`: Yerel/Colab ortak egitim scripti.
- `colab_pointnet2_orthodontic_gpu.ipynb`: Colab uzerinde hucre hucre calistirilacak notebook.
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
    pointnet2_runs/
```

`dataset` klasoru yereldeki `data/dataset` ile ayni formatta olmalidir. Transform klasoru yoksa `USE_TRANSFORMS = False` yapilabilir; fakat model karsilastirmasinda PAL-Net ve DiffusionNet ile ayni hizalanmis veri kullanmak icin transform kullanilmasi onerilir.

## Onerilen ana kosu

Colab Pro GPU icin mevcut PointNet++ v2 protokolunu kullan:

```bash
python run_orthodontic_pointnet2.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --transformation-dir /content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir /content/drive/MyDrive/orthodontic/pointnet2_runs/pointnet2_v2_gaussian_normals_p4096_e200 \
  --surface-points 4096 \
  --eval-surface-points 4096 \
  --sa1-points 1024 \
  --sa2-points 256 \
  --sa3-points 64 \
  --nsample 32 \
  --epochs 200 \
  --patience 40 \
  --batch-size 4 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --optimizer adamw \
  --scheduler cosine \
  --target-mode gaussian \
  --heatmap-sigma 3.5 \
  --use-normals \
  --coord-weight 0.25 \
  --coord-temperature 1.0 \
  --postprocess topk_softmax \
  --topk 10 \
  --temperature 1.0 \
  --device auto
```

Bellek hatasi alirsan sirasiyla:

1. `--batch-size 2`
2. `--surface-points 2048`
3. `--sa1-points 512 --sa2-points 128 --sa3-points 32`

## Daha buyuk Colab Pro denemesi

A100 veya yuksek VRAM varsa:

```bash
python run_orthodontic_pointnet2.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --transformation-dir /content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir /content/drive/MyDrive/orthodontic/pointnet2_runs/pointnet2_v2_gaussian_normals_p8192_e220 \
  --surface-points 8192 \
  --eval-surface-points 8192 \
  --sa1-points 2048 \
  --sa2-points 512 \
  --sa3-points 128 \
  --nsample 32 \
  --epochs 220 \
  --patience 45 \
  --batch-size 2 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --optimizer adamw \
  --scheduler cosine \
  --target-mode gaussian \
  --heatmap-sigma 3.5 \
  --use-normals \
  --coord-weight 0.25 \
  --coord-temperature 1.0 \
  --postprocess topk_softmax \
  --topk 10 \
  --temperature 1.0 \
  --device auto
```

## Evaluate-only postprocess kontrolu

Egitim bittikten sonra farkli postprocess ayarlari ayni checkpoint uzerinde denenebilir:

```bash
python run_orthodontic_pointnet2.py \
  --data-root /content/drive/MyDrive/orthodontic/data/dataset \
  --transformation-dir /content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801 \
  --output-dir /content/drive/MyDrive/orthodontic/pointnet2_runs/pointnet2_eval_topk20_t1 \
  --surface-points 4096 \
  --eval-surface-points 4096 \
  --sa1-points 1024 \
  --sa2-points 256 \
  --sa3-points 64 \
  --nsample 32 \
  --batch-size 4 \
  --target-mode gaussian \
  --heatmap-sigma 3.5 \
  --use-normals \
  --coord-weight 0.25 \
  --postprocess topk_softmax \
  --topk 20 \
  --temperature 1.0 \
  --device auto \
  --evaluate-only \
  --model-path /content/drive/MyDrive/orthodontic/pointnet2_runs/pointnet2_v2_gaussian_normals_p4096_e200/best_model.pth
```

## Beklenen ciktilar

Her run klasorunde:

- `metrics.json`: ALE, median, landmark bazli hata.
- `history.json`: epoch bazli train/val ve learning rate.
- `predictions_test.csv`: uzman ve PointNet++ tahmin koordinatlari.
- `group_metrics_test.csv`: class/cinsiyet bazli ALE.
- `best_model.pth`: en iyi validation ALE checkpoint.

## Mevcut yerel referans

Yerel CPU-dostu v2 kosusu:

```text
Run: pointnet2_v2_gaussian_normals_p1024_e60
ALE: 3.9706
Median: 3.3463
```

Colab Pro kosusu, modeli degistirmeden daha yogun nokta sayisi ve daha uzun egitim ile PointNet++ icin nihai karsilastirma sonucunu uretmek amaciyla kullanilmalidir.
