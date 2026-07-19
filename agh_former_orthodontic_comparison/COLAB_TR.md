# Google Colab Pro Kullanımı

## 1. Drive Bağlantısı

```python
from google.colab import drive
drive.mount('/content/drive')
```

GitHub repo ve dataset ayrı konumdadır. Bu runner varsayılan olarak şu yolları kullanır:

```text
CODE_ROOT     = /content/comparative-study
DATA_ROOT     = /content/drive/MyDrive/orthodontic/data/dataset
SPLITS_JSON   = /content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json
TRANSFORM_DIR = /content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801
RUN_ROOT      = /content/drive/MyDrive/orthodontic/diffusion_runs
```

`TRANSFORM_DIR` varsa Procrustes transformları kullanılır; yoksa script raw koordinatlarla devam eder.

## 2. Kurulum

Notebook ile çalışmak için:

```text
agh_former_orthodontic_comparison/colab_aghformer_orthodontic_gpu.ipynb
```

Manuel komutlarla çalışmak için:

```bash
cd /content/comparative-study
pip install -r agh_former_orthodontic_comparison/requirements.txt
```

## 3. Smoke Test

```bash
cd /content/comparative-study/agh_former_orthodontic_comparison
python -u colab_run_aghformer_shared_metrics.py \
  --preset smoke
```

## 4. Ana A100 Koşusu

```bash
cd /content/comparative-study/agh_former_orthodontic_comparison
python -u colab_run_aghformer_shared_metrics.py \
  --preset a100
```

## 5. Büyük Bellek Koşusu

```bash
python -u colab_run_aghformer_shared_metrics.py \
  --preset a100_16k
```

## 6. Notlar

- Logların akması için `python -u` kullanılmalıdır.
- İlk epoch yavaş olabilir; mesh örnekleme ve lokal geometri cache dosyaları oluşturulur.
- `best_model.pth` güncelleniyorsa validasyon ALE bakımından daha iyi bir checkpoint bulunmuştur.
- Nihai karşılaştırmada `metrics.json` içindeki `aghformer_snapped.ale` ana değer olarak kullanılmalıdır; direct residual modelin örnekleme bağımsız kontrolü için `aghformer_raw.ale` de raporlanmalıdır.
- Çıktılar varsayılan olarak `/content/drive/MyDrive/orthodontic/diffusion_runs` altına yazılır.
- Güncel iyileştirilmiş preset çıktıları `aghformer_v2_template_*` adıyla yazılır; önceki 8-9 mm koşuları ezilmez.
