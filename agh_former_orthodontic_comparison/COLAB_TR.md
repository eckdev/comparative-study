# Google Colab Pro Kullanımı

## 1. Drive Bağlantısı

```python
from google.colab import drive
drive.mount('/content/drive')
```

Repo ve dataset Drive üzerinde şu yapıda olmalıdır:

```text
/content/drive/MyDrive/comparative-study/
  data/dataset/
  shared_splits/
  palnet_orthodontic_comparison/transforms/
  agh_former_orthodontic_comparison/
```

## 2. Kurulum

```bash
cd /content/drive/MyDrive/comparative-study
pip install -r agh_former_orthodontic_comparison/requirements.txt
```

## 3. Smoke Test

```bash
cd /content/drive/MyDrive/comparative-study/agh_former_orthodontic_comparison
python -u colab_run_aghformer_shared_metrics.py \
  --preset smoke
```

## 4. Ana A100 Koşusu

```bash
cd /content/drive/MyDrive/comparative-study/agh_former_orthodontic_comparison
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
- Nihai karşılaştırmada `metrics.json` içindeki `aghformer_snapped.ale` ana değer olarak kullanılmalıdır.
