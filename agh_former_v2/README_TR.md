# AGH-Former v2 — sade 2 aşamalı yapı (Faz 1)

## Neden
`results/publication_cv_seed42` (vNext) çıktılarının aşama bazlı analizinde şunlar görüldü:

- Stage 2 yerel başlığı Core20'yi kötüleştiriyor: fusion 1.995 → local 2.019 mm.
- Stage 4 gate'i işlevsiz: gate'siz aday 1.663, final 1.661 mm.
- Hard3 blend'i sabit α=0.75 kullanıyor ve girdisi iyi olan noktaları bozuyor.
- Darboğaz aday üretmek değil, aday seçmek: ROI oracle Core20'de 0.78, Gonion'da 1.02 mm.

v2 bu parçaları **23 noktanın tamamı için tek bir çok-görüntülü yerel düzeltici** ile değiştirir.

## Yapı
```
Aşama A (global) : Stage-1 OOF merkezleri → vNext Stage-2 fusion çıktısı (arşivden, "coarse_*")
                   → Procrustes şekil önceli (train'de fit, val'de kalibrasyon)
Aşama B (yerel)  : All23-MVSC × {1,2} iterasyon (ROI her iterasyonda öneriye yeniden merkezlenir)
                   → OOF gate: p = p_in + α·(p_öneri − p_in), α tahmin bazında
                   → tip bazlı α üst sınırı + iterasyon sayısı val'de seçilip kilitlenir
```

| Bileşen | Dosya | vNext'e göre fark |
|---|---|---|
| Düzeltici | `all23_mvsc_refinement/model.py` | 23 nokta; köşe / kontur / doku-sınırı tip başlıkları; iç gate yok |
| Görüntüler | `all23_mvsc_refinement/patches.py` | 3 ince görünüm + ROI'yi **tamamen kapsayan** bağlam görünümü, görünürlük bitleri, silüet ve derinlik-gradyanı kanalları, Gonion için çene profili |
| Uzamsal prior | `all23_mvsc_refinement/spatial_prior.py` | Her hedef diğer Core20 noktalarından tahmin edilir; Hard3 bağlam olarak kullanılmaz |
| Gate | `all23_mvsc_refinement/gate.py` | Cross-fit OOF kanıtla eğitilir; kayıp doğrudan Öklid hatası |
| Şekil önceli | `all23_mvsc_refinement/shape_prior.py` | Generalized Procrustes hizalaması, sonra PCA ve ridge |
| Runner | `agh_former_v2/run_agh_v2.py` | Validasyon kilidi; test ROI'leri etiketsiz kurulur, test etiketleri tahminler yazıldıktan sonra okunur |

## Sızıntı ve kilit kuralları
- **Düzeltici, uzamsal prior, şekil önceli ve gate kanıtı:** yalnızca outer-train katılımcılarıyla eğitilir.
- **Gate kanıtı:** her outer-train katılımcısı, kendisini görmemiş bir iç düzelticiyle (`--all23-gate-folds`, varsayılan 5) ve kendi Stage-1 OOF merkezinden başlanarak üretilir.
- **Outer validasyon:** yalnızca şekil önceli kabulü, iterasyon sayısı ve tip bazlı α üst sınırı için kullanılır. Seçilen politika `v2_lock.json` içinde hash'lenir.
- **`--final-test`:** kilitli politika, şekil önceli seçimi ve 5 fold'luk validasyon kapısı yoksa çalışmaz.
- **Test akışı:**
  - ROI'ler `.geometry` modunda kurulur.
  - Tahminler önce `test_predictions_before_scoring.npz` dosyasına yazılır.
  - Ancak bundan sonra landmark dosyaları okunur.
- **Fold'lar:** `shared_splits/orthodontic_5fold_192_48_60_seed42.json` ile birebir aynı olduğu her fold'da doğrulanır.

## Çalıştırma (Colab)
`colab_agh_v2_tr.ipynb` defterindeki hücreleri sırayla çalıştırın:
1. `--preset v2_smoke`: Fold 1, 2 epoch, yalnızca validasyon.
2. `--preset v2_val_all_folds`: 5 fold'un validasyon bölümleri. Toplam 240 validasyon tahmini var; ama bölümler her fold içinde ayrı seçildiği için bunlar 184 farklı kişiye ait. Sonunda `validation_gate.json` yazılır.
3. `--preset v2_final_test`: yalnızca kapı geçtiyse çalışır, testi bir kez koşar.

**Validasyon kapısı** (vNext'in arşivdeki validasyon tahminleriyle aynı 240 fold-tahmini üzerinde; bootstrap aynı kişinin tüm satırlarını birlikte örnekler):
- all-23 ALE ≤ vNext,
- Hard3 kötüleşmesi ≤ 0.10 mm,
- Core20 kötüleşmesi ≤ 0.05 mm.

## Yerel test
```bash
palnet_orthodontic_comparison/.venv/bin/python -m pytest all23_mvsc_refinement/tests agh_former_v2/tests
```

## Bilinen sınırlar
- **Gate kanıtında dağılım farkı:** kanıt, Stage-1 OOF merkezlerinden başlıyor. Uygulamada ise girdi Aşama A + şekil önceli; bu girdi genelde daha iyi olduğu için gate muhafazakâr kalabilir.
- **Aşama A arşivden geliyor:** Faz 1'de Aşama A, vNext'in birlikte eğitilmiş Stage 2 modelinin fusion çıktısıdır. Local başlık olmadan yeniden eğitilmesi Faz 2'ye bırakıldı.
- **Hesaplama maliyeti:** outer fold başına 6 düzeltici eğitimi gerekir (5 iç + 1 tam). Görüntüler `--scratch-dir` altında memmap olarak tutulur ve eğitim sonrası silinir.
