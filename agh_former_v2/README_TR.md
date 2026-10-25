# AGH-Former v2 / v2.1 — tek yerel düzelticili kaskad

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

## v2.1 (güncel aday)

### Neden?
v2'nin 5 fold'luk validasyon tanı raporu (`validation_diagnostics/diagnostics_val.md`) iki bulgu verdi:
- **Core20:** v2, vNext'ten güvenilir şekilde daha iyi: −0.066 mm (GA −0.088 … −0.045).
- **Hard3:** v2 daha kötü: +0.63 mm. Kaybın ana kaynağı Aşama A seçimi. v2, vNext Stage 2'nin *fusion* merkezinden başlıyordu. Oysa Stage 2 yerel başlığı ve gate'i Hard3'ü validasyonda 6.10 → 5.24 mm'ye indiriyor. v1 tasarımında bu başlık yalnızca Core20 sonuçlarına bakılarak gereksiz sayılmıştı; bu yanlış bir değerlendirmeydi.

### Değişiklikler
**Model ve kaskad**
- **Aşama A:** vNext Stage 2'nin gate'li çıktısı (`--stage-a-column neural_prediction`). Kaskad: Stage 1 → Stage 2 → Procrustes şekil önceli → tek All23 düzeltici → OOF gate. vNext'in Hard3 ve Core20 uzmanları ile eski şekil önceli kullanılmıyor; çıktılar yan yana dikilmiyor.
- **Eğitim yok:** v2'nin eğittiği 30 düzeltici (fold başına 5 iç + 1 tam) `--refiner-source-run` ile değiştirilmeden yeniden kullanılır. Eğitim imzası sabittir ve bir testle korunur (`test_trained_refiner_signature_is_stable`).
- **Ensemble:** val/test önerisi, tam modelin ve 5 iç modelin önerilerinin ortalamasıdır. Hiçbiri val/test kişilerini görmedi.

**Gate**
- İki yeni sinyal eklendi:
  - MC-dropout yayılımı (tam model, 8 geçiş, sabit tohum);
  - anatomik tutarsızlık: orta hat noktasının orta-sagital düzleme uzaklığı, bilateral noktanın aynalanmış eşine uzaklığı. Düzlem gelen tahminden kurulur, uzman noktalarından değil.
- Gate kanıtının girdisi, Stage-1 OOF merkezlerine validasyonda seçilen şekil önceli ayarı uygulanarak üretilir. Önceli her iç fold'da yalnızca o fold'un eğitim kişileri üzerinde yeniden fit edilir.

**Politika ve kilit**
- İterasyon sayısı tip bazında seçilir. v2'de 2. iterasyon Hard3'ü kötüleştiriyordu.
- `locked_validation_predictions.npz` kaydedilir. Final test koşusu validasyonu yeniden hesaplar; sonuç 0.01 mm'den fazla sapıyorsa koşu durur.

### Çalıştırma
Notebook'taki v2.1 hücrelerini çalıştırın. GPU yalnızca çıkarım ve gate kanıtı için kullanılır; eğitim yok.
1. `--preset v21_val_all_folds`
2. `--preset v21_diagnostics`
3. `--preset v21_final_test`: yalnızca kapı geçerse çalışır.

Kapı ölçütleri v2 ile aynıdır ve v2.1 sonuçları görülmeden önce sabitlenmiştir:
- all-23 ALE ≤ vNext,
- Hard3 ≤ vNext + 0.10 mm,
- Core20 ≤ vNext + 0.05 mm.

### Not
Validasyon kümesi geliştirme sırasında birden çok kez kullanıldı (v2 → tanı → v2.1). Bu yüzden validasyon sonuçları tarafsız bir tahmin değildir. Tarafsız tek ölçüm, kapı geçtikten sonra yalnızca bir kez açılan testtir. Bu geliştirme geçmişi makalede raporlanmalıdır.

## Final model: gate'siz v2 (2026-10-21)
- Karar, gerekçe, geliştirme geçmişi ve raporlanacak sonuçlar [PREREGISTRATION_TR.md](PREREGISTRATION_TR.md) dosyasında. Bu dosya testten önce commit edildi.
- **Model:** v2.1'in güven kapısı, cross-fit gate kanıtı ve MC-dropout olmadan çalışan hali (`--no-all23-gate`). Politika yalnızca her tip için iterasyon sayısını ve düzeltmenin açık/kapalı olmasını seçer.
- **Colab sırası:**
  1. `v2final_val_all_folds`: yalnızca çıkarım.
  2. `v2final_diagnostics`
  3. `v2final_commit`: `final_model_decision.json` yazar.
  4. `v2final_final_test`: bir kez çalışır.
  5. `aggregate_cv_results`
- **Yeniden üretim kontrolü:** `v2final_val_all_folds` sonucu, karar anındaki gate'siz validasyon değerlerine (all-23 ≈ 1.994, Core20 ≈ 1.578, Hard3 ≈ 4.762) çok yakın olmalı. Belirgin bir fark varsa `v2final_commit` çalıştırılmamalı ve önce nedeni bulunmalı.

## Değişiklik: sınıf bilgisi olmadan (2026-10-25)
- Uzamsal prior yalnızca cinsiyeti eş değişken olarak kullanır (`--all23-prior-covariates sex`). Eğitilmiş modeller aynen kullanılır; yalnızca prior yeniden fit edilir.
- Bu değişiklik **testten sonra** yapıldı. Ayrıntı ve dürüstlük notu: [PREREGISTRATION_TR.md](PREREGISTRATION_TR.md) → "Değişiklik kaydı".
- Colab preset sırası: `v2final_noclass_val_all_folds` → `v2final_noclass_diagnostics` → `v2final_noclass_commit` → `v2final_noclass_final_test` → aggregate.
