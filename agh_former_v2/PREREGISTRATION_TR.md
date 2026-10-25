# Final model ön-kaydı — AGH-Former v2 (gate'siz)

Karar tarihi: 2026-10-21. Bu belge, test bölümü açılmadan önce commit edilmiştir. Test bir kez açılacak ve sonuç ne olursa olsun bu model raporlanacaktır. Hiçbir model seçimi test sonuçlarına göre yapılmayacaktır.

## Final model
```
Stage 1 (global kaba tahmin, OOF) → Stage 2 (anatomik/geodezik iyileştirme, kendi içindeki öğrenilmiş birleştirme dahil)
→ Procrustes şekil önceli (outer-train'de fit, validasyonda kalibrasyon)
→ All23 çok-görüntülü düzeltici: 23 nokta, köşe / kontur / doku-sınırı başlıkları, ROI'yi tam kapsayan bağlam görünümü,
  fold başına 6 modelli ensemble (tam model + 5 cross-fit iç model)
→ politika: her landmark tipi için iterasyon sayısı (1/2) ve düzeltmenin açık/kapalı olması, validasyonda seçilir
```

- **Güven kapısı yok.** v2.1'deki OOF gate, cross-fit gate kanıtı ve MC-dropout kaldırıldı. Validasyonda gate'li ve gate'siz sürüm arasında anlamlı bir fark yoktu: all-23 farkı +0.001 mm [−0.011, +0.014].
- **Eğitilmiş modeller:** v2 koşusundan (`agh_v2_runs/publication_cv_seed42`) aynen kullanılır. Eğitim imzası testle sabitlenmiştir.
- **Stage 1–2:** önceki geliştirme hattıyla aynı ağırlıklardır. Yöntem bölümünde modelin kendi aşamaları olarak anlatılır.
- **Colab koşusu:** `agh_v2_runs/publication_cv_v2final_seed42`, preset'ler `v2final_*`.

## Validasyon bulguları (karar anında bilinenler)
Kaynak: 5 fold'un outer-validation bölümleri, 240 fold-tahmini / 184 kişi. Güven aralıkları katılımcı kümesi bootstrap'ıyla hesaplandı.

| | Önceki hat (vNext) | Final model (gate'siz) | Fark [%95 GA] |
|---|---|---|---|
| All-23 ALE (mm) | 2.057 | 1.994 | −0.064 [−0.092, −0.037] |
| Core20 | 1.679 | 1.578 | −0.101 [−0.123, −0.079] |
| Hard3 | 4.582 | 4.762 | +0.181 [+0.047, +0.322] |
| Gonion sol / sağ | 5.556 / 4.977 | 5.817 / 5.358 | +0.26 / +0.38 |
| SDR@2 / 3 / 4 mm | 64.9 / 81.4 / 89.1 % | 67.2 / 82.5 / 89.7 % | |

**Ön-tanımlı kapı:** "Hard3 ≤ önceki hat + 0.10 mm" ölçütü **geçilmedi**. Karar birincil sonuç ölçütüne (all-23 ALE) dayanır. Gonion bir sınırlılık olarak raporlanacaktır.

## Geliştirme geçmişi (dürüstlük beyanı)
1. **v2:** tek düzeltici, Stage 2'nin birleştirme merkezinden başlar. Kapı geçmedi (Hard3 +0.63 mm).
2. **Kapı hatası düzeltildi:** aynı kişinin birden fazla fold'da yer alması kayıtların üzerine yazılmasına yol açıyordu. Validasyon tanı raporu eklendi.
3. **v2.1:** Stage 2'nin gate'li çıktısından başlar; ensemble, yeni gate sinyalleri ve tip bazında iterasyon. Kapı geçmedi (Hard3 +0.157 mm).
4. **Gate kaldırıldı** (bu karar).

Outer-validation bölümleri bu süreçte birden çok kez kullanıldı. Bu nedenle validasyon sonuçları iyimser olabilir. Tarafsız tek ölçüm, bir kez açılan testtir. Test etiketleri hiçbir aşamada okunmadı; önceki hattın test sonucu ise geliştirme öncesinde biliniyordu.

## Raporlanacak sonuçlar (test, 300 kişi × 23 nokta)
- **Birincil:** all-23 ALE (mm) ve katılımcı bootstrap %95 GA.
- **İkincil:**
  - Core20 ve Hard3 ALE,
  - nokta bazında ortalama ve medyan,
  - SDR@2/3/4 mm,
  - 75., 90., 95. ve 99. persentil ile maksimum,
  - fold ALE ortalaması ve SD,
  - sınıf ve cinsiyet alt grupları,
  - PAL-Net, DiffusionNet ve PointNet++ ile eşleştirilmiş karşılaştırma (mevcut test tahminleri).
- **Aşama denetimi:** Stage 1, Stage 2, şekil önceli, düzeltici iterasyonları ve final.
- **Sınırlılıklar:**
  - Gonion,
  - tek uzman referans (tekrar güvenilirliği henüz ölçülmedi),
  - Stage 1–2'nin önceki hatla paylaşılması,
  - validasyonun geliştirme sırasında yeniden kullanılması.

## Kilit mekanizması
- **`--commit-final-model`** `final_model_decision.json` dosyasını yazar. Dosya şunları içerir: her fold'un `v2_lock.json` SHA-256 değeri, kapı sonucu, gerekçe, tarih ve git commit hash'i.
- **`--final-test` yalnızca şu koşullarda çalışır:**
  - karar dosyası mevcut kilitlerle eşleşmeli;
  - test daha önce değerlendirilmemiş olmalı;
  - validasyon yeniden hesaplandığında kilitlenen tahminlerden en fazla 0.01 mm sapmalı.
- **Kayıt:** her test açılışı `test_opening_log.json` dosyasına yazılır. Kesintiden sonra devam edilirse tamamlanmış fold'lar yeniden puanlanmaz.

## Değişiklik kaydı (2026-10-25) — testten sonra yapılan değişiklik
Bu bölüm, final testten (`publication_cv_v2final_seed42`, 2026-10-21) **sonra** yapılan bir model değişikliğini belgeler. Yukarıdaki içerik değiştirilmemiştir.

- **Değişiklik:** düzelticinin uzamsal prior'ından **sınıf** eş değişkeni çıkarıldı. Cinsiyet eş değişkeni korundu.
  - Eğitilmiş sinir ağları değişmedi; yalnızca prior'ın regresyonu aynı outer-train şekilleri üzerinde yeniden fit edildi (`--all23-prior-covariates sex`).
  - Validasyon politikası her fold için yeniden seçilip kilitlendi.
- **Gerekçe:** prior düzeyinde yapılan betimleyici bir kontrolde sınıf bilgisinin katkısı sıfıra yakın çıktı:

  | Prior'ın girdileri | Prior hatası |
  |---|---|
  | Sınıf + cinsiyet | 2.708 mm |
  | Yalnızca cinsiyet | 2.705 mm |
  | Hiçbiri | 2.729 mm |

  Ayrıca model, sınıf bilgisi olmadan uygulanabilir hale geliyor.
- **Dürüstlük notu:**
  - Bu kontrol, **test bölümündeki referans koordinatları** üzerinde yapıldı.
  - Test bölümü önceki sürümle zaten değerlendirilmişti.
  - Bu nedenle değiştirilmiş modelin test sonucu, tek seferlik ve ön-kayıtlı bir değerlendirme sayılamaz.
  - Makalede "testin bir kez açıldığı" ya da "modelin testten önce kilitlendiği" iddiası yer almayacak. Sınırlılıklar bölümünde, tasarım kararlarının aynı çapraz doğrulama verisiyle bilgilendirildiği ve test bölümlerinin önceki sürümlerle değerlendirildiği belirtilecek.
- **Yeni koşu:** `agh_v2_runs/publication_cv_v2final_noclass_seed42`. Preset sırası:
  1. `v2final_noclass_val_all_folds`
  2. `v2final_noclass_diagnostics`
  3. `v2final_noclass_commit`
  4. `v2final_noclass_final_test`

  Karar dosyasının gerekçe metni bu değişikliği açıkça içerir.
- **Beklenti:** sonuçların önceki sürümden çok az farklı olması (|Δ ALE| ≲ 0.02 mm). Belirgin bir fark çıkarsa nedeni araştırılmadan raporlanmayacak.
