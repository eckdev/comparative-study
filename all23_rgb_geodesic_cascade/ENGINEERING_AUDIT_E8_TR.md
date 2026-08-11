# E8 Leakage-Free 5-Fold Muhendislik Denetimi

Bu rapor `all23_publication_cv_stage1_v4_seed42` sonuclarinin pooled prediction
dosyalari, fold metrikleri, registration raporlari ve candidate-oracle kayitlari
birlikte incelenerek hazirlanmistir. Test foldlari birlestirildiginde 300 benzersiz
ornek ve `300 x 23 = 6900` landmark tahmini vardir. Her ornek test havuzunda tam
bir kez bulunur; foldlar arasinda sample ID cakismasi saptanmamistir.

## Ana Sonuc

| Metrik | Deger |
|---|---:|
| All-23 ALE | 2.5372 mm |
| Median | 1.7509 mm |
| Standart sapma | 2.7680 mm |
| P75 / P90 / P95 / P99 | 3.029 / 5.143 / 7.149 / 13.449 mm |
| Maksimum | 73.707 mm |
| SDR@2 / @3 / @4 mm | 56.44% / 74.55% / 83.78% |
| Core20 ALE | 1.9622 mm |
| Hard3 ALE | 6.3707 mm |

Fold ALE degerleri `2.7362, 2.5545, 2.3765, 2.4646, 2.5542 mm` olup
ortalama `2.5372 mm`, foldlar arasi standart sapma yaklasik `0.1194 mm`dir.
Sonuc tek bir kotu fold ile aciklanamaz.

## Darbogaz Ayrisimi

Pooled coarse ALE `2.6375 mm`, ungated local refinement ALE `2.6584 mm`, mevcut
gate sonrasi ALE `2.5372 mm`dir. Lokal head tek basina coarse tahminden daha
kotudur; gate zarari kismen geri almaktadir. Refined endpoint Core20 tahminlerinin
yalnizca %49.3'unde, Hard3 tahminlerinin %56.1'inde coarse endpointten daha iyidir.

Validation uzerinde hesaplanan coarse-refined dogrusu boyunca sample/landmark
oracle ALE `2.1179 mm`, testte `2.1169 mm`dir. Dolayisiyla mevcut iki koordinatin
yalniz daha iyi karistirilmasi genel sonucu tek basina 2 mm altina indiremez.
Validation-optimal ortalama alpha Core20 icin yaklasik `0.49`, Hard3 icin `0.56`
iken model sirasiyla `0.77` ve `0.90` uygulamaktadir. Gate asiri agresiftir.

Validation ile secilen grup/per-landmark alpha kalibrasyonlari test ALE'yi ancak
yaklasik `2.48-2.50 mm` bandina getirir. Bu ucuz ve yararli bir guvenlik katmanidir,
ancak ana cozum degildir.

## Landmark Analizi

Hard3 pooled test hatalari:

| Landmark | Anatomik ad | Mean | Median | P95 | SDR@2 |
|---|---|---:|---:|---:|---:|
| LM0 | Trichion | 5.5165 | 4.63 | 13.46 | 15.7% |
| LM21 | Gonion Left | 7.2854 | 6.17 | 14.67 | 7.0% |
| LM22 | Gonion Right | 6.3102 | 5.17 | 14.28 | 8.3% |

Core20 icinde en zor noktalar LM10 `2.7622 mm`, LM11 `2.9997 mm`, LM12
`3.0460 mm`, LM1 `2.5246 mm` ve LM2 `2.4433 mm`dir. Buna ragmen Core20 grup
ortalamasi 2 mm altindadir; ana performans acigi LM0/21/22 tarafindadir.

Hard3 residual vektorlerinin ortalama bias'i kucuk, eksen bazli varyansi buyuktur.
Bu nedenle sabit koordinat bias duzeltmesi beklenen cozum degildir. Erkeklerde
Core20 `2.0306 mm`, Hard3 `6.8869 mm`; kadinlarda Core20 `1.8938 mm`, Hard3
`5.8546 mm`dir. Sinif I/II/III overall sonuclari birbirine yakindir.

## ROI ve Registration

Validation candidate-oracle ALE foldlar boyunca yaklasik `0.77 mm`dir ve tum
preflight kapilari gecmistir. Dogruya yakin yuzey vertexi ROI icinde vardir; sorun
ROI kapsami veya mesh cozumunurlugu degil, aday puanlamadir.

Registration residuali ile sample ALE arasindaki pooled korelasyon yaklasik
`0.27`, Hard3 ile `0.16`dir. Registration-outlier dokuz test orneginde hata daha
yuksektir, ancak kalan 291 ornekte de Hard3 ALE yuksek kalir. Registration ikincil
bir etkendir, ana darbogaz degildir.

En buyuk hata `Class2_M17 / LM21` ornegindedir. Coarse hata `38.76 mm`, ungated
refined hata `78.19 mm`, gate sonrasi hata `73.71 mm`dir. Model bu tahmine cok
dusuk confidence vermesine ragmen alpha `0.887` uygulamistir. Bu durum gate loss'un
hareket buyuklugunu ve coarse'a gore regret'i yeterince cezalandirmadigini gosterir.

## E9 Muhendislik Karari

E9, pahali Stage 1 ve registration hattini degistirmez. Mevcut cache'ler korunur.
Asagidaki tek ana hipotez test edilir: Hard3 icin dogru vertex ROI'de mevcutysa,
anatomik anchor kosullu keskin candidate ranking bu vertexi genis heatmap
regresyonundan daha guvenilir secebilir.

E9 degisiklikleri:

- LM0 icin LM1/LM2/LM3/LM12; LM21/22 icin LM10/LM11/LM12 ve karsi Gonion
  anchor ozellikleri.
- Hard3'e ozel candidate rank head ve expert-nearest vertex cross-entropy.
- Hard3 icin keskin top-k coordinate + yuzey projection.
- Ayrik Hard3 gate; optimal koordinat ve coarse-regret duyarlı gate loss.
- Validation checkpoint skorunda Core20/Hard3 dengesi.
- Validation-only iki gruplu alpha scale kalibrasyonu.
- Stage 2 epoch resume, tamamlanmis fold atlama ve eski preprocessing cache reuse.

## Maliyet Kontrollu Kabul Kapisi

Ilk olarak yalniz outer Fold 1 calistirilir. Bes folda gecmek icin E9 validation
sonucu ayni foldun E8 validation sonucuna gore su kosullari saglamalidir:

1. Hard3 ALE en az `0.75 mm` azalmali.
2. All-23 ALE en az `0.10 mm` azalmali.
3. Core20 regresyonu `0.05 mm`yi asmamali.
4. P95 ve maksimum hata E8'e gore artmamali.
5. Mutlak Fold 1 validation sonucu All-23 `<=2.25 mm`, Hard3 `<=4.5 mm` olmali.

All-23 `<=2.10 mm`, Core20 `<=1.95 mm` ve Hard3 `<=3.20 mm` birlikte saglanirsa
2 mm hedefi icin guclu uygulanabilirlik sinyali kabul edilir.

Bu kapilar gecilmezse ayni E9'u bes fold calistirmak bilimsel veya mali acidan
gerekceli degildir. O durumda bir sonraki ana degisiklik Stage 1'i buyutmek degil,
Gonion icin bilateral pair-ranking ve Trichion icin texture-line supervision olur.
