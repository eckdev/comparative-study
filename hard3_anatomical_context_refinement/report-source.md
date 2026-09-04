# Hard3 < 4 mm: araştırma ve mühendislik karar raporu

**Tarih:** 4 Eylül 2026

**Hedef kitle:** 3B bilgisayarlı görü, medikal görüntü analizi ve ortodonti araştırmacıları

**Kapsam:** AGH-Former vNext Fold 1 validation Hard3 hatasının leakage-free azaltılması

**Varsayım:** `LM0=Trichion`, `LM21/22=soft-tissue Gonion left/right`; Core20 tahminleri
korunacak ve test etiketi politika kilitlenmeden kullanılmayacaktır.

**Doğrudan karar:** Eski üç-noktalı koordinat/ranking head'i büyütmek yeterli değildir.
Trichion için RGB saç çizgisi, Gonion için bilateral profil konturu ve train-only Core20
atlas priorını birleştiren H3-DVAR uygulanmalıdır. `<4.00 mm` ancak Fold 1 dış validation
kapısı geçildiğinde elde edilmiş kabul edilir; kod bu koşul sağlanmazsa 5-fold çalışmayı
reddeder.

## Araştırma sorusu

AGH-Former vNext Fold 1 validation sonucunda Core20 ALE `1.8354 mm`, Hard3 ALE
`5.2696 mm` düzeyindedir. Hard3; Trichion (`LM0`) ve bilateral soft-tissue
Gonion (`LM21/22`) noktalarından oluşur. Amaç Core20'yi değiştirmeden Hard3'ü
`4.00 mm` altına indiren, outer-train/validation/test sınırlarını koruyan bir
refinement geliştirmektir.

## Araştırma yöntemi

Birinci tarama; 3B soft-tissue landmark, Trichion, Gonion, RGB-depth patch,
geodesic refinement, graph heatmap ve heatmap loss terimlerini kapsadı. İkinci,
hedefli taramada yalnız birincil makaleler üzerinden (i) Gonion'un operasyonel
tanımı, (ii) Trichion/hairline belirsizliği, (iii) sınır ve koordinat kanalları,
(iv) hard-landmark duyarlı loss incelendi. Sonuçlar mevcut Fold 1 hata eksenleri,
candidate oracle ve Core20/Hard3 bütçesiyle karşılaştırıldı. Yeni kaynakların aynı
tasarım kararlarını tekrar etmesi ve açık kalan ana sorunun yalnız dış-validation
performansı olması nedeniyle tarama sonlandırıldı.

## Mevcut sistemden elde edilen kanıt

- Hard3 candidate-surface oracle yaklaşık `0.80 mm` olduğundan hedef noktaya yakın
  yüzey vertexleri ROI içinde bulunmaktadır. Ana darboğaz ROI kapsaması değil,
  candidate sıralamasıdır.
- Mevcut structured ranker LM0'ı iyileştirebilse de Gonion adayını bozmuştur.
  Gonion hatasının ağırlığı lateral eksenden çok vertical/depth eksenlerindedir.
- Baseline ve eski candidate arasında validation-oracle örnek seçimi dahi Hard3'ü
  yalnız yaklaşık `4.39 mm` düzeyine indirebilmektedir. Bu nedenle daha iyi bir gate
  tek başına yeterli değildir; yeni yüzey kanıtı gerekir.
- Core20'e dayalı basit train-only yerel atlas E9 Fold 1 üzerinde yaklaşık
  `5.50 -> 5.09 mm` kazanç sağlamıştır. Atlas yararlı bir prior'dır ancak tek başına
  hedef değildir.

## Literatür sentezi

1. Al-Baker ve arkadaşları 408 textured 3B yüzde 37 landmark için RGB+depth içeren
   2.5D lokal patch CNN kullanmış, genel hatayı `0.83 +/- 0.49 mm` ve sol Gonion
   hatasını yaklaşık `1.61 mm` raporlamıştır. Gonion'u posterior ve inferior alt-yüz
   kontur teğetlerinin kesişimindeki en lateral nokta olarak tanımlar. Bununla
   birlikte patchler uzman landmarkında merkezlenmiş ve augmentation sonrasında
   patch düzeyinde bölünmüştür; bu yüzden sayı doğrudan leakage-free hasta düzeyi
   protokolümüzle karşılaştırılamaz. Kaynak:
   [European Journal of Orthodontics, 2024](https://doi.org/10.1093/ejo/cjae056).

2. Qiu ve arkadaşlarının PTv3 tabanlı iki aşamalı sistemi coarse heatmap, dinamik
   geodesic crop, landmark-içi ve landmarklar-arası attention, modified Adaptive
   Wing loss ve MSE-over-mesh kullanmıştır. Genel MRE `2.17 mm` olmasına rağmen
   çalışma Trichion'da saç çizgisi girişimini, Gonion'da ise zayıf lokal özellik ve
   ana yüz kümesine uzaklığı açıkça başarısızlık nedeni olarak bildirir. Kaynak:
   [Scientific Reports, 2025/2026](https://doi.org/10.1038/s41598-025-30383-w).

3. Wang ve arkadaşları, 3B yüz landmarklarını bağımsız koordinat regresyonu yerine
   mesh graph üzerinde Gaussian heatmap regression olarak ele almıştır. Bu bulgu
   surface candidate olasılık dağılımını korumayı destekler. Kaynak:
   [AAAI 2022](https://doi.org/10.1609/aaai.v36i3.20161).

4. Adaptive Wing Loss, landmark foreground bölgesindeki küçük fakat önemli heatmap
   hatalarına daha fazla ağırlık verir. Yeni refiner bu kaybı 2.5D heatmapler için
   kullanır. Kaynak: [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Adaptive_Wing_Loss_for_Robust_Face_Alignment_via_Heatmap_Regression_ICCV_2019_paper.html).

5. PossLoss, heatmap peak uyumsuzluğunu ve zor örnekleri dağılım hatasına göre daha
   güçlü ağırlıklandırır; makaledeki ana ayarlar `r=2`, `d=0.1` olarak raporlanmıştır.
   H3-DVAR bunu Adaptive Wing'e tamamlayıcı denetim olarak ekler. Kaynak:
   [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_PossLoss_A_Reliable_and_Sensitive_Facial_Landmark_Detection_Loss_Function_ICCV_2025_paper.html).

6. Curvature-map/profile çalışması, düzenli sagittal profil ile birinci/ikinci
   türevlerin landmark ayrımında kullanılabildiğini göstermiştir. Gonion doğrudan
   o deneyin hedefi olmasa da, düz 3B nokta özellikleri yerine sıralı kontur
   temsilinin mühendislik gerekçesini sağlar. Kaynak:
   [Head & Face Medicine, 2014](https://doi.org/10.1186/1746-160X-10-54).

7. Plooij ve arkadaşları kemik ilişkili soft-tissue Gonion'un 3B fotoğrafta yeniden
   tanımlanması gerektiğini ve paired landmarkların midline noktalardan daha az
   hassas olduğunu bildirmiştir. Bu, bilateral ortak head ve belirsizlik raporunu
   destekler. Kaynak: [IJOMS, 2009](https://doi.org/10.1016/j.ijom.2008.12.009).

8. Trichion için tekrarlı işaretleme çalışmasında yaklaşık `0.73 mm` intraserial
   imprecision raporlanmıştır. Dolayısıyla tek annotator etiketi çevresinde delta
   hedefi yerine geniş heatmap ve confidence gerekir. Kaynak:
   [Journal of Orofacial Orthopedics, 2014](https://doi.org/10.1007/s00056-013-0201-9).

## Seçilen mimari: H3-DVAR

`H3-DVAR` (Hard3 Dual-View Anatomical Refiner) şu akışı uygular:

```text
frozen AGH-vNext + shape-prior prediction
  -> Stage1 geodesic candidate ROI
  -> canonical frontal RGB-depth-normal patch
  -> canonical profile RGB-depth-normal patch
  -> LM0 appearance U-Net / shared bilateral Gonion contour U-Net
  -> surface candidate heatmap scores
  -> bilateral midpoint + width consistency
  -> Core20-conditioned train-only local atlas prior
  -> validation-locked candidate/alpha selection
  -> LM0/21/22 replacement; Core20 byte-for-byte unchanged
```

Her patch 20 kanal taşır: RGB, lokal RGB kontrastı, canonical normal, intensity,
chroma, curvature, density, projected depth, view identity, CoordConv `u/v`, signed
silhouette distance, depth-gradient ve occupancy. Sparse mesh rasterı yalnız kısa
mesafeli nearest fill ile tamamlanır ve occupancy kanalı gerçek/interpolated pixel
ayrımını korur. CoordConv/sınır kanalları Gonion'un kontur tanımını doğrudan görünür
kılar; PossLoss ise peak kayması yüksek örneklerin gradyan katkısını artırır.

Model seçimi için inner OOF kullanılır. OOF en iyi epoch medyanı belirlendikten
sonra üç final seed outer-train'in tamamında sabit epoch eğitilir. Outer validation
model ağırlığına girmez; yalnız neural/atlas fusion ve blend seçer. Test etiketi,
bu politika kilitlenmeden okunmaz.

## Önceden tanımlı kabul ölçütü

- Birincil: validation Hard3 ALE `<4.00 mm`.
- Koruma: Core20 koordinatlarında tam sıfır değişim.
- Destekleyici: overall gain `>=0.03 mm`, Hard3 gain `>=0.20 mm`, bootstrap
  `P(improved)>=0.90`, overall p95 regresyonu `<=0.10 mm`.
- Fold 1 bu eşiği karşılamazsa tam CV çalıştırılmaz.
- Eşik karşılanırsa aynı kilitli yapı 5 fold çalıştırılır; yayın sonucu fold mean,
  SD ve bootstrap CI ile verilir.

## Claim-to-source ledger

| İddia | Birincil kaynak (yazar, yayın, yıl) | URL / erişim notu | Tasarımdaki karşılığı |
|---|---|---|---|
| RGB+depth lokal patchler 3B yüz landmarkında etkilidir; Gonion kontur-teğet tanımlıdır | Al-Baker et al., *European Journal of Orthodontics*, 2024 | [Tam metin](https://doi.org/10.1093/ejo/cjae056), erişim 2026-09-04 | Dual-view 2.5D raster ve contour head |
| Trichion saç çizgisinden etkilenir; geodesic crop ve landmark ilişkileri yararlıdır | Qiu et al., *Scientific Reports*, 2025/2026 | [Tam metin](https://doi.org/10.1038/s41598-025-30383-w), erişim 2026-09-04 | Appearance head, ROI ve anatomik prior |
| Mesh graph heatmap regression 3B yüz landmarkına uygulanabilir | Wang et al., *AAAI*, 2022 | [Yayın sayfası](https://doi.org/10.1609/aaai.v36i3.20161), erişim 2026-09-04 | Surface candidate heatmap distribution |
| Foreground-aware heatmap ve boundary/CoordConv bilgisi lokalizasyonu iyileştirebilir | Wang, Bo ve Fuxin, *ICCV*, 2019 | [CVF tam metin](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Adaptive_Wing_Loss_for_Robust_Face_Alignment_via_Heatmap_Regression_ICCV_2019_paper.html), erişim 2026-09-04 | Adaptive Wing, `u/v`, silhouette/depth-gradient |
| Peak mismatch ve zor örnek duyarlılığı ayrı ele alınabilir | Zhu, *ICCV*, 2025 | [CVF tam metin](https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_PossLoss_A_Reliable_and_Sensitive_Facial_Landmark_Detection_Loss_Function_ICCV_2025_paper.html), erişim 2026-09-04 | PossLoss, `r=2`, `d=0.1` |
| Profil ve türev bilgileri landmark ayrımında kullanılabilir | Lippold et al., *Head & Face Medicine*, 2014 | [Tam metin](https://doi.org/10.1186/1746-160X-10-54), erişim 2026-09-04 | Profil görünümü, curvature ve depth-gradient |
| Paired soft-tissue landmarklar daha değişken olabilir ve 3B tanım önemlidir | Plooij et al., *IJOMS*, 2009 | [Yayın kaydı](https://doi.org/10.1016/j.ijom.2008.12.009), erişim 2026-09-04 | Ortak bilateral head, pair loss ve confidence |
| Trichion işaretlemesinde ölçülebilir gözlemci belirsizliği vardır | Fink et al., *Journal of Orofacial Orthopedics*, 2014 | [Yayın kaydı](https://doi.org/10.1007/s00056-013-0201-9), erişim 2026-09-04 | Geniş heatmap ve belirsizlik farkındalığı |

## Sınırlamalar

Bu rapor mimari gerekçeyi ve çalışan uygulamayı belgeler; `<4 mm` sonucu Fold 1
validation çalıştırılmadan elde edilmiş sayılmaz. Al-Baker çalışmasının çok düşük
sonucu, uzman-merkezli crop ve patch-level split olasılığı nedeniyle hedef için
kanıt sunar fakat doğrudan karşılaştırma değeri olarak kullanılmamalıdır. Ayrıca
tek uzmanlı veri setinde LM0/21/22 annotation belirsizliği ikinci bir uzman veya
tekrarlı işaretleme olmadan ayrıştırılamaz.
