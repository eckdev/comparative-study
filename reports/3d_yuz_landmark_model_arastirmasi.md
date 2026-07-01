# 3B Yuz Landmark Lokalizasyonu Icin Acik Kaynak Model Arastirmasi

Tarih: 2026-06-29

## Proje baglami

Bu projenin veri seti 3B PLY yuz meshleri ve uzman ortodontist tarafindan isaretlenmis 23 adet 3B yumuşak doku landmark noktasindan olusur. Bu nedenle uygun model ailesi, 2B fotograf tabanli landmark tespiti degil; dogrudan 3B mesh, point cloud veya vertex/point-wise yuzey temsili ile calisabilen mimarilerdir.

Mevcut deneylerde ayni train/validation/test dagilimi kullanilmistir:

| Model | Girdi | Test ALE | Not |
|---|---:|---:|---|
| PAL-Net raw | 3B point cloud / patch | 3.080 | Domain'e en yakin mevcut model |
| PAL-Net snapped | 3B point cloud / mesh yuzeyi | 2.574 | En iyi mevcut sonuc |
| DiffusionNet mask segmentation | 3B point cloud / geometri | 3.764 | Paper-benzeri 3.5 mm maske hedefi ile iyilesti |

ALE: 23 landmark icin ortalama Oklid lokalizasyon hatasi.

## Model secim kriterleri

Bir modelin bu makale projesine uygun sayilmasi icin su kriterler kullanildi:

1. PLY mesh veya point cloud girdisini dogrudan ya da az donusumle kullanabilmeli.
2. Cikti olarak 23 landmark icin koordinat regresyonu veya point/vertex-wise heatmap/segmentation uretebilmeli.
3. Acik kaynak kodu bulunmali.
4. 300 ornekli nispeten kucuk medikal veri setinde egitilebilir olmali.
5. Sonuc ALE ile raporlanabilir olmali.

## En uygun kisa liste

### 1. PAL-Net

Kaynak: <https://github.com/Ali5hadman/PAL-Net-A-Point-Wise-CNN-with-Patch-Attention>

**Uygunluk:** Cok yuksek.

PAL-Net, point-wise CNN ve patch attention mantigi ile 3B yuz landmark lokalizasyonuna en dogrudan uyan modeldir. Bizim dataset PLY yuzeyleri ve 23 landmark koordinatlari icerdigi icin modelin hedef problemiyle uyumludur. Mevcut deneylerde de en iyi sonucu vermistir.

**Makale degeri:** Ana model veya en guclu baseline olarak konumlandirilabilir.

**Onerilen kullanim:**

- Ana sonuc olarak PAL-Net snapped ALE verilmeli.
- Raw ve snapped farki ayrica raporlanmali.
- Landmark bazli hata tablosu, anatomik bolge bazli hata ve sinif/cinsiyet bazli hata eklenmeli.

### 2. DiffusionNet

Kaynaklar:

- Kod: <https://github.com/nmwsharp/diffusion-net>
- Paper: <https://arxiv.org/abs/2012.00888>
- Klinik kullanim ornegi: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10948387/>

**Uygunluk:** Cok yuksek.

DiffusionNet mesh ve point cloud uzerinde intrinsic/geometrik operatorlerle calisabilen modern bir yuzey ogrenme modelidir. Bizim problemde iki sekilde kullanilabilir:

1. Her landmark icin nokta siniflandirma/regresyon.
2. Paper-benzeri sekilde her landmark cevresinde 3.5 mm pozitif maske olusturup semantic segmentation mantigi.

Ikinci yaklasim bizim deneylerde daha iyi calismistir.

**Makale degeri:** PAL-Net'e karsi en guclu geometrik deep learning baseline.

**Onerilen kullanim:**

- `mask_bce`, `mask_radius=3.5`, `XYZ` input, activation-weighted postprocess.
- Colab Pro/A100 varsa full mesh, `width=384`, `MLP=768`, `blocks=12`, `epochs=200` denenmeli.
- Standart GPU icin 4096/8192 sampled points ile ara deney yapilmali.

### 3. PointNet++

Kaynaklar:

- Kod: <https://github.com/charlesq34/pointnet2>
- Paper: <https://arxiv.org/abs/1706.02413>

**Uygunluk:** Yuksek.

PointNet++ point cloud verisinde lokal komsuluk bilgisiyle hiyerarsik ogrenme yapar. PLY meshlerden sampled point cloud uretilerek dogrudan kullanilabilir. Landmark lokalizasyonu icin iki makul uyarlama vardir:

1. Her landmark icin koordinat regresyon head'i.
2. Her point icin 23 kanalli heatmap/segmentation head'i ve argmax/weighted average postprocess.

**Makale degeri:** Klasik ve cok bilinen point cloud baseline. PAL-Net ve DiffusionNet sonuclarinin daha guvenilir yorumlanmasi icin iyi bir referans modeldir.

**Riskler:**

- Mesh yuzey baglantisini kullanmaz.
- Kucuk veri setinde overfit olabilir.
- Nokta ornekleme yogunlugu sonucu etkiler.

**Onerilen deney:**

- 4096 veya 8192 sampled points.
- 23 kanalli mask segmentation hedefi.
- ALE ve landmark bazli hata raporu.

### 4. DGCNN

Kaynaklar:

- Kod: <https://github.com/WangYueFt/dgcnn>
- Paper: <https://arxiv.org/abs/1801.07829>

**Uygunluk:** Yuksek.

DGCNN, dinamik k-NN graflari ve EdgeConv operatoru ile point cloud uzerinde lokal geometriyi yakalar. Yuzey meshini point cloud olarak ele aldigimizde PAL-Net'e yakin bir problem formulasyonuna uyarlanabilir.

**Makale degeri:** PointNet++'a gore daha guclu lokal geometri baseline'i olabilir.

**Onerilen deney:**

- Input: Procrustes hizalanmis sampled surface points.
- Output: 23 landmark icin point-wise logits.
- Loss: 3.5 mm landmark maskesi ile BCE veya soft heatmap MSE.
- Postprocess: softmax-weighted coordinate.

**Riskler:**

- Dinamik k-NN CPU/GPU bellek maliyeti yuksek olabilir.
- Tam mesh yerine sampled point cloud daha pratik olur.

### 5. KPConv

Kaynaklar:

- Kod: <https://github.com/HuguesTHOMAS/KPConv>
- Paper: <https://arxiv.org/abs/1904.08889>

**Uygunluk:** Yuksek, ancak entegrasyon maliyeti orta-yuksek.

KPConv nokta bulutu uzerinde kernel point convolution kullanir ve nokta bulutu segmentasyonu icin gucludur. Bizim problemde landmark cevresi maske segmentasyonu olarak formulle edilebilir.

**Makale degeri:** Point cloud segmentation literaturunden guclu bir baseline.

**Riskler:**

- Kod tabani ve veri hazirligi PointNet++/DGCNN'e gore daha agir olabilir.
- 300 ornek icin hiperparametre ayari gerekebilir.

**Onerilen durum:**

- Makalede model sayisi 3-4 ile sinirli tutulacaksa KPConv opsiyonel kalabilir.
- Daha genis karsilastirma istenirse eklenmeli.

### 6. PointNeXt / OpenPoints

Kaynaklar:

- PointNeXt: <https://github.com/guochengqian/PointNeXt>
- OpenPoints: <https://github.com/guochengqian/openpoints>
- Paper: <https://arxiv.org/abs/2206.04670>

**Uygunluk:** Yuksek.

PointNeXt, PointNet++ ailesinin modern ve daha guclu bir surumu gibi dusunulebilir. OpenPoints ise farkli point cloud modellerini daha standart bir framework icinde calistirmaya yardimci olur.

**Makale degeri:** Modern point cloud baseline olarak degerli.

**Riskler:**

- Framework entegrasyonu daha fazla zaman alabilir.
- Medikal/ortodontik veri icin hazir landmark pipeline yoktur; custom dataset ve custom head gerekir.

**Onerilen deney:**

- PointNet++ yerine daha modern baseline istenirse PointNeXt tercih edilebilir.
- Cikti yine 23 kanalli point-wise segmentation veya koordinat regresyonu olmali.

### 7. Point Transformer

Kaynaklar:

- Kod: <https://github.com/POSTECH-CVLab/point-transformer>
- Paper: <https://arxiv.org/abs/2012.09164>

**Uygunluk:** Orta-yuksek.

Point Transformer attention tabanli point cloud ogrenmesi yapar. Landmark lokalizasyonunda uzak ve lokal yuz bolgeleri arasindaki iliskileri modelleyebilir.

**Makale degeri:** Modern transformer tabanli baseline olarak ilgi cekici.

**Riskler:**

- 300 ornekli veri setinde transformer mimari overfit edebilir.
- Daha yuksek GPU bellegi ve iyi augmentation gerekebilir.

**Onerilen durum:**

- Ana deneylerden sonra, kaynak ve zaman varsa eklenmeli.
- Kucuk model varyanti veya pretraining dusunulmeli.

## Mesh tabanli ama dikkatli kullanilmasi gereken modeller

### MeshCNN

Kaynaklar:

- Kod: <https://github.com/ranahanocka/MeshCNN>
- Paper: <https://arxiv.org/abs/1809.05910>

**Uygunluk:** Orta.

MeshCNN mesh edge convolution ve mesh pooling kullanir. Mesh segmentasyonu/siniflandirmasi icin onemlidir; ancak landmark koordinat lokalizasyonu icin dogrudan tasarlanmamistir. Bizim problem icin vertex/edge bazli landmark heatmap uyarlamasi gerekir.

**Makale degeri:** Mesh deep learning literaturunden baseline olabilir; fakat uygulama maliyeti DiffusionNet'e gore daha yuksek ve beklenen performans daha belirsizdir.

**Oneri:** Ilk karsilastirma turuna alinmasin. DiffusionNet mesh baseline olarak yeterince guclu.

### SpiralNet++ / CoMA ailesi

Kaynaklar:

- SpiralNet++ kod: <https://github.com/sw-gong/spiralnet_plus>
- CoMA kod: <https://github.com/anuragranj/coma>

**Uygunluk:** Kosullu.

Bu aile, genellikle ayni topolojiye sahip kayitli meshlerde cok daha uygundur. Bizim PLY yuzeylerinin vertex sayisi/topolojisi degisiyorsa dogrudan kullanmak zordur. Tum meshler ortak template'e non-rigid register edilirse guclu olabilir.

**Makale degeri:** Ancak meshler ortak topolojiye getirilebilirse yuksek.

**Oneri:** Mevcut proje akisi icin oncelikli degil. Non-rigid registration pipeline kurulacaksa tekrar degerlendirilmeli.

## Uygun olmayan veya ikincil kalan yaklasimlar

### 2B fotograf landmark modelleri

Ornekler: HRNet-Facial-Landmark-Detection, FAN, MediaPipe Face Mesh.

Bu modeller 2B RGB yuz goruntusu bekler. Bizim dataset PLY mesh ve 3B landmarklardan olustugu icin 2B projeksiyona zorlamak ciddi bilgi kaybina neden olur. HRNet denemesinde bu nedenle zayif sonuc alinmistir. Bu aile, ancak gercek 2B yuz fotograflari ve 2B landmark etiketleri varsa makaleye dahil edilmelidir.

### 3DMM / image-to-3D face alignment modelleri

Bu modeller genellikle tek 2B goruntuden 3B yuz modeli veya 3DMM parametresi tahmin eder. Bizim problemimizde girdi zaten 3B yuzey oldugu icin dogrudan uygun degildir.

## Makale icin onerilen deney seti

Makale icin model sayisini cok dagitmadan su siralama onerilir:

| Oncelik | Model | Neden |
|---:|---|---|
| 1 | PAL-Net | Domain'e en yakin ve mevcut en iyi sonuc |
| 2 | DiffusionNet | Modern mesh/point cloud geometric learning baseline |
| 3 | PointNet++ veya PointNeXt | Klasik/modern point cloud baseline |
| 4 | DGCNN | Lokal geometriyi guclu yakalayan point cloud baseline |
| 5 | KPConv | Daha genis karsilastirma icin opsiyonel segmentation baseline |

Minimum makale seti:

```text
Mean shape baseline
PAL-Net
DiffusionNet
PointNet++ veya PointNeXt
DGCNN
```

Genisletilmis set:

```text
Mean shape baseline
PAL-Net
DiffusionNet
PointNet++
DGCNN
KPConv
Point Transformer
```

## Ortak deney protokolu

Adil karsilastirma icin tum modellerde ayni protokol kullanilmalidir:

1. Ayni train/validation/test split: 180/60/60.
2. Ayni Procrustes rigid alignment transformlari.
3. Ayni landmark sirasi: 23 nokta.
4. Ayni metrik: Average Localization Error.
5. Ek raporlar:
   - Landmark bazli mean/median hata.
   - Anatomik bolge bazli hata.
   - Class1/Class2/Class3 ve cinsiyet bazli hata.
   - Boxplot veya violin plot.
   - 3B gorsel uzerinde uzman vs model tahminleri.

## Onerilen hedef formulasyonlari

### A. Point-wise segmentation / heatmap

Her landmark icin mesh/point cloud noktalarinda 23 kanalli cikti uretilir. Uzman landmarkin 3.5 mm cevresindeki noktalar pozitif kabul edilir.

```text
target_j(point) = 1 if distance(point, landmark_j) <= 3.5 mm else 0
```

Tahmin:

```text
pred_landmark_j = weighted_average(points, activation_j)
```

Bu formulasyon DiffusionNet deneyinde iyi calismistir ve PointNet++/DGCNN/KPConv icin de uygundur.

### B. Koordinat regresyonu

Model dogrudan 23 x 3 koordinat uretir.

Avantaji basit olmasidir. Dezavantaji, tahminin yuzey uzerinde kalmasini garanti etmez. Bu nedenle son adimda en yakin mesh vertexine snapping eklenmelidir.

### C. Hybrid

Model once landmark bolgesi icin heatmap uretir, sonra lokal patch icinde coordinate refinement yapar. PAL-Net mantigina en yakin formulasyondur ve en iyi performans potansiyeline sahiptir.

## Sonuc ve karar

Bu dataset icin en dogru model ailesi 3B point cloud/mesh tabanli modellerdir. HRNet gibi 2B fotograf modelleri, veri yapisina uymadigi icin ana karsilastirmaya alinmamalidir.

Mevcut sonuclar dikkate alindiginda makale icin en guclu hikaye su olabilir:

1. PAL-Net, ortodontik 3B yuz landmark lokalizasyonunda en iyi performansi vermektedir.
2. DiffusionNet, paper-benzeri 3.5 mm landmark maskesi ile guclu bir geometrik baseline olarak yaklasmaktadir.
3. PointNet++/PointNeXt ve DGCNN eklenirse, sonuclarin sadece iki modele bagli olmadigi ve genel point cloud literaturuyle karsilastirildigi gosterilebilir.
4. MeshCNN/SpiralNet++ gibi modeller, ya uyarlama maliyeti yuksek ya da ortak mesh topolojisi gerektirdigi icin ikincil/opsiyonel kalmalidir.

## Kaynak listesi

- PAL-Net: <https://github.com/Ali5hadman/PAL-Net-A-Point-Wise-CNN-with-Patch-Attention>
- DiffusionNet: <https://github.com/nmwsharp/diffusion-net>
- DiffusionNet paper: <https://arxiv.org/abs/2012.00888>
- Klinik DiffusionNet kullanim ornegi: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10948387/>
- PointNet++: <https://github.com/charlesq34/pointnet2>
- PointNet++ paper: <https://arxiv.org/abs/1706.02413>
- DGCNN: <https://github.com/WangYueFt/dgcnn>
- DGCNN paper: <https://arxiv.org/abs/1801.07829>
- KPConv: <https://github.com/HuguesTHOMAS/KPConv>
- KPConv paper: <https://arxiv.org/abs/1904.08889>
- PointNeXt: <https://github.com/guochengqian/PointNeXt>
- OpenPoints: <https://github.com/guochengqian/openpoints>
- PointNeXt paper: <https://arxiv.org/abs/2206.04670>
- Point Transformer: <https://github.com/POSTECH-CVLab/point-transformer>
- Point Transformer paper: <https://arxiv.org/abs/2012.09164>
- MeshCNN: <https://github.com/ranahanocka/MeshCNN>
- MeshCNN paper: <https://arxiv.org/abs/1809.05910>
- SpiralNet++: <https://github.com/sw-gong/spiralnet_plus>
- CoMA: <https://github.com/anuragranj/coma>
