# H3-DVAR Fold 1 Sonuc Denetimi

## Olculen durum

Eklenen dis-fold validation dosyalari, test etiketleri kullanilmadan incelenmistir.

| Metrik | Stage 2 + shape prior | H3-DVAR v1 | Kazanc |
|---|---:|---:|---:|
| All-23 ALE | 2.2795 mm | 2.1984 mm | 0.0811 mm |
| Core20 ALE | 1.8360 mm | 1.8360 mm | 0.0000 mm |
| Hard3 ALE | 5.2358 mm | 4.6143 mm | 0.6215 mm |
| p95 | 6.5318 mm | 5.9755 mm | 0.5564 mm |

Nihai landmark hatalari `LM0=2.9303`, `LM21=5.6220` ve `LM22=5.2905 mm`dir.
Dolayisiyla v1 Trichion'u anlamli bicimde cozmus, ancak bilateral Gonion secimi
performans tavanini belirlemistir. Core20'nin tam olarak ayni kalmasi, kazancin Hard3
asamasindan geldigini dogrular.

Validation uzerinde her landmark icin baseline ile v1 adayi arasindan kusursuz secim
yapilabilse Hard3 ALE yaklasik `3.88 mm` olurdu. Bu, mevcut adaylarda 4 mm alti bilgi
bulundugunu; fakat basit confidence/gate ozelliklerinin dogru adayi guvenilir bicimde
secemedigini gosterir. Sabit alpha veya daha buyuk bir U-Net tek basina bu secim
sorununu cozmez.

## H3-DVAR v2 hipotezi

V1, LM21 ve LM22 heatmaplerini ortak agirliklarla uretse de iki noktayi bagimsiz
decode eder. Sonradan uygulanan midpoint/width loss, aday secimi sirasinda karsi
taraftaki konturu gorunur kilmaz. V2 bu islemi gercek bir ortak aday problemi yapar:

1. En guclu sol ve sag Gonion adaylari unary heatmaplerden onerilir.
2. Her `sol x sag` cifti mirrored canonical koordinatlar ve tahmin edilen LM10-12
   alt-orta-hat anchorlariyla temsil edilir.
3. Kucuk bir MLP ortak cifti soft-listwise ranking ve hard-negative loss ile puanlar.
4. LM21/22 koordinatlari ayni joint dagilimin iki marjinalinden uretilir.
5. Frontal/profil agirligi her ornegin heatmap kalitesine gore dinamik hesaplanir.

Bu degisiklik model kapasitesini rastgele buyutmez; Fold 1 sonucunda olculen bilateral
secim hatasini hedefler. Trichion yolu korunur ve Core20 degistirilemez.

## Sonraki karar

- `gonion_pair_topk_recall < 0.90`: asil sorun unary proposal recall'dir. Pair modelini
  buyutmeden once top-k veya ROI yeniden ele alinmalidir.
- Recall yuksek, Gonion hala `>4.3 mm`: adaylar mevcut fakat pair geometrisi yeterince
  genellenmiyordur; daha fazla fold maliyetine girilmemelidir.
- Hard3 `<4.00 mm`, All-23 `<=2.25 mm` ve p95 kapisi gecerse: ayni kilitli ayarlarla
  bes-fold CV calistirilabilir.
- All-23 `<2.00 mm` icin Core20 `1.8360 mm` sabitken Hard3'ün yaklasik `3.0931 mm`
  olmasi gerekir. Bu nedenle 4 mm kapisi nihai iddia degil, tam CV harcamasindan onceki
  ara fizibilite esigidir.
