# `greedy` eşleştiricisi: arşiv kararı

2026-09-15. Bu belge bir tartışmayı kapatır. `greedy` motoru **arşivlenmiştir**:
silinmemiştir, varsayılan olmaktan çıkmıştır.

## Kesin kural

> `greedy` ile `optimal` sayıları **asla yan yana konmaz**, farkları "iyileşme"
> diye sunulmaz, ve her iddia hangi motorla ölçüldüğünü söyler. `greedy` yalnız
> dondurulmuş bir koşuyu yeniden üretmek için, **adıyla** seçilir.

Bu kural `HANDOFF.md`'nin değişmez kuralları listesine de yazıldı.

## `greedy` neydi

Run 015-021'in "staged first-fit" eşleştirmesi. Altın ve tahmin referanslarını
katmanlı bir ilk-uyan yürüyüşüyle eşliyor. Projenin ilk metrik tanımı buydu ve
o dönemin bütün sayıları onunla üretildi.

## Neden bırakıldı — üç kusur

`benchmark/reference/matching.py`'nin kendi docstring'i sayıyor:

1. **Maksimum-kardinalite değil.** Eşleşebilecek çiftleri kaçırıyor, yani doğru
   cevabı cezalandırabiliyor.
2. **Satır sırasına bağlı.** Aynı girdi, farklı sıra, farklı sonuç. Bir metrik
   için bu tek başına diskalifiye sebebidir.
3. **Asimetrik.** Maddesiz (kanun düzeyi) altın satırları recall paydasından
   düşürüyor, ama aynı şekli tahmin tarafında yanlış pozitif diye
   cezalandırıyor. Yani o referansı doğru bilen model ceza alıyor.

Ek olarak `greedy` yolunda madde metni büyük/küçük harfe katlanmıyor:
`Mükerrer 257` ile `mükerrer 257` ayrı madde sayılıyor.

`optimal` üçünü de kapatıyor: simetrik uygunluk, birleştirme temelli uyumluluk,
ve minimum-maliyet maksimum-akış ataması — sonuç bir **graf değişmezi**, sıradan
bağımsız, eşitlik bozmaları deterministik.

## Ölçülmüş maliyet

| ölçüm | greedy | optimal |
| --- | --- | --- |
| Run 023 çift-motor mühürlü karşılaştırma, aynı Run 021 tahminleri | F1 **0.8825** | F1 **0.9348** |
| aynı koşuda docwise kazananı | Qwen3.5 397B (0.5182) | **DeepSeek v3.2 (0.5931)** |
| taban model, 50 validation belgesi (2026-09-15) | F1 **0.707736** | F1 **0.759055** |
| aynı ölçümde `tp+fn` | 372 | 339 |

Motor yalnız seviyeyi değil **sıralamayı** da değiştiriyor: hangi modelin
kazandığı motora bağlı. Etkilenen veri payı küçük değil — kanonik 3523
referansın **249'u (%7.1)** maddesiz, yani greedy'nin recall'dan düşürdüğü sınıf.

`evaluate.py`'nin kendi CLI yardımı da şunu yazıyordu: *GT v4, greedy altında
alt-bent bölünmeleri yüzünden **geçersizdir**.*

## Neden silinmedi

İki sebep, ikisi de yeniden üretilebilirlik:

1. **Dondurulmuş Run 015-021** o motorla üretildi. Onları yeniden üretebilmek
   tarihî kaydın bütünlüğüdür.
2. **Run 023 makaleye bakan çift-motor karşılaştırmasıdır** — greedy yarısı
   olmadan karşılaştırma diye bir şey kalmaz.

Bu yüzden `greedy` `SUPPORTED_MATCHING_ENGINES` içinde kalır ve
`--matching-engine greedy` ile seçilebilir.

## Ne değişti (2026-09-15)

| yer | önce | sonra |
| --- | --- | --- |
| `evaluate.evaluate_prediction_file` varsayılanı | greedy | **optimal** |
| `evaluate.py` CLI varsayılanı | greedy | **optimal** |
| `run.py` CLI varsayılanı | greedy | **optimal** |
| `normalize_matching_engine("")` / `(None)` | greedy | **optimal** |
| `q36_p1` değerlendirme sözleşmesi | greedy | **optimal** |

**Bilerek değişmeyen:** motoru kaydetmemiş bir raporu geri okurken kullanılan
`result.get("matching_engine", MATCHING_ENGINE_GREEDY)`. Motoru yazmayan bir
rapor **eski** bir rapordur ve gerçekten greedy ile üretilmiştir; onu optimal
diye etiketlemek tarihi yeniden yazmak olurdu.

**Neden güvenliydi:** depoda varsayılana bel bağlayan hiçbir çağıran yoktu.
Tek çağıran `g0_training.canonical_evaluate` ve o bayrağı açıkça geçiyor.

`q36_p1` özel olarak incelendi: oradaki `"greedy"` literalinin **hiçbir
gerekçesi yazılı değildi** (karşılaştırın: döngünün kendi seçiminin on iki
satırlık gerekçesi var), Q36-P1 dondurulmuş bir koşu değil yeni bir deney, ve
bugüne dek **hiç artefakt üretmemişti**. Yani kopyalanarak gelmiş sessiz bir
varsayılandı — DQ-Loop'u ısıran kusurun aynısı.

## Koruma

`tests/test_matching_engine_default.py` altı testle bu kararı çiviliyor. Beş
varsayılanın her biri ayrı ayrı mutasyonla doğrulandı: birini greedy'ye geri
almak tam olarak bir testi düşürüyor. Altıncı test de greedy'nin **hâlâ adıyla
erişilebilir** olduğunu koruyor — arşivlemek silmek değildir.
