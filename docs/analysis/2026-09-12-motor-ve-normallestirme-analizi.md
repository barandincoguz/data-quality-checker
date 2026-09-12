# Ölçüm, normalleştirme ve kova mantığı — bulgu raporu

2026-09-12. Üç ayrı inceleme, hepsi salt-okunur. Her bulgu gerçek kodu
çalıştırarak veya gerçek veriyi sayarak kanıtlandı; tahmine dayanan madde yok.
Aşağıdaki sayıları ayrıca kendim tekrar ölçtüm.

## Özet: tek bir bayrak, ilk üç bulguyu birden kapatıyor

Projenin iki eşleştirme motoru var. `optimal` motoru **3 Ağustos 2026'da kabul
edilmiş**, README'ye şöyle yazılmış: aynı tahminlerle `greedy` F1 **0.8825**,
`optimal` **0.9348** veriyor — ve kazanan model bile değişiyor. README'nin kendi
kuralı: *"her iddia hangi motorla ölçüldüğünü söylemek zorunda."*

Ama DQ-Loop değerlendiriciyi çağırırken motoru **hiç belirtmiyor**
(`g0_training.py:446` `canonical_evaluate`, komut satırında `--matching-engine`
yok), o yüzden varsayılan `greedy` çalışıyor.

**Sonuç: M₀'ın 0.853933'ü ve M₀′'nin 0.864569'u eski motorla hesaplandı.**
Validation kaydı hangi motorun kullanıldığını yazmıyor bile.

Karşılaştırmalar güvende — M₀ ile M₀′ aynı motorla ölçüldü, aradaki fark
geçerli. Kaybolan şey mutlak seviye: gerçek doğruluk raporlanandan yüksek.

---

## 1. Varsayılan motor, kabul edilmiş düzeltmeyi kullanmıyor — ÇÖZÜLDÜ

> Düzeltildi: commit `dbd4ccf`. `canonical_evaluate` artık
> `--matching-engine optimal` geçiyor ve motoru `metric_view`'a damgalıyor.
> M₀ ve M₀′ bu motorla yeniden ölçüldü; ikisinde de seçilen checkpoint değişti.
> Aşağıdaki 2. ve 3. bulgular da aynı düzeltmeyle kapandı.

`benchmark/reference/run.py:196` varsayılanı `greedy`. Düzeltme yazılmış,
test edilmiş, kabul edilmiş, ama çağrılmıyor.

**En küçük düzeltme:** `canonical_evaluate`'in komutuna
`--matching-engine optimal` eklemek. Sonrasında M₀ ve M₀′ yeniden
*ölçülmeli* (eğitim gerekmez) ki seri tutarlı kalsın.

## 2. Maddesiz referanslarda asimetri — YÜKSEK, altın verinin %7.1'i

`evaluate.py:642` altın taraftaki maddesiz satırı `excluded` diye atıyor, yani
recall paydasına girmiyor. `evaluate.py:900` aynı şekli tahmin tarafında
`invalid` diye **ceza** yazıyor. `evaluate.py:1338` hiç puanlanmamış belgeye
**1.0 tam puan** veriyor.

Sonuç ters: o referansı doğru bilen model sıfır alıyor, hiç cevap vermeyen model
tam puan alıyor. Gerçek altın veride **249 / 3523 referans (%7.1)** maddesiz.

`optimal` motoru bunları iki tarafta da puanlıyor — 1. maddedeki bayrak bunu da
kapatıyor.

## 3. Büyük/küçük harf katlanmıyor — YÜKSEK

`greedy` yolunda madde metni hiç katlanmıyor (`canonical_article_fields` yalnız
`optimal` tarafından çağrılıyor). `Mükerrer 257` ile `mükerrer 257` ayrı madde
sayılıyor, F1 sıfır.

Gerçek altın veride **7 vaka**: doc_47 (`Mükerrer 257`), doc_234
(`Mükerrer 298`, `Geçici 62`), doc_148 (`Geçici 19/20/21`), doc_254
(`Geçici 2`). 1. maddedeki bayrak bunu da kapatıyor.

## 4. İki normalleştirme uygulaması çelişiyor — YÜKSEK

Ölçüm motoru `annotator/reference/common.py`'yi, DQ boru hattı kendi
`normalization.py`'sini kullanıyor. Aynı madde iki farklı sonuç veriyor:

| girdi | DQ boru hattı | ölçüm motoru |
| --- | --- | --- |
| `13/a` ile `13`+bent `a` | aynı madde ✓ | **farklı madde** ✗ |
| `41 inci maddeleri` | **soyulmuyor** ✗ | `41` ✓ |

`common.py:312`'deki kural büyük/küçük harfe bakmıyor, yani **`32/A` ile `32/a`
ayrımı ölçüm tarafında hiç uygulanmıyor** — projenin kutsal saydığı kural.
Gerçek 100 belgede küçük harfli `\d+/harf` biçimi **55 kez**, büyük harfli
**11 kez** geçiyor.

**En küçük düzeltme:** `common.py:312`'yi `normalization.py:54` gibi
büyük/küçük harfe duyarlı yapmak.

## 5. Türkçe çekimler soyulmuyor — ORTA, 22 gerçek vaka

`normalization.py:48` yalnız `madde|maddesi|maddesinin|maddesinde|maddesinden`
tanıyor. Çoğul (`maddeleri`), araçlı (`maddesiyle`) ve çoğul-ilgi
(`maddelerinin`) tanınmıyor:

```
normalize_article("41 inci maddesinde") -> '41'                  ✓
normalize_article("41 inci maddeleri")  -> '41 inci maddeleri'   ✗
```

Gerçek 100 belgede **22 kez** geçiyor. `common.py:278` bunu genel bir kalıpla
(`madd\w*`) doğru yapıyor; aynı yaklaşım buraya taşınmalı.

## 6. VUK 413 politikası 26 belgenin kovasını belirliyor — ORTA

`reference_policy.py:12` her VUK 413 referansını iki taraftan da atıyor.
Gerekçe doğru: 413, özelgenin kendi dayanağı, yani boilerplate.

Ama pratikte tek yönlü: **bursiyerde 0/100, modelde 59/100**. Ve kova etkisi
büyük — **22 belge KIRMIZI'dan YEŞİL'e, 4 belge KIRMIZI'dan SARI'ya**. Yeşil
belgeler uzman incelemesine hiç girmiyor.

Sorun politikanın kendisi değil, **karar kaydının bunu yazmaması**: "iki taraf
anlaştı" diye yeşil olan belge ile "boilerplate elendi" diye yeşil olan belge
kayıtta ayırt edilemiyor.

**En küçük düzeltme:** atılan satır sayısını `RouteDecision.reasons`'a yazmak.

## 7. Dotsuz `ı` harfi siliniyor — ORTA

`common.py:188` Türkçe harfleri NFKD ile ayrıştırıp ASCII'ye indiriyor. Ama
`ı` harfinin NFKD ayrışımı yok, o yüzden **katlanmıyor, siliniyor**:

```
_normalize_turkish_key("Motorlu Taşıtlar Vergisi Kanunu")  -> 'MOTORLUTASTLARVERGISIKANUNU'   (I yok)
_normalize_turkish_key("MOTORLU TAŞITLAR VERGİSİ KANUNU")  -> 'MOTORLUTASITLARVERGISIKANUNU'
```

Aynı kanun, küçük harfle yazılınca başka kimlik alıyor. `normalization.py:107`
bunu açık bir çeviri tablosuyla doğru yapıyor.

## 8. Satır birleştirme gerçek bir eksiği silebiliyor — ORTA (bugün zararsız)

`normalization.py:234` "bu satırı başka satır karşılıyor" diyerek satır atıyor,
ama bunu tüm listeye bakarak yaptığı için modelin gerçekten kaçırdığı bir
kanun düzeyi referansını da eleyebiliyor — ve eksik, yeşile dönüşüyor.

Gerçek veride: çekirdek kimlikte **0/100**, tam kimlikte **6/100** vaka, altısı
da incelenip zararsız bulundu. Mekanizma gerçek, bugün ısırmıyor.

## 9. Küçükler

- `common.py:234` bent harfini küçültmüyor: bent `c` ile `C` ayrı kimlik.
  (`fikra` için yapılmamalı — VUK mükerrer 298'in fıkraları resmen büyük harf.)
- `common.py:40,44` iki ayrı KHK'yı (90 ve 375) aynı isme eşliyor.
- Kesme işaretli ordinal (`104'üncü maddelere`) iki tarafta da katlanmıyor,
  5 gerçek belgede var.
- `evaluate.py:303` altın dosya adını konuma göre ayrıştırıyor; aynı numaralı
  ikinci bir dosya (örn. `_backup`) bir belgenin referanslarını sessizce eziyor.
- `matching.py:219` `source_text` hâlâ bir eşitlik bozma girdisi — çekirdek F1'i
  değiştirmiyor ama fıkra/bent yan metriklerini oynatıyor. Alan hiçbir karara
  girmemeliydi.
- `matching.py:209` "onaylı taslağı yener" kuralı ölü kod: `normalize_reference`
  `status` alanını zaten siliyor. Bugünkü veride hepsi onaylı, o yüzden uyuyor.

## Temiz çıkanlar

- `32/A` kuralı **DQ boru hattında** uçtan uca doğru çalışıyor.
- Kova sıradan etkilenmiyor; benzerlik skoru hiçbir dalda okunmuyor; karantina
  yolu sağlam; iki taraf aynı işlemlerden geçiyor.
- `router.py` 100 belgenin 100'ünde depolanan kovayı yeniden üretiyor.
- NFC/NFD, kırılmaz boşluk, yumuşak tire: 100 belgede hiç geçmiyor.
- Türkçe `İ` sorunu kanun adı çözümüne sızmıyor.
- Rapor kökeni (`matching_engine` vb. her satıra damgalanıyor) sağlam.
