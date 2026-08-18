"""Prompt builders for LLM-based reference extraction."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_PROMPT_VARIANT = "few-shot-cot-v3-en"
FEW_SHOT_COT_PROMPT_VARIANT = "few-shot-cot"
FEW_SHOT_COT_V2_PROMPT_VARIANT = "few-shot-cot-v2"
EN_BASELINE_PROMPT_VARIANT = "en-baseline"
EN_NEGATIVE_GUARDED_PROMPT_VARIANT = "en-negative-guarded"
EN_RECALL_RECOVERY_PROMPT_VARIANT = "en-recall-recovery"
FEW_SHOT_COT_V3_EN_PROMPT_VARIANT = "few-shot-cot-v3-en"

ENGLISH_PROMPT_VARIANTS = (
    EN_BASELINE_PROMPT_VARIANT,
    EN_NEGATIVE_GUARDED_PROMPT_VARIANT,
    EN_RECALL_RECOVERY_PROMPT_VARIANT,
    FEW_SHOT_COT_V3_EN_PROMPT_VARIANT,
)

SUPPORTED_PROMPT_VARIANTS = (
    DEFAULT_PROMPT_VARIANT,
    FEW_SHOT_COT_PROMPT_VARIANT,
    FEW_SHOT_COT_V2_PROMPT_VARIANT,
    *ENGLISH_PROMPT_VARIANTS,
)

FEW_SHOT_PROMPT_VARIANTS = tuple(
    variant for variant in SUPPORTED_PROMPT_VARIANTS if variant != DEFAULT_PROMPT_VARIANT
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLE_SET_DIR = PROJECT_ROOT / "benchmark" / "reference" / "example_sets"

RULES_BLOCK = """Sen bir Türk hukuk metni analisti uzmanısın. Görevin vergi özelgesi metinlerindeki kanun referanslarını yapılandırılmış tabloya dönüştürmek.

Metindeki tüm referanslar TEK TİPTE dönecek:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## TİP: KANUN MADDE REFERANSI (type = "kanun_madde_referansi")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hiyerarşi: Kanun → Madde → Fıkra → Bent

### Alanlar:
- **type**: "kanun_madde_referansi" (sabit)
- **kanun_no**: Kanunun numarası (örn. "213", "3065", "488"). Sadece sayısal değer.
- **kanun_ad**: Kanunun tam adı, çekimsiz hali (örn. "Vergi Usul Kanunu"). Suffix'leri kaldır: "Kanununun" → "Kanunu".
- **madde**: SADECE madde numarası ve varsa prefix (örn. "298", "mükerrer 298", "geçici 25", "Ek 6"). "madde", "maddesi" gibi kelimeler DAHİL ETME. Spesifik madde yoksa boş string.
- **fikra**: Fıkra tanımlayıcısı varsa (örn. "A", "1", "2"). Parantezi kaldır: "(A)" → "A". Yoksa boş string.
- **bent**: Bent tanımlayıcısı varsa (örn. "a", "b", "1", "g", "2/b"). Parantezi kaldır: "(g)" → "g". Yoksa boş string.
- **source_text**: Metinde bu referansın geçtiği orijinal metin parçası (20-80 karakter).

### Slash notasyonu (Madde referansında):
| Metindeki ifade | madde | fikra | bent |
|-----------------|-------|-------|------|
| "16 ncı maddesi" | 16 | | |
| "16/1" | 16 | 1 | |
| "16/1-a" | 16 | 1 | a |
| "13/a maddesi" | 13/a | | |
| "17/4-e" | 17 | 4 | e |

Kurallar:
- Slash'ten önceki sayı → **madde**
- Slash'ten sonra sadece sayı varsa (örn. /1, /4) → **fıkra**
- Slash'ten sonra sayı-harf varsa (örn. /1-a, /4-e) → sayı **fıkra**, harf **bent**
- `13/a maddesi` gibi `sayı/tek_harf + maddesi` formu varsa bunu **bent değil madde** kabul et: `madde="13/a"`, `fikra=""`, `bent=""`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## EK HATA KORUMALARI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Tebliğ, Yönetmelik, Karar, Genelge, Sirküler gibi ikincil düzenlemeleri tek başına referans objesi yapma.
- Kanun maddesi içindeki kopyalanmış `(1)`, `(2)`, `a)`, `b)` liste içeriklerini tek başına sahte referanslara dönüştürme.
- Aynı local mention içinde spesifik madde/fıkra/bent atfı varsa ayrıca generic `kanun` objesi üretme.
- Belge genelinde aynı kanuna ait en az bir spesifik `madde/fıkra/bent` referansı varsa ayrıca generic law-only satır üretme.
- Belge genelinde aynı `kanun_no / kanun_ad / madde / fikra / bent` tuple'ı tekrar ederse yalnız 1 kez üret; farklı `source_text` ile tekrarlansa bile ayrı obje oluşturma.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## TABLO / CETVEL / LİSTE ATIFLARI İÇİN ZORUNLU KURAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Artık **tablo_referansi** kullanılmaz. Çıktıda **tablo, bolum, kalem, alt_bent** gibi alanlar ASLA bulunmaz.

Eğer metinde tablo/cetvel/liste atfı varsa (örn. "5018 sayılı Kanunun eki I, II, III ve IV sayılı cetveller"):
- Bu atfı yine **kanun_madde_referansi** tipinde yaz.
- Tablo/cetvel numarasını ve bölüm/part etiketlerini alanlara taşıma (örn. `IV`, `I. Akitlerle ilgili kağıtlar`, `(1) sayılı tablo`).
- Açık tablo/tarife kalemi varsa kalem kimliğini koru: `IV/1-a fıkrası` → `madde="1"`, `fikra=""`, `bent="a"`; `A/1 fıkrası` → `madde="A"`, `fikra=""`, `bent="1"`.
- Yalnız bağlı kanun ve tablo/cetvel listesi varsa, spesifik kalem yoksa **madde="", fikra="", bent=""** bırak.

Çoklu/ardışık tablo-cetvel atıflarında mükerrer kayıt üretme:
- Aynı kanuna ait "I, II, III, IV sayılı cetveller" gibi ifadeler için yalnızca **1 adet** obje oluştur.
- Her cetvel için ayrı obje oluşturma.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## ORTAK KURALLAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **Kanun adı kuralları**:
   - Metinde açıkça yazılmışsa yaz. Yazılmamışsa metnin öncesinde aynı kanun_no ile geçmişse o adı yaz.
   - Hiç geçmemişse boş bırak. Kanun adını UYDURMA.
   - Metinde yalnız `6728 sayılı Kanun`, `6745 sayılı Kanun` gibi generic bir yüzey form varsa ve güvenli tam ad yoksa `kanun_ad=""` bırak. ASLA yalnız `Kanun` yazma.
   - "Aynı Kanunun", "anılan Kanunun" gibi atıflar → önceki bağlamda ad varsa yaz, yoksa boş.
   - Kanun adı HER ZAMAN çekimsiz: "Kanununun" → "Kanunu"

2. Madde/fıkra/bent içeren normal kanun atıflarında her referans için ayrı satır oluştur.

3. Değişiklik/ekleme zincirlerini AYIR:
   - `6728 sayılı Kanunun 41 inci maddesiyle değişen (d) fıkra`
   - `6745 sayılı Kanunun 51 inci maddesi ile ... eklenen (8) numaralı bent`
   gibi yapılarda değişikliği yapan kanunun kendi maddesi AYRI bir referanstır.
   `değişen`, `eklenen`, `değiştirilen`, `yürürlükten kaldırılan` sonrası gelen hedef madde/fıkra/bent ise yakınındaki hedef/host kanuna bağlanmalıdır; amendment kanununa yanlış taşınmamalıdır.

4. Tablo/cetvel/liste atıflarında aynı kanun tekrar ediyorsa tek satır üret (deduplikasyon zorunlu).

5. Kanun numarası/adı tek başına geçiyorsa:
   - Aynı kanunun belge genelinde hiç spesifik `madde/fıkra/bent` referansı yoksa en fazla 1 generic law-only satır üret.
   - Aynı kanun için belge genelinde en az bir spesifik referans varsa generic law-only satır üretme.

6. Belge-seviyesi tekrar temizliği:
   - Aynı `kanun_no / kanun_ad / madde / fikra / bent` tuple'ı belge içinde birden fazla kez geçse bile tek satır tut.
   - Ancak farklı `madde/fıkra/bent` kombinasyonları ayrı satır olarak korunur.

7. Footer referansları ("(*) Bu Özelge 213 sayılı Vergi Usul Kanununun 413.maddesine dayanılarak verilmiştir.") DAHİL ET.
"""

HIDDEN_COT_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## İÇSEL ÇALIŞMA TALİMATI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Bu bölümdeki adımları sessizce uygula; ASLA çıktı olarak reasoning, düşünce özeti, adım listesi, not, açıklama veya markdown üretme. Final cevapta yalnızca geçerli JSON ver.

İçsel kontrol sırası:
1. Metindeki tüm aday kanun referanslarını tara.
2. Her aday için kanun_no, kanun_ad, madde, fikra, bent ayrımını yap.
3. Slash/composite notasyonu normalize et.
4. Anaforik atıfları mevcut bağlamdan resolve et; emin değilsen kanun_ad veya kanun_no alanını boş bırak, uydurma yapma.
5. Tablo/cetvel/liste atıflarında tek satır deduplikasyonu uygula.
6. Belge genelinde aynı kanun için spesifik referans varken generic law-only satır bırakma.
7. Belge genelinde aynı normalize tuple tekrar ediyorsa tek satıra indir.
8. Bir kanun yalnız generic geçiyorsa en fazla 1 generic law-only satır bırak.
9. Final JSON'u şu kontrollerle sessizce doğrula:
   - geçerli schema mı
   - gereksiz duplicate var mı
   - belge genelinde aynı tuple iki kez kaldı mı
   - aynı kanun için spesifik referans varken generic law-only satır kaldı mı
   - bent/fıkra alanları gereksiz boş bırakıldı mı
   - kanun adı veya numarası uyduruldu mu
"""

FEW_SHOT_COT_V2_RULES_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## V2 EK DETERMINISM KURALLARI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Bu varyantta amaç daha deterministik ve tekrarlanabilir çıktı üretmektir.

1. Aynı metin için her çalıştırmada AYNI en muhafazakâr çözümü seç:
   - emin değilsen alanı boş bırak
   - bir çalıştırmada doldurup diğerinde boş bırakma

2. Kanun adı/numarası çözümleme önceliği:
   - açıkça yazılan bilgi
   - aynı cümledeki yakın bağlam
   - anaforik bağlam ("Aynı Kanunun", "anılan Kanun")
   - hala emin değilsen boş string

3. Scope disi materyalleri referans olarak cikarma:
   - Teblig, Yonetmelik, Genel Teblig, Ozelge, Sirkuler, Genelge, Bakanlar Kurulu Karari, Cumhurbaskani Karari, Karar, Tarife, Rehber gibi ikincil duzenlemeleri AYRI bir obje olarak yazma.
   - Bunlar yalnizca bir kanun referansinin acik kanit baglamiysa yardimci baglamdir; tek basina referans objesi olusturma.
   - Ozellikle `32.1.2.5`, `A/2-b`, `IV. Ticari ve medeni isler`, `Karar eki cetvel`, `Genel Tebligin 1.2.3 bolumu` gibi numaralandirmalari kanun maddesi/fikrasi sanma.

4. Bent ve slash cozulemede deterministik ayrim:
   - `16/1-a` → madde=`16`, fikra=`1`, bent=`a`
   - `17/4-e` → madde=`17`, fikra=`4`, bent=`e`
   - `(2/b) numarali bent` → bent=`2/b`
   - `13/a maddesi` veya `13/a maddesinde` gibi `sayi/tek_harf` dogrudan madde formunda geciyorsa → madde=`13/a`, fikra=``, bent=``
   - `IV/1-a fıkrası` gibi tablo-cetvel baglaminda `IV` tablo bilgisidir; alanlara yazilmaz

5. Çoklu bent listelerinde acikca sayilan her bent icin ayri satir uret:
   - `(1), (2), (4) ve (7) numaralı bentleri` → 4 ayrı obje
   - ancak metin tek bir bent mi çoklu bent mi belirsizse çoğaltma yapma

6. Tablo/cetvel/liste atiflarinda:
   - tablo numarasını ve bölüm/part etiketini alanlara taşıma
   - açık tablo/tarife kalemi varsa bölüm etiketini atıp kalem kimliğini koru (`IV/1-a` → `madde=1`, `bent=a`)
   - aynı kanuna bağlı çoklu cetvel atfını tek satıra indir
   - spesifik madde yoksa `madde="", fikra="", bent=""`

7. Kopyalanmis madde metni ile hukuki atif arasini ayir:
   - Kanun maddesi icindeki alinti/icerik listeleri hukuki atif degildir.
   - Ornegin bir maddede alinti halinde `(1) ... (2) ... (3) ...` veya `a) ... b) ... c) ...` diye sayilan icerik ogelerini AYRI referanslar olarak cikarma.
   - Ancak cumle acikca `... bendinde`, `... numarali bendinde`, `... fikrasinda`, `... IV/53 numarali fıkrasında` diyorsa bu bir referanstir ve cikartilmalidir.

8. Ayni kanuna ait genel ve ozel atif ayni yerde geciyorsa en spesifik olanı tercih et:
   - `5811 sayılı Kanunun 3 üncü maddesinin beşinci fıkrası` varken ayrica `5811 sayılı Kanun kapsamında` objesi üretme.
   - `Kanunun 36 ncı maddesinde ... (1) ... (2) ...` gibi ifadelerde once asil madde atfini koru; alinti icerigindeki liste ogelerini referanslastirma.

9. Belge-seviyesi kompaksiyon:
   - Metinde ayni hukuki tuple farkli cumlelerde/farkli baglamlarda tekrar acikca anilsa bile tek obje olarak tut.
   - `source_text` farki tek basina yeni obje sebebi degildir.
   - Ayni kanun icin belge genelinde en az bir spesifik referans varsa generic law-only obje tutma.
   - `I, II, III, IV sayili cetveller` gibi ayni kanuna bagli tablo listeleri yine tek satirda birlestirilir.

10. Kaynak metin seçimi:
   - mümkün olan en kısa ama ayırt edici span'i seç
   - aynı referans için bazen dar bazen geniş source_text seçme

11. Bent ve fikra hatalarını azaltmak icin son kontrol:
   - `numaralı bent`, `bendi`, `alt bent`, `bendinde` goruyorsan bent alanini bos birakma
   - `birinci/ikinci/... fıkra` veya `/4` gibi acik ipucu varsa fikrayi kaybetme
   - `13/a maddesi` turunde tek-harfli slashli maddeyi yanlislikla fikra=`a` yapma
"""

HIDDEN_COT_V2_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## İÇSEL ÇALIŞMA TALİMATI (V2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Bu bölümdeki adımları sessizce uygula; ASLA reasoning, açıklama, not veya markdown üretme. Final cevapta yalnızca geçerli JSON ver.

İçsel kontrol sırası:
1. Önce açık referansları çıkar, sonra anaforik çözümleme yap.
2. Once scope disi ikincil duzenlemeleri ele: Teblig, Yonetmelik, Karar, BKK, Genelge gibi yapilari kanun referansi olarak yazma.
3. Bent/slash/composite yapıları soldan sağa ve en muhafazakâr kuralla ayır.
4. `sayi/tek_harf + maddesi` formunu madde icinde koru; yanlislikla fikraya tasima.
5. Çoklu bent listelerinde yalnız açıkça sayılan bentleri çoğalt.
6. Kanun maddesi icindeki alinti/liste iceriklerini referans gibi ayiklama.
7. Tablo/cetvel/liste atıflarında kanunu koru, tablo bilgisini alanlara taşıma.
8. Belge genelinde ayni tuple tekrar ediyorsa tek satirda birlestir.
9. Aynı kanun için spesifik referans varsa generic law-only satırları temizle; hiç spesifik yoksa en fazla 1 generic satır bırak.
10. Emin olmadığın alanı boş bırak; alan uydurma veya kanun adı tahmini yapma.
11. Final JSON'u sessizce şu kontrollerle doğrula:
   - schema geçerli mi
   - scope disi teblig/karar/yonetmelik objesi var mi
   - quoted liste iceriginden uydurma bent/fıkra objesi cikmis mi
   - bent/fıkra ayrımı kurala uygun mu
   - anaforik çözümleme aynı bağlamla tutarlı mı
   - aynı kanun için spesifik referans varken generic law-only obje kaldı mı
   - belge genelinde aynı tuple iki kez kaldı mı
   - source_text kısa ama ayırt edici mi
"""

ENGLISH_ROLE_AND_SCHEMA_BLOCK = """You are an expert analyst of Turkish tax rulings and Turkish legal references.

Your task is to extract structured law references from Turkish legal text and return only JSON.

All extracted references must use a single object type:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## TYPE: LAW ARTICLE REFERENCE (`type = "kanun_madde_referansi"`)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hierarchy: Law -> Article -> Paragraph -> Subparagraph

Fields:
- `type`: always `"kanun_madde_referansi"`
- `kanun_no`: numeric law number only, for example `"213"`, `"3065"`, `"488"`
- `kanun_ad`: exact law name copied from the text or resolved from nearby context; keep the Turkish surface form, only normalize inflectional suffixes
- `madde`: article identifier only, for example `"298"`, `"mükerrer 298"`, `"geçici 25"`, `"Ek 6"`; do not include words like `madde` or `maddesi`
- `fikra`: paragraph identifier if present, for example `"1"`, `"2"`, `"A"`
- `bent`: subparagraph identifier if present, for example `"a"`, `"g"`, `"2/b"`
- `source_text`: short original Turkish span from the document that directly supports the extracted reference

Slash rules:
| Surface form | madde | fikra | bent |
|--------------|-------|-------|------|
| `16 ncı maddesi` | 16 | | |
| `16/1` | 16 | 1 | |
| `16/1-a` | 16 | 1 | a |
| `17/4-e` | 17 | 4 | e |
| `(2/b) numaralı bent` | | | 2/b |
| `13/a maddesi` | 13/a | | |

Never translate:
- `kanun_ad`
- `source_text`
- Turkish reference surface forms

General rules:
1. If a specific article / paragraph / subparagraph is explicitly present, prefer that specific reference.
2. If a law appears only as a generic law mention and that law has no specific article / paragraph / subparagraph reference anywhere in the document, emit at most one generic law-only object for that law.
3. Footer references such as `213 sayılı Vergi Usul Kanununun 413. maddesine dayanılarak verilmiştir` are valid and must be extracted.
4. Table / cetvel / attachment references still use the same object type; keep the linked law. Do not store table numbers or section labels in `madde`, `fikra`, or `bent`; when an explicit table/tariff item exists, keep only the item identity after dropping the table/section label (`IV/1-a` -> `madde=1`, `bent=a`; `A/1` -> `madde=A`, `bent=1`).
5. In amendment chains, keep the amending-law article separate from the host-law target reference. Example pattern: `6745 sayılı Kanunun 51 inci maddesi ile ... eklenen (8) numaralı bent`. Here `6745/51` is one reference, while `(8) bent` belongs to the nearby host law being amended, not automatically to `6745`.
6. If the document already contains a specific tuple for the same law, do not also keep a generic law-only object for that law.
7. If the exact same `kanun_no / kanun_ad / madde / fikra / bent` tuple appears multiple times in the document, return it only once even if the supporting span appears again elsewhere.
"""

ENGLISH_BASELINE_RULES_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## ENGLISH BASELINE EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use a progressive extraction order:
1. Find explicit law references.
2. Resolve article / paragraph / subparagraph structure.
3. Resolve local anaphora such as `Aynı Kanunun` or `anılan Kanun`.
4. Apply document-level compaction: keep one row per exact legal tuple and suppress generic law-only rows when a more specific tuple exists for the same law.

Law-name policy:
- If `kanun_ad` is explicitly written, copy it from the text and only normalize suffixes.
- Do not paraphrase a law name into a cleaner or shorter Turkish form.
- If the law name is not explicit, resolve it only from close local context.
- If the surface form is only a generic `X sayılı Kanun` and the full title is not safely recoverable, keep `kanun_ad` empty; never output the generic filler `Kanun`.
- If a well-known tax-law abbreviation uniquely identifies the statute (`VUK`, `GVK`, `KVK`, `KDVK`, `KDV Kanunu`, `ÖTVK`, `ÖTV Kanunu`, `DVK`), resolve it to the canonical full Turkish law name and fill `kanun_no` when unambiguous.
- If still uncertain, keep `kanun_ad` empty instead of inventing it.

Context-carryover policy:
- Resolve a law or article only from explicit local evidence, a very close same-law anaphora, or an unambiguous abbreviation.
- Do not attach a later bare article mention to an earlier law just because a different sentence mentioned that law before.
- If a later sentence only says something like `5 inci maddede` and there is no clear same-law signal nearby, leave the law fields empty or skip the object instead of guessing.

Source-text policy:
- Use the shortest Turkish span that still uniquely supports the extracted tuple.
- Keep source_text consistent and evidence-oriented.
"""

ENGLISH_NEGATIVE_GUARD_RULES_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## FAILURE-DRIVEN NEGATIVE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Do not create standalone reference objects for secondary regulations:
- Tebliğ
- Yönetmelik
- Karar
- Bakanlar Kurulu Kararı
- Cumhurbaşkanı Kararı
- Kanun Hükmünde Kararname
- Genelge
- Sirküler
- Tarife

Treat those items only as supporting context unless the document explicitly cites a real law article reference tied to them.

Do not turn quoted article content into fake references:
- numbered list items like `(1)`, `(2)`
- lettered list items like `a)`, `b)`
- copied internal clauses inside quoted article text

If a local mention or the broader document already contains a specific article / paragraph / subparagraph for the same law, do not also emit a generic `law-only` object for that law.
"""

ENGLISH_RECALL_RECOVERY_RULES_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## RECALL RECOVERY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Do not become overly conservative:
- If a concrete article reference is explicit, extract it even when nearby secondary regulations also appear.
- If a different legal tuple is explicitly present, keep it; do not collapse distinct tuples into a weaker generic object.
- If the exact same legal tuple is explicitly repeated in a different sentence or clause, keep only one row for that tuple across the whole document.
- Preserve article-only references such as `413 üncü maddesi`, `32 nci madde`, `geçici 7 nci madde`.
- Preserve paragraph-level references such as `birinci fıkra`, `/4`, `(A) fıkrası`.
- Preserve subparagraph references when the text explicitly says `bendinde`, `numaralı bendi`, `alt bendi`.
- Preserve collapsed article-range references as a single row.
  - `1 ila 24 üncü maddeleri`, `1-24 üncü maddeleri`, `1'den 24'e kadar maddeleri` -> one row, not one row per integer.
  - Normalize such article ranges into a collapsed `madde` value such as `1 ila 24`.
- Expand only true explicit enumerations.
  - `61, 94, 103 ve 104 üncü maddeleri` -> separate rows for `61`, `94`, `103`, `104`.

Strict-match stability rules:
- Do not replace a specific article with a generic law-only object.
- Do not collapse `madde/fıkra/bent` into a less specific tuple.
- Keep exact law-name wording from the text whenever possible because benchmark matching is strict.
"""

ENGLISH_FINAL_LOCK_RULES_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## FINAL V3-EN LOCK RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before returning the final JSON, silently confirm:
- no secondary-regulation object leaked into output
- no quoted list item was mistaken for a real legal reference
- no generic law-only object replaced a more specific local reference
- no generic law-only object survived when the same law had a more specific tuple elsewhere in the document
- no exact legal tuple survived twice in the same document
- no `13/a maddesi` item was split into `fikra=a`
- no collapsed article range such as `1 ila 24` was incorrectly expanded into multiple article rows
- no correct Turkish law name was rewritten into a different surface form
- no later bare article mention was attached to an earlier law without clear same-law evidence
- no unambiguous tax-law abbreviation was left in a weaker non-canonical identity
"""

ENGLISH_NEGATIVE_MINI_EXAMPLES_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## NEGATIVE MINI-EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mini-example 1:
Input fragment: `5520 sayılı Kurumlar Vergisi Kanununun 32 nci maddesi ile 1 Seri No.lu Kurumlar Vergisi Genel Tebliğinin 32.1.2.5 bölümü`
Correct behavior: extract the law article reference; do not emit a standalone `Genel Tebliğ` object.

Mini-example 2:
Input fragment: `Kanunun 36 ncı maddesinde yer alan (1), (2), (3), (4) bentleri aşağıdaki gibidir: ...`
Correct behavior: keep the real `36 ncı madde` reference; do not turn copied list content into separate fake bent references unless the sentence explicitly says `numaralı bendi` as a legal citation.

Mini-example 3:
Input fragment: `488 sayılı Damga Vergisi Kanununun 3 üncü ve 9 uncu maddeleri`
Correct behavior: emit article-level objects for `madde=3` and `madde=9`; do not replace them with a single generic `488 sayılı Damga Vergisi Kanunu` object.

Mini-example 4:
Input fragment: `4572 sayılı Tarım Satış Kooperatif ve Birlikleri Hakkında Kanunun 6 ncı maddesi`
Correct behavior: keep `kanun_ad` as `Tarım Satış Kooperatif ve Birlikleri Hakkında Kanun`; do not rewrite it as `... Kanunu`.

Mini-example 5:
Input fragment: `13/a maddesi ile 17/4-e maddesi`
Correct behavior: `13/a maddesi` stays `madde=13/a`; `17/4-e` becomes `madde=17, fikra=4, bent=e`.

Mini-example 6:
Input fragment: `6745 sayılı Kanunun 51 inci maddesi ile Özel Tüketim Vergisi Kanununun 7 nci maddesinin birinci fıkrasına eklenen (8) numaralı bent`
Correct behavior: emit one reference for `6745 / 51`; emit a separate host-law reference for `Özel Tüketim Vergisi Kanunu / 7 / 1 / 8`; do not attach `(8) numaralı bent` to `6745`.

Mini-example 7:
Input fragment: `213 sayılı Vergi Usul Kanunu uygulanır. Sonraki değerlendirmede 5 inci maddede belirtilen usul izlenir.`
Correct behavior: do not assume that `5 inci maddede` belongs to `213 sayılı Vergi Usul Kanunu` unless a clear same-law signal exists nearby.

Mini-example 8:
Input fragment: `KDV Kanununun 17/4-a maddesinde yer alan istisna`
Correct behavior: use the canonical law identity: `kanun_no=3065`, `kanun_ad=Katma Değer Vergisi Kanunu`, `madde=17`, `fikra=4`, `bent=a`.

Mini-example 9:
Input fragment: `4760 sayılı Özel Tüketim Vergisi Kanununun 7/2-c maddesi uygulanır. Aynı Kanunun 7/2-c maddesi tekrar hatırlatılır. Ayrıca Özel Tüketim Vergisi Kanunu kapsamında değerlendirme yapılır.`
Correct behavior: keep only one row for `4760 / 7 / 2 / c`; do not keep an extra generic law-only object.

Mini-example 10:
Input fragment: `4958 sayılı Sosyal Sigortalar Kanununun 1 ila 24 üncü maddeleri`
Correct behavior: emit one collapsed article-range row with `madde=1 ila 24`; do not expand it into `1`, `2`, `3`, ..., `24`.

Mini-example 11:
Input fragment: `193 sayılı Gelir Vergisi Kanununun 61, 94, 103 ve 104 üncü maddeleri`
Correct behavior: emit four separate article rows for `61`, `94`, `103`, and `104`; do not collapse a true enumeration into a range.
"""

ENGLISH_BASELINE_CHECKLIST_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SILENT SELF-CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Apply this short checklist silently before answering:
1. Is every object schema-valid?
2. Did I preserve the original Turkish law name instead of paraphrasing it?
3. Did I keep the most specific explicit reference in the local mention?
4. Did I leave a generic law-only object even though the same law had a more specific tuple elsewhere in the document?
5. Did I leave the same exact legal tuple twice in the same document?
6. Is the final answer JSON only?
"""

ENGLISH_NEGATIVE_GUARDED_CHECKLIST_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SILENT SELF-CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Apply this short checklist silently before answering:
1. Is every object schema-valid?
2. Did any Tebliğ / Yönetmelik / Karar leak into the output as a standalone object?
3. Did any quoted `(1) / (2) / a) / b)` list item become a fake reference?
4. Did I leave a generic law-only object even though the same law had a more specific tuple elsewhere in the document?
5. Did I leave the same exact legal tuple twice in the same document?
6. Is the final answer JSON only?
"""

ENGLISH_RECALL_RECOVERY_CHECKLIST_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SILENT SELF-CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Apply this short checklist silently before answering:
1. Is every object schema-valid?
2. Did I accidentally drop any explicit `madde`, `fıkra`, or `bent` reference?
3. Did I replace a specific local reference with a generic law-only object?
4. Did I leave a generic law-only object even though the same law had a more specific tuple elsewhere in the document?
5. Did I preserve the exact Turkish law-name surface form when it was explicit?
6. Is the final answer JSON only?
"""

ENGLISH_FINAL_CHECKLIST_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## FINAL SILENT CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Silently verify:
1. No secondary-regulation leak
2. No quoted-list hallucination
3. No generic-law drift
4. No duplicate exact legal tuple inside the same document
5. No lost article / paragraph / subparagraph detail
6. No Turkish law-name wording drift
7. JSON only
"""

OUTPUT_CONTRACT_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## JSON FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Yanıtını SADECE aşağıdaki JSON formatında ver, başka açıklama ekleme:

{
  "references": [
    {
      "type": "kanun_madde_referansi",
      "kanun_no": "213",
      "kanun_ad": "Vergi Usul Kanunu",
      "madde": "mükerrer 298",
      "fikra": "A",
      "bent": "1",
      "source_text": "213 sayılı Vergi Usul Kanununun mükerrer 298 inci maddesinin (A) fıkrasının (1) numaralı bendinde"
    },
    {
      "type": "kanun_madde_referansi",
      "kanun_no": "5018",
      "kanun_ad": "Kamu Mali Yönetimi ve Kontrol Kanunu",
      "madde": "",
      "fikra": "",
      "bent": "",
      "source_text": "5018 sayılı Kanunun eki I, II, III ve IV sayılı cetvellerde yer alan kamu idarelerinde"
    },
    {
      "type": "kanun_madde_referansi",
      "kanun_no": "488",
      "kanun_ad": "Damga Vergisi Kanunu",
      "madde": "1",
      "fikra": "",
      "bent": "",
      "source_text": "488 sayılı Damga Vergisi Kanununun 1 inci maddesinde"
    }
  ]
}"""

ENGLISH_OUTPUT_CONTRACT_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## JSON OUTPUT CONTRACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return JSON only. Do not add explanations, reasoning, Markdown, or prose.

{
  "references": [
    {
      "type": "kanun_madde_referansi",
      "kanun_no": "213",
      "kanun_ad": "Vergi Usul Kanunu",
      "madde": "mükerrer 298",
      "fikra": "A",
      "bent": "1",
      "source_text": "213 sayılı Vergi Usul Kanununun mükerrer 298 inci maddesinin (A) fıkrasının (1) numaralı bendinde"
    },
    {
      "type": "kanun_madde_referansi",
      "kanun_no": "5018",
      "kanun_ad": "Kamu Mali Yönetimi ve Kontrol Kanunu",
      "madde": "",
      "fikra": "",
      "bent": "",
      "source_text": "5018 sayılı Kanunun eki I, II, III ve IV sayılı cetvellerde yer alan kamu idarelerinde"
    }
  ]
}"""

LANGEXTRACT_OUTPUT_CONTRACT_BLOCK_TR = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## LANGEXTRACT CIKTI KONTRATI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Bu varyantta benchmark satiri semasi degil, `LangExtract` extraction wrapper semasi kullanilmalidir.

Gecerli JSON sekli:

{
  "extractions": [
    {
      "kanun_madde_referansi": "metindeki kanit span",
      "kanun_madde_referansi_attributes": {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "mükerrer 298",
        "fikra": "A",
        "bent": "1"
      }
    }
  ]
}

Zorunlu kurallar:
- Extraction key olarak yalniz `kanun_madde_referansi` kullan.
- Kaynak span `source_text` alanina degil, `kanun_madde_referansi` alanina yazilir.
- Yapilandirilmis alanlar yalniz `kanun_madde_referansi_attributes` altina yazilir.
- `type` alani yazma.
- `source_text` alani yazma.
- Her extraction icin en az bir yapilandirilmis alan (`kanun_no`, `kanun_ad`, `madde`, `fikra`, `bent`) dolu olmali.
- Yalniz JSON dondur. Markdown fence kullanma.
"""

LANGEXTRACT_OUTPUT_CONTRACT_BLOCK_EN = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## LANGEXTRACT OUTPUT CONTRACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Do not return the benchmark row schema directly. Use the `LangExtract` extraction wrapper schema.

Valid JSON shape:

{
  "extractions": [
    {
      "kanun_madde_referansi": "supporting span from the text",
      "kanun_madde_referansi_attributes": {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "mükerrer 298",
        "fikra": "A",
        "bent": "1"
      }
    }
  ]
}

Required rules:
- Use only `kanun_madde_referansi` as the extraction key.
- Put the supporting span in `kanun_madde_referansi`, not in `source_text`.
- Put structured fields only under `kanun_madde_referansi_attributes`.
- Do not output a `type` field.
- Do not output a `source_text` field.
- Every extraction must have at least one non-empty structured field among `kanun_no`, `kanun_ad`, `madde`, `fikra`, `bent`.
- Return JSON only. Do not use markdown fences.
"""

PROMPT_EXAMPLE_SOURCE_REGISTRY = {
    FEW_SHOT_COT_PROMPT_VARIANT: EXAMPLE_SET_DIR / "few_shot_cot_examples.json",
    FEW_SHOT_COT_V2_PROMPT_VARIANT: EXAMPLE_SET_DIR / "few_shot_cot_v2_examples.json",
    EN_BASELINE_PROMPT_VARIANT: EXAMPLE_SET_DIR / "few_shot_cot_v3_en_examples.json",
    EN_NEGATIVE_GUARDED_PROMPT_VARIANT: EXAMPLE_SET_DIR / "few_shot_cot_v3_en_examples.json",
    EN_RECALL_RECOVERY_PROMPT_VARIANT: EXAMPLE_SET_DIR / "few_shot_cot_v3_en_examples.json",
    FEW_SHOT_COT_V3_EN_PROMPT_VARIANT: EXAMPLE_SET_DIR / "few_shot_cot_v3_en_examples.json",
}


def normalize_prompt_variant(value: str) -> str:
    """Normalize prompt variant names while keeping a strict supported set."""
    normalized = str(value or DEFAULT_PROMPT_VARIANT).strip().lower().replace("_", "-")
    if normalized not in SUPPORTED_PROMPT_VARIANTS:
        supported = ", ".join(SUPPORTED_PROMPT_VARIANTS)
        raise ValueError(f"Unsupported LLM prompt variant: {value!r}. Supported: {supported}")
    return normalized


def _is_english_prompt_variant(variant: str) -> bool:
    """Return whether a prompt variant uses the English control prompt family."""
    return normalize_prompt_variant(variant) in ENGLISH_PROMPT_VARIANTS


def _build_reasoning_block(variant: str) -> str:
    """Return the hidden reasoning/checklist block for advanced variants."""
    normalized = normalize_prompt_variant(variant)
    if normalized == FEW_SHOT_COT_PROMPT_VARIANT:
        return HIDDEN_COT_BLOCK
    if normalized == FEW_SHOT_COT_V2_PROMPT_VARIANT:
        return HIDDEN_COT_V2_BLOCK
    if normalized == EN_BASELINE_PROMPT_VARIANT:
        return ENGLISH_BASELINE_CHECKLIST_BLOCK
    if normalized == EN_NEGATIVE_GUARDED_PROMPT_VARIANT:
        return ENGLISH_NEGATIVE_GUARDED_CHECKLIST_BLOCK
    if normalized == EN_RECALL_RECOVERY_PROMPT_VARIANT:
        return ENGLISH_RECALL_RECOVERY_CHECKLIST_BLOCK
    if normalized == FEW_SHOT_COT_V3_EN_PROMPT_VARIANT:
        return _assemble_prompt(
            ENGLISH_RECALL_RECOVERY_CHECKLIST_BLOCK,
            ENGLISH_FINAL_CHECKLIST_BLOCK,
        )
    return ""


def _build_variant_rules_block(variant: str) -> str:
    """Return extra rule guidance for prompt variants that need it."""
    normalized = normalize_prompt_variant(variant)
    if normalized == FEW_SHOT_COT_V2_PROMPT_VARIANT:
        return FEW_SHOT_COT_V2_RULES_BLOCK
    if normalized == EN_BASELINE_PROMPT_VARIANT:
        return ENGLISH_BASELINE_RULES_BLOCK
    if normalized == EN_NEGATIVE_GUARDED_PROMPT_VARIANT:
        return _assemble_prompt(
            ENGLISH_BASELINE_RULES_BLOCK,
            ENGLISH_NEGATIVE_GUARD_RULES_BLOCK,
            ENGLISH_NEGATIVE_MINI_EXAMPLES_BLOCK,
        )
    if normalized == EN_RECALL_RECOVERY_PROMPT_VARIANT:
        return _assemble_prompt(
            ENGLISH_BASELINE_RULES_BLOCK,
            ENGLISH_NEGATIVE_GUARD_RULES_BLOCK,
            ENGLISH_RECALL_RECOVERY_RULES_BLOCK,
            ENGLISH_NEGATIVE_MINI_EXAMPLES_BLOCK,
        )
    if normalized == FEW_SHOT_COT_V3_EN_PROMPT_VARIANT:
        return _assemble_prompt(
            ENGLISH_BASELINE_RULES_BLOCK,
            ENGLISH_NEGATIVE_GUARD_RULES_BLOCK,
            ENGLISH_RECALL_RECOVERY_RULES_BLOCK,
            ENGLISH_FINAL_LOCK_RULES_BLOCK,
            ENGLISH_NEGATIVE_MINI_EXAMPLES_BLOCK,
        )
    return ""


@lru_cache(maxsize=None)
def load_prompt_example_records(variant: str) -> tuple[dict[str, Any], ...]:
    """Load canonical prompt example records from source-controlled JSON files."""
    normalized = normalize_prompt_variant(variant)
    source_path = PROMPT_EXAMPLE_SOURCE_REGISTRY.get(normalized)
    if source_path is None:
        return ()

    with source_path.open("r", encoding="utf-8") as file:
        payload: Any = json.load(file)

    raw_examples: Any
    if isinstance(payload, dict):
        raw_examples = payload.get("examples", [])
    else:
        raw_examples = payload

    if not isinstance(raw_examples, list):
        raise ValueError(f"Invalid prompt example payload in {source_path}")

    normalized_examples: list[dict[str, Any]] = []
    for item in raw_examples:
        if isinstance(item, dict):
            normalized_examples.append(item)
    return tuple(normalized_examples)


@lru_cache(maxsize=None)
def _load_prompt_examples(variant: str) -> tuple[str, ...]:
    """Load curated few-shot examples from a source-controlled JSON file."""
    normalized = normalize_prompt_variant(variant)

    use_english_labels = _is_english_prompt_variant(normalized)
    formatted_examples: list[str] = []
    for index, item in enumerate(load_prompt_example_records(normalized), start=1):
        example_id = str(item.get("id", f"example_{index}")).strip() or f"example_{index}"
        focus_tags_raw = item.get("focus_tags", [])
        focus_tags = (
            ", ".join(str(tag).strip() for tag in focus_tags_raw if str(tag).strip())
            if isinstance(focus_tags_raw, list)
            else ""
        )
        input_text = str(item.get("input_text", "")).strip()
        expected_output = item.get("expected_output", {})
        if not input_text:
            continue
        expected_json = json.dumps(expected_output, ensure_ascii=False, indent=2)

        if use_english_labels:
            header = f"Example {index} ({example_id})"
            if focus_tags:
                header = f"{header} | focus: {focus_tags}"
            formatted_examples.append(
                "\n".join(
                    [
                        header,
                        "Input text:",
                        input_text,
                        "Expected JSON:",
                        expected_json,
                    ]
                )
            )
        else:
            header = f"Örnek {index} ({example_id})"
            if focus_tags:
                header = f"{header} | focus: {focus_tags}"
            formatted_examples.append(
                "\n".join(
                    [
                        header,
                        "Girdi metni:",
                        input_text,
                        "Beklenen JSON:",
                        expected_json,
                    ]
                )
            )

    return tuple(formatted_examples)


def _build_examples_block(variant: str, examples: tuple[str, ...]) -> str:
    """Build an optional examples section for few-shot variants."""
    normalized_examples = [example.strip() for example in examples if str(example).strip()]
    if not normalized_examples:
        return ""
    joined_examples = "\n\n".join(normalized_examples)
    if _is_english_prompt_variant(variant):
        return (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "## CURATED FEW-SHOT DEMONSTRATIONS\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{joined_examples}"
        )
    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "## ÖRNEKLER\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{joined_examples}"
    )


def _assemble_prompt(*sections: str) -> str:
    """Join prompt sections while dropping empty blocks."""
    return "\n\n".join(section.strip() for section in sections if str(section).strip())


def is_few_shot_prompt_variant(variant: str) -> bool:
    """Return whether a prompt variant depends on curated few-shot examples."""
    return normalize_prompt_variant(variant) in FEW_SHOT_PROMPT_VARIANTS


def build_langextract_prompt_description(variant: str = DEFAULT_PROMPT_VARIANT) -> str:
    """Build a LangExtract-friendly prompt description with wrapper-format output contract."""
    normalized = normalize_prompt_variant(variant)
    base_rules_block = ENGLISH_ROLE_AND_SCHEMA_BLOCK if _is_english_prompt_variant(normalized) else RULES_BLOCK
    variant_rules_block = _build_variant_rules_block(normalized)
    reasoning_block = _build_reasoning_block(normalized)
    output_contract_block = (
        LANGEXTRACT_OUTPUT_CONTRACT_BLOCK_EN
        if _is_english_prompt_variant(normalized)
        else LANGEXTRACT_OUTPUT_CONTRACT_BLOCK_TR
    )
    return _assemble_prompt(
        base_rules_block,
        variant_rules_block,
        reasoning_block,
        output_contract_block,
    )


def build_system_prompt(variant: str = DEFAULT_PROMPT_VARIANT) -> str:
    """Return the system prompt for a supported prompt variant."""
    normalized = normalize_prompt_variant(variant)
    base_rules_block = ENGLISH_ROLE_AND_SCHEMA_BLOCK if _is_english_prompt_variant(normalized) else RULES_BLOCK
    variant_rules_block = _build_variant_rules_block(normalized)
    reasoning_block = _build_reasoning_block(normalized)
    examples_block = _build_examples_block(normalized, _load_prompt_examples(normalized))
    output_contract_block = ENGLISH_OUTPUT_CONTRACT_BLOCK if _is_english_prompt_variant(normalized) else OUTPUT_CONTRACT_BLOCK
    return _assemble_prompt(
        base_rules_block,
        variant_rules_block,
        reasoning_block,
        examples_block,
        output_contract_block,
    )
