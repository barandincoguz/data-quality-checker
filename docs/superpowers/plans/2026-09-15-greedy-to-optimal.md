# `greedy` → `optimal`: varsayılanı çevirme ve greedy'yi arşivleme planı

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development veya
> superpowers:executing-plans. Adımlar `- [ ]` ile izlenir.

**Amaç:** Eşleştirme motorunun varsayılanını `optimal` yapmak, `greedy`'yi adıyla
seçilebilir bir **tarihî artefakt** olarak arşivlemek, ve bunu bir daha
tartışılmayacak biçimde kayda geçirmek.

**Karar:** `optimal` daha mantıklı. Bu bir görüş değil; projenin kendi kaydı ve
bugün yeniden ölçtüğüm sayılar öyle diyor.

## Kanıt — neden optimal

`benchmark/reference/matching.py`'nin kendi docstring'i `greedy`'yi üç kusurla
tarif ediyor:

1. **Maksimum-kardinalite değil** — doğru eşleşebilecek çiftleri kaçırıyor.
2. **Satır sırasına bağlı** — aynı girdi, farklı sıra, farklı sonuç. Bir metrik
   için bu tek başına diskalifiye sebebi.
3. **Maddesiz altın satırları sessizce düşürüyor** — kanun düzeyi referanslar
   recall paydasından çıkıyor ama aynı şekil tahmin tarafında yanlış pozitif
   diye cezalandırılıyor. Asimetrik.

`optimal` bunların üçünü de kapatıyor: simetrik uygunluk, birleştirme temelli
uyumluluk, ve minimum-maliyet maksimum-akış ataması — yani sonuç bir **graf
değişmezi**, sıradan bağımsız, eşitlik bozmaları deterministik.

**Ölçülmüş maliyet, aynı tahminler üzerinde:**

| kaynak | greedy | optimal |
| --- | --- | --- |
| README, Run 023 çift-motor mühürlü karşılaştırma | F1 **0.8825** | F1 **0.9348** |
| docwise kazananı | Qwen3.5 397B (0.5182) | **DeepSeek v3.2 (0.5931)** — kazanan değişiyor |
| Taban model, 50 validation belgesi (bugün ölçtüm) | F1 **0.707736** | F1 **0.759055** |
| aynı ölçümde `tp+fn` | 372 | 339 |

Motor yalnız seviyeyi değil **sıralamayı** da değiştiriyor. Bir makale için bu,
"hangi model kazandı" cümlesinin motora bağlı olması demek.

**Etkilenen veri payı:** kanonik 3523 referansın **249'u (%7.1)** maddesiz, yani
greedy'nin recall'dan düşürdüğü sınıf. Ayrıca büyük/küçük harf katlanmadığı için
`Mükerrer 257` ile `mükerrer 257` ayrı madde sayılıyor — altın veride 7 gerçek vaka.

**Ve `evaluate.py`'nin kendi CLI yardımı zaten şunu yazıyor:**

> CRITICAL: GT v4 ground truth MUST be evaluated with `--matching-engine optimal`.
> Greedy matching on GT v4 is invalid due to subparagraph splits.

Bu proje GT **v3** kullanıyor (`gt_v3_triangulated_2026-05-15`), yani o uyarı
doğrudan bu veriye değmiyor — ama motorun yazarlarının yönü belli.

## Risk taraması — neden varsayılanı çevirmek güvenli

Depoda `evaluate.py`/`run.py`'yi çağıran **tek** yer var:
`g0_training.canonical_evaluate`, ve o **zaten bayrağı geçiyor**.
`g0_training.py:370` çağıran değil — yalnız `evaluate.py`'nin sha256'sını
sözleşme parmak izine yazıyor.

Yani varsayılanı çevirmek hiçbir dondurulmuş koşuyu sessizce değiştirmez.
Dondurulmuş Run 015-021 ve Run 023'ün greedy yarısı bayrağı **açıkça** geçerek
üretilmişti ve öyle üretilmeye devam edecek.

## Değişecek dört yer

| yer | ne | durum |
| --- | --- | --- |
| `benchmark/reference/evaluate.py:1548` | `evaluate(...)` fonksiyon varsayılanı | çevrilecek |
| `benchmark/reference/evaluate.py:1970` | CLI `--matching-engine` varsayılanı | çevrilecek |
| `benchmark/reference/run.py:196` | CLI varsayılanı | çevrilecek |
| `src/data_quality_checker/q36_p1.py:985` | sabitlenmiş `"greedy"` literali | çevrilecek |

`q36_p1` özel bir vaka ve incelendi: literalin **hiçbir gerekçesi yazılı değil**
(karşılaştırın: `g0_training.MATCHING_ENGINE = "optimal"`'in 12 satırlık gerekçesi
var), Q36-P1 **dondurulmuş tarihî bir koşu değil** yeni bir deney, ve bugüne kadar
**hiç artefakt üretmemiş** (ne JSON sonucu, ne config kaydı). Yani kopyalanarak
gelmiş sessiz bir varsayılan — DQ-Loop'u ısıran kusurun aynısı. Çevirmek bedava.

## Değişmeyecek olan

Dondurulmuş Run 015-021 sayıları ve Run 023'ün greedy yarısı **kayıtta kalır**.
Makale Run 023'ü tam da çift-motor karşılaştırması olduğu için alıntılıyor.
`greedy` **adıyla seçilebilir kalır**; yalnız varsayılan olmaktan çıkar.

---

### Görev 1: Önce koruma — varsayılanın greedy olmasını yakalayan test

**Dosyalar:** Oluştur: `tests/test_matching_engine_default.py`

- [ ] **Adım 1: Başarısız testi yaz**

```python
"""Varsayılan motor `optimal` olmak zorunda.

Bu dalın tarihi: DQ-Loop değerlendiriciyi bayraksız çağırdığı için her döngü
metriği, projenin 2026-08-03'te çoktan geçtiği bir eşleştiriciyle üretilmişti.
Çağıran düzeltildi (`dbd4ccf`) ama varsayılan yerinde kaldı, yani bir sonraki
çağıran aynı tuzağa düşebilir. Bu test varsayılanın kendisini çiviliyor.
"""

from benchmark.reference import evaluate as evaluate_module
from benchmark.reference import run as run_module


def test_library_default_is_optimal() -> None:
    import inspect

    signature = inspect.signature(evaluate_module.evaluate)
    assert signature.parameters["matching_engine"].default == "optimal"


def test_both_cli_defaults_are_optimal() -> None:
    for module in (evaluate_module, run_module):
        source = inspect.getsource(module.parse_args)
        index = source.index("--matching-engine")
        window = source[index : index + 400]
        assert "MATCHING_ENGINE_OPTIMAL" in window, module.__name__


def test_greedy_is_still_reachable_by_name() -> None:
    """Arşivlenmek silinmek değil: dondurulmuş Run 015-021 ve Run 023'ün
    greedy yarısı yeniden üretilebilir kalmalı."""
    assert "greedy" in evaluate_module.SUPPORTED_MATCHING_ENGINES
    assert evaluate_module.normalize_matching_engine("greedy") == "greedy"
```

- [ ] **Adım 2: Koş, üçünün de düştüğünü gör** (ilk ikisi; üçüncüsü bugün de geçer)

`.venv/bin/python -m pytest tests/test_matching_engine_default.py -v`

- [ ] **Adım 3: Commit** — `test: pin the matcher default to optimal (failing)`

### Görev 2: Üç varsayılanı çevir

**Dosyalar:** `benchmark/reference/evaluate.py:1548,1970`, `benchmark/reference/run.py:196`

- [ ] **Adım 1** Üç yerde `MATCHING_ENGINE_GREEDY` → `MATCHING_ENGINE_OPTIMAL`.
- [ ] **Adım 2** Her iki CLI'nın `help` metnini güncelle: `optimal` artık
      varsayılan, `greedy` dondurulmuş Run 015-021 eşliğini yeniden üretmek için
      **adıyla** seçilir.
- [ ] **Adım 3** Mutasyon kanıtı: birini greedy'ye geri al, Görev 1'in testi
      düşmeli. Geri al, geçmeli. **Düşmezse BLOCKED bildir.**
- [ ] **Adım 4** `make test` — tüm paket. Özellikle
      `tests/test_g0_training.py:305` geçmeye devam etmeli.
- [ ] **Adım 5: Commit** — `fix(eval): make optimal the default matcher`

### Görev 3: `q36_p1`'i çevir ve gerekçesini yaz

**Dosyalar:** `src/data_quality_checker/q36_p1.py:985`

- [ ] **Adım 1** `"greedy"` → `"optimal"`, ve üstüne gerekçeyi yaz: bu literalin
      hiçbir gerekçesi yoktu, Q36-P1 dondurulmuş bir koşu değil, ve bugüne dek
      hiç artefakt üretmedi — yani hiçbir sayı bozulmuyor.
- [ ] **Adım 2** Bu sözleşmenin parmak izini üreten testler varsa beklenen
      değerleri **hata çıktısından okuyarak** güncelle, hesaplayarak değil.
- [ ] **Adım 3** `make test`, `make lint`.
- [ ] **Adım 4: Commit** — `fix(q36): score the controlled track with optimal too`

### Görev 4: Greedy'yi tarihî artefakt olarak arşivle ve kesin kayda geç

**Dosyalar:** Oluştur `docs/analysis/2026-09-15-greedy-arsiv-karari.md`;
`benchmark/reference/README.md`'ye işaret ekle.

- [ ] **Adım 1** Arşiv belgesini yaz. İçermesi gerekenler:
      greedy'nin ne olduğu (staged first-fit, Run 015-021 eşliği); üç kusuru;
      ölçülmüş maliyeti (0.8825/0.9348 ve sıralamanın değişmesi; bugünkü
      0.707736/0.759055; tp+fn 372/339); **neden silinmediği** (dondurulmuş
      koşular ve Run 023'ün makaleye bakan yarısı yeniden üretilebilir kalmalı);
      ve kesin kural.
- [ ] **Adım 2** Kesin kuralı hem arşiv belgesine hem README'ye yaz:

      > `greedy` ile `optimal` sayıları **asla** yan yana konmaz, farkları
      > "iyileşme" diye sunulmaz, ve her iddia hangi motorla ölçüldüğünü söyler.
      > `greedy` yalnız dondurulmuş bir koşuyu yeniden üretmek için, adıyla
      > seçilir.

- [ ] **Adım 3** `HANDOFF.md`'nin değişmez kurallarına aynı kuralı ekle.
- [ ] **Adım 4: Commit** — `docs: archive greedy as a named historical engine`
