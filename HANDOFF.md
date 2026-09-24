# Devir — DQ-Loop tur 1

> **25 Eylül 2026 — DÖNGÜ GT750 İLE YENİDEN KURULDU. Önce bunu oku.** Aşağısı
> eski düzenin (GT v3, 300/50/150, 12×100) 13 Eylül kaydıdır; eski 1. tur (`M01`)
> silinmedi, dokunulmadı, uzman kararı 0/60 idi.
>
> - Plan: `/Users/student2/ner-project/docs/superpowers/plans/2026-09-25-dq-loop-gt750-block-b.md`
>   (kullanıcı kararları B-1…B-4). Dal: `feat/gt750-loop`.
> - Ayar: `--config configs/presets/gt750_loop.json` — ayrı kökler
>   (`data/sensitive/data_quality_checker_gt750`, `artifacts/data_quality_checker_gt750`).
> - Altın veri GT750 (depo içi kopya `data/ground_truth/gt_v6_gt750_2026-09-24/`), bölme
>   500/50/200 (kalıp ikizleri bölünmez), havuz 1.191, batch'ler 4×250 + 191 (`R01`…`R05`).
> - Komutlar: `loop-train --round N [--resume]`, `loop-predict --round N --hmac-key-file …`,
>   `loop-mark --round N --stage judged|adjudicated`, `loop-open-test --confirm-open-test`
>   (yalnız son turdan sonra). Hakem/uzman/sürüm eski komutlarla (`pilot-judges`, `judge-lock`,
>   `serve`, `release`) `--batch-id R0k --generation M00(k-1)`.
> - Başlama kapısı doğrulamada (M000 > eğitilmemiş temel model); **test en sona kadar kapalı**.
> - Puanlamada `213/413` tahminden de düşer (loop ayarında `canonical_evaluate`).
> - M000 eğitimi `tmux` oturumu `gt750-m0`'da; log `artifacts/data_quality_checker_gt750/loop/m0_train.log`.
>   Çökerse aynı komut `--resume` ile.


Son güncelleme: 2026-09-13. Bu belge yeni bir oturumun sıfırdan başlamadan
devam edebilmesi içindir. Hiçbir sayı tahminle yazılmadı; hepsi canlı
artefaktlardan okundu veya kod çalıştırılarak ölçüldü.

Dal: `feat/source-text-out-of-decisions`. Paket: 666 geçiyor, 1 atlanıyor.

---

## 1. Şu an nerede duruyoruz

Tur 1 **başa sarıldı** ve yeniden kuruldu. Bekleyen tek iş insan işi.

| Adım | Durum |
| --- | --- |
| Sıfır modeli seçimi | ✅ M₀′ v11 @ update 425 |
| M01 tahmini | ✅ 100/100 başarılı, karantina yok |
| Yönlendirme | ✅ GREEN 39 / YELLOW 17 / RED 44 / QUARANTINE 0 |
| Hakem pilotu | ✅ 60/60 geçerli, 0 hata |
| **Uzman adjudikasyonu** | ⬜ **0 / 60 pilot belgesi** (100 review beklemede) |
| `judge-lock` | ⬜ 60'ın hepsi finalized olmadan açılmıyor |
| Release → M₁ verisi | ⬜ |

Arayüz: `http://127.0.0.1:5055/login`, anahtar scratchpad'deki `token.txt`.
Yeniden başlatmak için `serve_env.sh`'ı `source` et.

**Hakem tarafında teknik engel kalmadı.** Kalan yol: uzman 60 kararı verir →
`judge-lock` → release → M₁ (400 belgeyle).

---

## 2. Turun sıfır modeli

| | |
| --- | --- |
| Koşu | `dqcheck_loop_9bb9e6994494`, aday `loop-lr2.5e-5-full-epoch` |
| Seçilen checkpoint | update **425** |
| Validation | core F1 **0.910798**, P 0.970000, R 0.858407 |
| | docwise 0.42, loss 0.143919, parse 50/50, runaway 0 |
| Adapter sha | `500aff4486edff62e18e951a33a2e286ddde7b17e3ff5c19de6f6fc7b8d4c0eb` |
| Ölçüm motoru | **optimal** |
| Eğitim görünümü | v11 — 2194 satır, 549 optimizer adımı |

Seçim göz kararı değil, projenin kendi kuralıyla: `core_f1` →
`docwise_accuracy` → `recall` → en düşük `validation_loss`, ve
`parse_count >= 49`. Kural işe yaradı: update 200 (F1 0.8713) 48 belge
ayrıştırabildiği için elendi.

### Taban çizgisi — fine-tune edilmemiş temel model (2026-09-15)

Aynı 50 belgelik validation, aynı `optimal` motoru, aynı istek (update 425'in
isteğinden yalnız adaptör çıkarıldı). `assert_test_improves_base` kapısının
bugüne kadar karşılaştıracak ölçülmüş bir sayısı yoktu; artık var.

| | Taban | M000 | fark |
| --- | --- | --- | --- |
| core F1 | 0.759055 | **0.910798** | +0.151743 (+%20.0) |
| precision | 0.814189 | **0.970000** | +0.155811 (+%19.1) |
| recall | 0.710914 | **0.858407** | +0.147493 (+%20.7) |

Fine-tune **hem precision hem recall** kazandırıyor, ikisi de ~%20 bağıl.
Taban modelin tp/fp/fn'i 241/55/98.

Maliyet (50 belge): taban medyanda daha hızlı (4.0 s vs 6.2 s) ama kuyruğu
belirgin kötü (p95 **28.6 s** vs 16.4 s) ve %31 fazla token üretiyor
(16.000 vs 12.207, p95 893 vs 534). Yani fine-tune modeli kısaltıp
öngörülebilir kılıyor. Tepe bellek 9.81 vs 10.26 GB.

**Taban model JSON sözleşmesine uyuyor:** 50/50 ayrıştırıldı, 0 runaway, 0 boş.
Yani fine-tune'un kazandırdığı şey format disiplini değil, doğruluk.

Kanıt: `evidence/2026-09-15_base_model_baseline/`

### Ölçümün doğrulanması ve motor kararı (2026-09-15)

Taban çizgisi üç ayrı sınavla doğrulandı, tekrar iddia ederek değil:

1. **Adaptör yüklenmedi.** İstekte `adapter_path` ve `adapter_sha256` yok,
   worker `passed`, `model_path` pinli taban snapshot, `data_sha256` M000'inkiyle
   aynı.
2. **Aynı kod yolu.** M000'in *kendi* kayıtlı tahminleri aynı `canonical_evaluate`
   çağrısından geçirildiğinde **0.910798 / 0.970000 / 0.858407** çıktı — mühürlü
   kayıtla bit düzeyinde aynı.
3. **Motor bayrağı fiilen çalışıyor.** *Aynı* taban tahminleri: greedy F1
   **0.707736**, optimal **0.759055**. Rapordaki `matching_engine` alanı bizim
   damgamız değil, değerlendiricinin kendi kaydı (`evaluate.py:1626`), ve
   `:1585`'te gerçek bir davranış dalı var.

**Ve öğrenilen:** iki motorun **paydası farklı**. `tp+fn` greedy'de 372,
optimal'de 339 — iki modelde de aynı, yani motorun özelliği. Sebebi kodda yazılı:
optimal kendi ön-işleme zincirine sahip, çünkü greedy'nin "hangi kopya hayatta
kalır" seçimi **satır sırasına bağlı** ve bastırması yöntemler arası asimetrik.

Sonuç: **taban 0.759055 ile M000 0.910798 aynı 339 paydasında ve doğrudan
karşılaştırılabilir. Hiçbiri greedy dönemi bir sayıyla karşılaştırılamaz.**

### Greedy → optimal: karar ve plan

`optimal` daha doğru motordur; bu bir tercih değil. `greedy` maksimum-kardinalite
değil, **satır sırasına bağlı**, ve maddesiz altın satırları recall'dan düşürürken
aynı şekli tahmin tarafında yanlış pozitif sayıyor (kanonik 3523 referansın
**249'u, %7.1'i** maddesiz). Aynı tahminlerde README'nin kaydı F1 0.8825 / 0.9348
ve **kazanan model bile değişiyor**.

Varsayılan hâlâ üç yerde `greedy` (`evaluate.py:1548`, `evaluate.py:1970`,
`run.py:196`) ve `q36_p1.py:985` onu gerekçesiz sabitliyor. Depoda varsayılana
bel bağlayan **hiçbir çağıran yok** — tek çağıran `canonical_evaluate` ve o
bayrağı geçiyor — yani çevirmek hiçbir dondurulmuş koşuyu bozmaz.

Plan: `docs/superpowers/plans/2026-09-15-greedy-to-optimal.md`

`greedy` **silinmiyor, arşivleniyor**: dondurulmuş Run 015-021 ve Run 023'ün
makaleye bakan greedy yarısı yeniden üretilebilir kalmalı. Adıyla seçilebilir
kalır, yalnız varsayılan olmaktan çıkar.

---

**Neden v10 değil v11:** eski model tam da bu dalda kaldırdığımız şeyle —
girdisinde `source_text` ile — eğitilmişti. Seri, kullandığı tarifle kurulmalı.

---

## 3. Hâlâ geçerli ve yeniden üretilmemeli

| Şey | Değer |
| --- | --- |
| Bölme | 300/50/150, seed `20260902`, fingerprint `7967d589…` |
| Batch manifesti | 12×100, seed `20260902`, havuz 1411 |
| Base model | `mlx-community/Qwen3.5-9B-MLX-4bit` rev `938d8919941c6e7efd3c7150eff7fe9d12afa631` |
| Hakem modeli | `gemma-4-31b-it-optiq-4bit` rev `23616162…`, yerel, sıcaklık 0 |
| Hakem enstrümanı | prompt `97544894…25eb` / `decision_fields_only_v1` |
| Doğrulama sözleşmesi | `per_span_evidence_v2` |

**150 belgelik test bölmesi hiç açılmadı** ve 13 ölçüm noktasının sonuna kadar
açılmayacak.

---

## 4. Değişmez kurallar

1. **`source_text` hiçbir model yüzeyine girmez.** Ne hakem payload'ına, ne
   fine-tune girdisine, ne karşılaştırmaya, ne kova belirlemeye. Yalnız
   veritabanında, uzman ekranında, ve derleme anında gerçek pasajı bulan bir
   **adres** olarak yaşar. (Bursiyerler metnin farklı yerlerinden kırpıyor.)
2. **Test bölümü hiçbir karara girmez.** Checkpoint seçimi yalnız validation.
3. `32/A` = ek madde A (asla bölünmez); `32/a` = bent a (bölünebilir).
   Büyük/küçük harf belirleyicidir.
4. **Uzmanı hiçbir şey bloke etmez.** İçerik yargısıyla red yok.
5. **`pip install` yok, venv oluşturma yok.** `.venv` paylaşımlı lab ortamına
   sembolik bağ. Python: `.venv/bin/python`.
6. Her doğruluk sayısı maliyetiyle yayımlanır: gecikme p50/p95, token, bellek.
7. **`greedy` ile `optimal` sayıları asla yan yana konmaz**, farkları
   "iyileşme" diye sunulmaz. `greedy` tarihî bir artefakttır ve yalnız
   dondurulmuş bir koşuyu yeniden üretmek için, **adıyla** seçilir.
8. **Her ölçüm hangi motorla alındığını söyler.** `canonical_evaluate` artık
   `--matching-engine optimal` geçiyor ve bunu `metric_view`'a damgalıyor.
9. Her koruma **mutasyonla** doğrulanır, okuyarak değil.
10. Gemini 3 ve altı desteklenmiyor.

---

## 5. Açık başlıklar

**Çözümleme kaçağı.** Model bazen JSON dizisini kapatmayıp sınıra kadar
üretiyor. İki şekli var ve ikisi farklı ele alınıyor:

- *Döngü* (aynı referanslar dönüp duruyor) — kurtarılıyor. Kuyruk kanıtlanmış
  tekrarsa dizi kapatılıp benzersizler alınıyor. d000031 böyle kurtarıldı:
  263 nesne → 10 benzersiz referans.
- *Sayma* (bent 201, 202, 203…) — **reddediliyor ve reddedilmeli.** Her referans
  farklı olduğu için kapatmak yüzlerce uydurma referans enjekte eder. Arşivdeki
  d000059 üzerinde mutasyonla kanıtlandı: koruma kaldırılınca 424 uydurma
  referans "kurtarılıyor".

Kaçak **belgenin özelliği değil**: eski turda d000059'a, yeni turda d000031'e
düştü. Şema-zorlamalı çözümleme (`outlines`/`llguidance` kurulu) kalıcı çözüm
olurdu ama örnekleme dağılımını değiştirir; **döngü bitmeden yapılmamalı.**

**Hakemin sınavı ve karar desteği aynı artefakt.** `judge_expert_metrics`
hakemi uzmanın kararıyla notlandırıyor, ve hakemin görüşü uzmana karar desteği
olarak gösterilebiliyor. Şu an çelişki yok — görüş yalnız kilit varken
gösteriliyor, kilit de sınav bittikten sonra kuruluyor — ve bu değişmez
`tests/test_judges.py`'de çivilendi. Ama biri "pilot verdictini kilitten önce
de gösterelim" derse sınav kirlenir.

**`judge_override` ve `source_text`.** Körleşmiş hakem `source_text`
üretemiyor. `reference_locator.py` bilinen bir referansın span'ini belgede
buluyor (231 referansta 220 otomatik, 11 uzmana). Reddedilenler sinyal:
çoğu modelin yanlış pozitifi.

**Dokümantasyon denetimi.** `docs/analysis/2026-09-12-…md` ölçüm motoru,
Türkçe normalleştirme ve kova mantığında 9 bulgu içeriyor; ilki düzeltildi,
kalanı açık.

---

## 6. Kanıt arşivi

`artifacts/data_quality_checker/evidence/`

- `2026-09-12_round1_rewind/` — geri sarma: öncesi/sonrası DB kopyaları, geri
  çekme kaydı, eski ve yeni pilot özetleri, v10 ve v11 seçim kayıtları
- `2026-09-07_source_text_off_model_surfaces/` — körleme, eğitim görünümü v11
  ölçümleri, M₀′ sonucu, verdict kayması
- `2026-09-07_m01_m000_runaway/` — kaçak probları, G0 dönemi verdictleri

İlerleme defteri: `.superpowers/sdd/progress.md` (her görevin kanıtı).

---

## 7. Sıradaki adım

1. Uzman 60 pilot belgesini karara bağlar (release için ayrıca 61 RED/YELLOW).
2. `dqcheck judge-lock --batch-id M01 --model gemma-4-31b-it-optiq-4bit --reason "…"`
3. `dqcheck release --batch-id M01`
4. M₁ eğitimi: 400 belge, v11 tarifi, program satır sayısından türetilir
   (506'dan taşınmaz).
