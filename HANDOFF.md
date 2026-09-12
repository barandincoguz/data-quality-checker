# Devir — DQ-Loop tur 1

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
7. **Her ölçüm hangi motorla alındığını söyler.** `canonical_evaluate` artık
   `--matching-engine optimal` geçiyor ve bunu `metric_view`'a damgalıyor.
8. Her koruma **mutasyonla** doğrulanır, okuyarak değil.
9. Gemini 3 ve altı desteklenmiyor.

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
