## 0. DURUM — 2026-09-12, TUR 1 BAŞA SARILIYOR (otonom)

Proje sahibi turu başa sarmaya karar verdi: "en doğrusu ve bilimseli neyse onu
yapalım". Bu belge o işin planı ve ilerlemesi. Oturum kesilirse buradan devam.

### Neden başa sarılıyor

Ölçüm motoru yanlıştı. `canonical_evaluate` değerlendiriciyi motor belirtmeden
çağırıyordu, o yüzden projenin 2026-08-03'te terk ettiği `greedy` motoru
çalışıyordu. Düzeltildi (commit `dbd4ccf`): motor artık açıkça `optimal`, ve
hangi motorla ölçüldüğü `metric_view`'a damgalanıyor.

43 validation noktası yeniden puanlandı (çıkarım yok, sadece puanlama):
**hepsi yükseldi**, ortalama +0.0402, hiçbiri düşmedi. Ve iki modelde de
**seçilen checkpoint değişti**:

| | eski motorla | doğru motorla |
| --- | --- | --- |
| M₀ (v10) | upd 325, F1 0.8539 | upd 450, F1 0.8899 |
| M₀′ (v11) | upd 375, F1 0.8646 | **upd 425, F1 0.9108** |

Önceki rapordaki "v11 recall kazandırdı, precision'dan verdi" yorumu da yanlıştı:
doğru motorla v11 **iki eksende de önde** (P 0.970 vs 0.938, R 0.858 vs 0.847).

### Turun sıfır modeli: M₀′ @ update 425

v10 değil v11 seçildi, çünkü v10 modeli tam da kaldırdığımız şeyle
(girdisinde `source_text`) eğitilmişti. Seri, kullandığımız tarifle kurulmalı.

Seçim göz kararı değil, projenin kendi kuralıyla yapıldı — `core_f1` →
`docwise_accuracy` → `recall` → en düşük `validation_loss`, ve `parse_count>=49`.
Kural işe yaradı: update 200 (F1 0.8713) parse_count 48 olduğu için elendi.

  upd 425 | core F1 0.910798 | P 0.970000 | R 0.858407
          | docwise 0.42 | loss 0.143919 | parse 50/50 | runaway 0

### Config, gerçek veriye karşı doğrulandı (sahibin şartı)

| | ölçülen | sınır | |
| --- | --- | --- | --- |
| girdi (en uzun M01 belgesi) | 7660 token | 10240 | ✓ |
| çıktı (meşru en uzun cevap) | 903 token, 22 referans | 8192 | ✓ 9.1 kat pay |

Sınır dar değil. Tek başarısız belge d000059, uzunluktan değil **döngüye
girmekten** patlıyor (`bent` alanını 201, 202, 203 diye sayıyor). Sınırı
büyütmek çözmez — kayıt ölçmüş: 8192'de 260 kopya, 16384'te 524. İki bilinen
çözüm de daha önce bilinçle ertelenmişti (şema-zorlamalı çözümleme örnekleme
dağılımını değiştirir; "artan sayı" sezgiseli meşru listeden ayırt edilemez).
Deney ortasında örnekleme değiştirmek turu kirletir. Belge karantinada kalır.

### Adımlar ve durum

- [x] 1. Ölçüm motoru düzeltmesi (`dbd4ccf`)
- [x] 2. 43 noktanın yeniden ölçümü
- [x] 3. Config doğrulaması
- [x] 4. Sıfır modelinin seçimi: M₀′ @ 425
- [x] 5. Seçim kaydı mühürlendi: `round_000_checkpoint_v11.json` (v10'unki duruyor)
- [x] 6. M000 nesil kaydı `500aff44…` (upd 425) adaptörüne çevrildi; eskisi arşivde
- [x] 7. M01 geri çekildi: 100 tahmin, 10 uzman kararı, 60 hakem verdicti, 3 artefakt
        (kayıt: `evidence/2026-09-12_round1_rewind/withdrawal_record.txt`)
- [x] 8. M01 yeniden tahmin edildi: **100/100 başarılı, karantina yok**
        (ilk geçişte d000031 döngüye girmişti; kurtarma kuralı genişletildi,
         commit `dbddc5d`, belge 10 referansla kurtarıldı)
- [x] 9. Yeni kovalar: GREEN 39, YELLOW 17, RED 44, QUARANTINE 0
        (eski: GREEN 36, YELLOW 12, RED 51, QUARANTINE 1)
- [x] 10. Hakem pilotu: **60/60 geçerli, 0 hata** (ilk kez tam temiz), 1824 s
- [x] 11. Arayüz açık: http://127.0.0.1:5055/login
        anahtar `scratchpad/token.txt` içinde

### Tur 1 geri sarma TAMAMLANDI

| | eski tur | yeni tur |
| --- | --- | --- |
| sıfır modeli | M₀ v10 @ 325, F1 0.854 | **M₀′ v11 @ 425, F1 0.911** |
| ölçüm motoru | greedy (terk edilmiş) | **optimal** |
| tahmin | 99 başarılı, 1 karantina | **100 başarılı, karantina yok** |
| kovalar | 36 / 12 / 51 / 1 | **39 / 17 / 44 / 0** |
| hakem pilotu | 56 geçerli, 4 hata | **60 geçerli, 0 hata** |

Pilot verdictleri (körlenmiş):

| kova | bursiyer | model | TIE | NEITHER |
| --- | --- | --- | --- | --- |
| GREEN | 2 | 5 | **14** | 0 |
| YELLOW | 9 | 8 | 0 | 0 |
| RED | 8 | 12 | 0 | 2 |

Karantinaya alınan kanıt cümlesi: 1 (verdicti düşürmedi).

**Bekleyen tek iş insan işi:** uzmanın 60 pilot belgesini karara bağlaması.
Sonra `judge-lock` → release → M₁. `judge-lock` 60'ın hepsi finalized olmadan
açılmıyor; hakem tarafında teknik engel kalmadı.

### Sıfırlanacak insan emeği — bilinçli karar

10 review karara bağlanmıştı (7 accept_human, 3 accept_model). Hepsi
sıfırlanıyor. accept_model olanlar artık var olmayacak bir çıktıyı işaret
ediyor. accept_human olanlar da uzmanın önünde bir model çıktısı varken
verildi, yani karar modelden bağımsız değil. Karışık kökenli bir küme
savunulamaz; 10 belge bedeli, temiz köken için ödenir.

---

## 0. DURUM — 2026-09-07 20:35

**Sızıntı kapatıldı ve gerçek veride doğrulandı. Dal:
`feat/source-text-out-of-decisions`, 12 commit, paket 647 geçiyor / 1 atlanıyor /
0 düşüyor.**

Plan: `docs/superpowers/plans/2026-09-07-source-text-off-model-surfaces.md`
İlerleme defteri: `.superpowers/sdd/progress.md` (her görevin kanıtı orada)

### Bitenler

| Görev | Durum |
| --- | --- |
| 1 Değişmez korumalar | ✅ uygulandı + gözden geçirildi |
| 2 Hakem adayları körlendi | ✅ uygulandı + gözden geçirildi |
| 3 Enstrüman parmak izi + resume | ✅ uygulandı + gözden geçirildi |
| 4 Boyutlandırma servis edilen hedefe | ✅ uygulandı + gözden geçirildi |
| 5 Kurtarma gerçek pencereye | ✅ uygulandı + gözden geçirildi |
| 6 Dense replay gerçek pasaja | ✅ uygulandı, ⚠️ **gözden geçirme yarıda kesildi** |
| 7 Tarif adı v11 | ✅ uygulandı, ⚠️ **gözden geçirilmedi** |
| 8 Uçtan uca doğrulama | 🔶 ölçümler yapıldı, `verification.json` yazılmadı |
| 9 M₀′ eğitimi | ⬜ **Görev 6 gözden geçirilmeden BAŞLAMASIN** |

Görev 6 ve 7'nin gözden geçirmeleri oturum kota limitine takıldı (sıfırlanma
22:30). İkisi de yeniden devredilmeli.

### Hakem pilotu yeniden koştu — asıl sonuç

```
stale_verdicts_rejudged : 57      <- yeniden koşunun gerçekten koştuğunun makbuzu
valid / error           : 56 / 4  (geçersiz koşu: 57 / 3)
gecikme                 : 1835.6 s (geçersiz koşu: 2050.6 s, -%10.5)
enstrüman               : prompt 97544894…25eb / decision_fields_only_v1
```

**Verdict kayması (bursiyer / model / TIE):**

| Kova | ÖNCE | SONRA |
| --- | --- | --- |
| GREEN | 17 / 1 / 4 | **4 / 2 / 17** (kesin karar 18 → 6) |
| YELLOW | 9 / 3 / 0 | **5 / 7 / 0** |
| RED | 12 / 9 / 0 | **9 / 12 / 0** |
| TOPLAM | 38 / 13 | **18 / 21** |

GREEN'de beklenen TIE ağırlığı geldi. **Asıl bulgu:** sızıntı GREEN'e özgü
değilmiş — her kovada bursiyeri şişiriyormuş. Körlenmiş hakem YELLOW ve RED'i
modele veriyor ve toplam tersine dönüyor. Geçersiz pilotun M000 hakkında verdiği
tablo derecede değil, **yönde** yanlışmış.

Hata 3 → 4: hakem artık kendi kanıt dizisi için bursiyerin birebir span'lerini
kopyalayamıyor, span'i belgede kendisi bulmak zorunda. Körlemenin ölçülen bedeli.

### Eğitim görünümü v11, mühürlü 300 belgede ölçüldü

| | v10 | v11 |
| --- | --- | --- |
| satır | 2022 | **2194** |
| bütün belge | 28 | **35** |
| dense replay satırı | 333 | **506** |
| dense replay referansı | 2075 | 2041 |
| **kullanıcı turu belgede DEĞİL** | **324** | **0** |
| **kapsanmayan referans** | — | **0** |

2075 → 2041 düşüşü **kayıp değil**: tam olarak Görev 4'ün bütün belgeye taşıdığı
7 belgenin (9, 109, 166, 213, 363, 443, 477) replay örnekleri. Bütün belgeler hiç
replay edilmiyor. Gerçek kayıp göstergesi kapsanmayan referans ve o sıfır.

**Yeni program geometrisi: 2194 satır → 549 optimizer adımı** (mühürlü koşu 506).
M₀′ ve M₁ bundan türetilecek, 506'dan taşınmayacak.

### Koşu kimlikleri — mühürlü koşu güvende

v10 tarifiyle payload'ı yeniden üretmek `dqcheck_loop_fd1abc3af6b7`'yi **birebir**
veriyor (yöntem doğrulandı). v11 altında tur 0 `dqcheck_loop_9bb9e6994494`,
tur 1 `dqcheck_loop_8ab546b4ce49`. Çakışma yok; M₀′ mühürlü dizini ezemez.
Bootstrap kimliği `dqcheck_g0_qwen3_5_9b_e613d3647c11` olarak yeniden çivilendi.

### Kanıt arşivi

`artifacts/data_quality_checker/evidence/2026-09-07_source_text_off_model_surfaces/`
- `dqcheck_before_blinded_rerun.sqlite3` — koşu öncesi DB
- `training_view_v11_rebuild.txt` — görünüm ölçümleri
- `judge_pilot_summary_blinded.json`, `verdict_shift.txt`

### 2026-09-07 23:02 — Görev 6, 7 tamam; M₀′ eğitimi başladı

Görev 6 ve 7 gözden geçirildi ve kapandı. Görev 6'nın iki Important'ı (bayat
docstring sayıları, iki kapsanmamış doğruluk özelliği) `cb195ab` ile düzeltildi;
`src/` içinde tek çalıştırılabilir satır değişmedi. Paket: **649 geçiyor / 0
düşüyor**.

**M₀′ eğitimi 23:01'de başladı** (arka planda, saatler sürer):
`dqcheck_loop_9bb9e6994494`, generation M000, bölme 300/50/150 (mühürlü,
fp `7967d589…`), aday `loop-lr2.5e-5-full-epoch`, **target_updates 549**.
Mühürlü `dqcheck_loop_fd1abc3af6b7` el değmiyor.

Boru hattının kurduğu eğitim görünümü, benim ölçüp kapılarını kanıtladığımla
**bit düzeyinde aynı**: sha256 `d7fb3ea…`, 2194 satır, 35 bütün belge.

Bittiğinde: checkpoint **yalnız validation** ile seçilecek; M₀′ core F1/P/R,
M₀'ın 0.853933 / 0.894118 / 0.817204'ünün yanına gecikme p50/p95, token ve
bellekle birlikte yazılacak. **Riskteki sayı recall.** 150'lik test bölmesi
kapalı kalır. Çökerse aynı çağrı `resume=True` ile tekrarlanır.

Log: `scratchpad/m0prime.log`, sonuç: `scratchpad/m0prime_result.json`

### 2026-09-08 08:42 — M₀′ bitti, plan tamamlandı

**9 görevin 9'u kapandı.** M₀′ 9.68 saatte tamamlandı
(`dqcheck_loop_9bb9e6994494`, 549 adım). Checkpoint **yalnız validation** ile
seçildi; 150 belgelik test bölmesi açılmadı.

| | M₀ (v10, upd 325) | M₀′ (v11, upd 375) | fark |
| --- | --- | --- | --- |
| core F1 | 0.853933 | **0.864569** | +0.0106 |
| precision | 0.894118 | 0.880223 | −0.0139 |
| **recall** | 0.817204 | **0.849462** | **+0.0323** |

**Riskteki sayı olan recall yükseldi.** Update 50'deki düşük recall geçiciymiş.

Tüm eğri (20 ortak nokta): M₀′ F1'de 12/20, recall'da 13/20 önde; ortalama F1
0.8203 vs 0.8080, ortalama recall 0.7629 vs 0.7382, precision fiilen berabere.

Maliyet (50 doğrulama belgesi): gecikme p50 6.80→6.95 s, **p95 15.72→13.38 s**,
duvar saati 603.9→530.6 s, tepe bellek 10.26 GB (aynı), çıktı token 12329→13075.
Daha çok üretip daha hızlı bitiriyor.

**Saklanmayan uyarılar:** koşu boyunca runaway M₀ 0 / M₀′ 2 (ikisi de update
200'de; **seçilen** 375 checkpoint'inde runaway 0, parse 50/50). Fallback
kurtarma 14 → 26. Ve M₀′ − M₀ farkı saf tarif ablasyonu değil, "tarif + tarifin
gerektirdiği program uzunluğu" (549 vs 506 adım) — bu karışma yapısal ve önceden
ilan edilmişti. Tek koşu nedensellik kurmaz: v11 daha iyi **kanıtlanmadı**, daha
kötü olmadığı ve riskteki eksende daha iyi olduğu **ölçüldü**.

Kanıt: `evidence/2026-09-07_source_text_off_model_surfaces/m0prime_result.txt`

### Sıradaki adım

1. ~~Görev 6 ve 7'nin gözden geçirmeleri~~ ✅ bitti.
2. ~~Görev 9: M₀′ eğitimi~~ ✅ bitti.
2. Görev 8: `verification.json`'ı kanıt dizinine yaz.
3. Görev 6 temiz gelirse Görev 9: M₀′'ı mühürlü 300 belgede v11 ile eğit.
4. Sonra: uzman adjudikasyonu (60 pilot belgesi) → `judge-lock` → release → M₁.
   Sıra bu; `judge-lock` 60 belgenin hepsi `finalized` olmadan açılmıyor.

---

## 1. Bloke edici bulgu — ÇÖZÜLDÜ

Aşağısı tarihsel kayıt. Sorun tanısı doğruydu; düzeltmesi Görev 2'de
(`7f501b4`) yapıldı ve yukarıdaki verdict kaymasıyla doğrulandı.

## 2. Nerede duruyoruz

DQ-Loop: 12 tur × 100 belge. Eğitim 300 → 1500. Batch *k*, model M(k-1) ile
temizlenir. **Şu an tur 1** (batch `M01`, model `M000`).

| Adım | Durum |
| --- | --- |
| M₀ eğitimi (300 belge) | ✅ bitti |
| Checkpoint seçimi (update 325) | ✅ mühürlendi |
| M000 model kaydı | ✅ `artifacts/data_quality_checker/g0/M000.json` |
| M01 tahmini + yönlendirme | ✅ 36 GREEN / 12 YELLOW / 51 RED / 1 QUARANTINE |
| Hakem pilotu | ✅ körlenmiş hâliyle yeniden koştu (57 verdict yenilendi) |
| `judge-lock` | ⬜ karar verilmedi |
| Uzman adjudikasyonu | ⬜ |
| Serbest bırakma → M₁ verisi | ⬜ |

**Çalışan süreç yok. Aktif izleyici yok.**

---

## 3. Mühürlü ve geçerli olanlar — yeniden üretmeyin

| Şey | Değer |
| --- | --- |
| Eğitim koşusu | `dqcheck_loop_fd1abc3af6b7`, aday `loop-lr2.5e-5-full-epoch` |
| Seçilen checkpoint | update **325** — core F1 **0.853933**, P 0.894118, R 0.817204 |
| Seçim kaydı | `artifacts/.../loop/selection/round_000_checkpoint.json` sha `fad98b99…` |
| Adapter | `…/loop-lr2.5e-5-full-epoch/checkpoints/update_0000325`, sha `0089e99a…` |
| Base model | `mlx-community/Qwen3.5-9B-MLX-4bit` rev `938d8919941c6e7efd3c7150eff7fe9d12afa631` |
| Bölme | 300/50/150, seed `20260902`, fingerprint `7967d589…` |
| Batch manifesti | 12×100, seed `20260902`, havuz 1411 |
| Eğitim görünümü | 2022 satır → 506 optimizer adımı, warmup 25 |
| Hakem modeli | `gemma-4-31b-it-optiq-4bit`, rev `23616162…`, yerel, sıcaklık 0 |

**M₀'ın 0.853933 ölçümü bugünkü tüm değişikliklerden sonra bit düzeyinde
yeniden doğrulandı.** `benchmark/` kodunun `data_quality_checker`'a hiç import
bağımlılığı yok; iki motor gerçekten ayrı.

**150 belgelik test bölümü hiç açılmadı** ve 13 ölçüm noktasının sonuna kadar
açılmayacak.

---

## 4. Değişmez kurallar

1. **`source_text` ölçüt değildir.** Karşılaştırmaya, kova belirlemeye ve model
   eğitimine giremez. Yalnız veritabanında, proje sahibinin bursiyer hata
   analizi için. (Bursiyerler metnin farklı yerlerinden kırpıp yapıştırıyor.)
2. **Test bölümü hiçbir karara girmez.** Checkpoint seçimi yalnız validation.
3. `32/A` = ek madde A (asla bölünmez); `32/a` = bent a (bölünebilir). Büyük/küçük
   harf belirleyicidir.
4. **Uzmanı hiçbir şey bloke etmez.** İçerik yargısıyla red yok; şema hatası ve
   iş akışı kaydı (erteleme gerekçesi) yerinde kalır.
5. **`pip install` yok, venv oluşturma yok.** Paylaşımlı lab venv'i repo
   `.venv` sembolik bağı üzerinden. Python: `.venv/bin/python`.
6. Gemini 3 ve altı desteklenmiyor.
7. Her doğruluk sayısı maliyetiyle yayımlanır: gecikme p50/p95, token, bellek.
8. Her koruma **mutasyonla** doğrulanır, okuyarak değil.
9. `ner-project`'te push hedefi `baran` fork'u (origin 403).

---

## 5. Bugün yapılanlar (tekrar etmeyin)

`data-quality-checker` — dal `feat/source-text-out-of-decisions`, temiz:

| Commit | Ne |
| --- | --- |
| `91f4ae3` | `train_round` preflight düzeltmesi; `canonical:` öneki kaldırıldı (yoksa 50 doğrulama belgesi çözülemezdi) |
| `767f6bd` | `process`/`reroute`/`serve` tur etiketi kabul ediyor |
| `e935a90` | Kaçak üretim: erken durdurma (eşik 12, veriden) + kurtarma dağıtım yoluna taşındı |
| `f2bb4e3` | Çıplak madde, kendi fıkrası tarafından soğuruluyor (YELLOW 17→12, GREEN 30→36) |
| `fb3e9f9` | Uzman yolundaki içerik redleri kaldırıldı; `conflicting_law_identity` düzeltildi |
| `3ea1f5a` | Hakem verdicti artık `judged_prediction` taşıyor; resume koruması |
| `16fc992` | `--generation` döngü komutlarında zorunlu, varsayılansız |
| `110bb7c` | `docs/prompts/repo-cleanup.md` |

`ner-project` — dal `report-24-scope-decision`, temiz:
`4da1c2f` (tur 0 kaydı, C019/C020 iddiaları), `d99e029` (tur 1 bulguları).

Test: **636 geçiyor**, 1 atlanıyor. `make -C Journal audit` geçiyor.

---

## 6. Bilinen açık başlıklar

- **Sayma kaçağı** (`d000059`): `bent` alanını 201, 202, 203 diye sayan kaçak
  hâlâ 8192 sınırına kadar koşuyor (166 s) ve doğru şekilde karantinada kalıyor.
  Sezgisel "artan sayı" dedektörü **bilerek eklenmedi** — madde 1,2,3,4 diye
  giden meşru listeden ayırt edilemez. Güvenli çözüm şema-zorlamalı çözümleme
  (`outlines`/`llguidance` zaten kurulu, `generate_step` `logits_processors`
  alıyor); **döngü bittikten sonra**, çünkü örnekleme dağılımını değiştirir.
- **Yönlendirici değişiklikleri 16 dakikalık GPU geçişi gerektiriyor.** Ayrı bir
  `reroute` aracı 12 tur boyunca kendini amorti eder. Bugün 5 geçiş yapıldı.
  **Kural: bir yönlendirici değişikliğinin kova etkisini önce simülasyonla ölç,
  sonra geçiş koş.**
- `M000.json`, `artifacts/.../g0/` dizininde duruyor (`_base_run_context` o yolu
  sabitliyor). Kozmetik; mühürlü yollar oraya bağlı olduğu için dokunulmadı.
- Hakem sonuçları `(batch_id, doc_id, model)` ile anahtarlı — `generation` yok.
  Anahtar genişletilmedi (döngüde bir batch tek nesille yargılanır) ama
  provenance artık kayıtlı ve resume onu doğruluyor.

---

## 7. Kanıt arşivi

`artifacts/data_quality_checker/evidence/2026-09-07_m01_m000_runaway/`
- `predictions_pass1/` — ilk M000 geçişi (2 karantina)
- `predictions_pass2_threshold3/` — eşik-3 geçişi (d000052 yanlış pozitifi)
- `predictions_pass_subsumption/` — kapsama kuralı geçişi
- `judge_g0_era/` — G0 döneminin 27 hakem verdicti + DB satırları
- `d000021_cap16384.txt`, `d000059_cap16384.txt`, `probe_cap16384.log` — sınır probu
- `dqcheck_before_rerun.sqlite3` — DB yedeği

---

## 8. Sıradaki adım

Planı izle: `docs/superpowers/plans/2026-09-07-source-text-off-model-surfaces.md`

1. Görev 1 — değişmez korumaları yaz, bugünkü koda karşı başarısız olduklarını gör.
2. Görev 2-3 — hakemi körleştir; enstrüman parmak izi + resume koruması.
3. Görev 4-7 — `source_text`'i eğitim girdisinden çıkar, tarif adını v11'e taşı.
4. Görev 8 — uçtan uca doğrula; pilotu yeniden koş (~35 dk, tmux).
5. Görev 9 — M₀′'ı mühürlü 300 belgede eğit, 0.853933'ün yanına yaz.
6. Sonra: uzman adjudikasyonu (60 pilot belgesi) → `judge-lock` → release → M₁.
