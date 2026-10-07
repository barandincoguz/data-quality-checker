# Devir — DQ-Loop

> **3 Ekim 2026, 12:41 — SIRADA UZMAN.**
> - M000 (P4, GT v8) `dqcheck_loop_88cf7872b3ed` mühürlü: @550, F1 0,9469; temel model kapısı 0,7742. Önkayıtlı
>   doğrulama: P4 ≥ P0 (0,9443) → **P4 kaldı**.
> - R01 bitti:
>   - tahmin 149/100/1 (karantina d000112: kaçak çıktı; pencere geri düşmesi denendi, kurtarılamadı,
>     karantinada kalır);
>   - uzmanın 23 kararı taşındı;
>   - hakem 131/131;
>   - sıfırdan kanıt tuttu;
>   - **ekran 5056 açık (tmux `gt750la-serve`), uzman kuyruğu 111**.
> - Gece düzeltilen döngü açıkları:
>   - karantina belgesi uzmana ve hakeme gider (`e40bdc1` + judges);
>   - boş cevap kapısı tek kaçırmayı değil çöküşü durdurur (`d115e17`).
> - Toplu tahmine pencere geri düşmesi eklendi (`7126ba4`): yalnız hata veren belge pencerelerle yeniden denenir;
>   d000112'de ilk pencere de sınıra koştu.
> - İkinci hat: tam belge `dqcheck_loop_1090f276e398` @125 0,9296 vs pencereli 0,9469; fark anlamsız; ana hat
>   pencereli kalır. Durdurulan eski koşu `Fine-Tuning/runs/_archive/` altında.
> - GT v9 (test bölmesi denetimi bitti): 200 belge tarandı, 191 şüpheli, iki bağımsız okuyucu (Claude, kör agy)
>   191/191 tam uyum; 1 belgede (doc 230) 2 tarife satırı (492/13, 492/20) düşürüldü; test satırı 896 → 894; döngü
>   v8'de kalır; v9 testi test açılırken kullanılacak (ayrı ve kayıtlı bir adımla yazılacak).
> - Yabancı dal (`wip/foreign-deep-audit-2026-10-02`) değerlendirildi: Yürürlükteki motorda adayın yanlış numarasını
>   doğru sayan sızıntı bulundu; sızıntıyı düzelten `proposal/engine-valid-numbers` (`0ac39e6`) öneri paketi hazırlandı;
>   R01 sonrasında, M001 öncesinde testler uyarlanarak uygulanacak ve önceki turlar yeniden puanlanacak (P4 M000
>   0,9429 olur; kullanıcı kararıyla P4 kaldı).
> - Açık kararlar (kullanıcı): yok (açık uçlu aralık yazımı olduğu gibi kalır).
> - Bu kutunun altındaki numaralı bölümler ("1. Şu an nerede duruyoruz" … "7. Sıradaki adım": M01 pilotu, port 5055)
>   13 Eylül döneminin kaydıdır; güncel durum ve sıradaki adım yalnız bu kutudadır.
> - 4 Ekim: her tur gürültülü ↔ temiz ablasyonu (`M_k_raw`, ham bursiyer etiketi; karar R05 sonrası testte) — ner
>   `Journal/experiments/dq_loop_gt750/ablation_noisy_vs_clean/` (önkayıt, kontrol listesi). Ham etiket dışa aktarımı salt
>   okunur + yedek: `data/sensitive/raw_label_snapshots/`.
> - 5 Ekim: hedef dergi Expert Systems with Applications (kullanıcı). Taslak ner `Journal/manuscript/latex_eswa/main.tex`
>   (elsarticle, çift kör; kırmızı = yapılmamış); dergi bilgileri `Journal/manuscript/journal_targets/README.md`; iddia
>   kaydında `journal_scope` sütunu. R01 → M001 ve her tur sonucu bu taslağın tablolarına kayıttan girer.
> - Belge işleri `auditing-project-docs` skill'iyle (ner `.claude/skills/`, `~/.claude/skills`'e bağlı): önem 1–3 agy
>   (gemini-3.8-flash-high) yazar, Claude doğrular; 4–5 (LEDGER, önkayıt, karar, makale; haftalık rapor özeti, genel
>   defterler, HANDOFF, canlı belge mantığı) Claude yazar; kapanışta `doc_guard.py` kontrolleri. Canlı belge
>   `ner-project/Journal/experiments/dq_loop_gt750/CANLI_DURUM.md`
>   (`Journal/analysis/scripts/dq_loop_gt750_live_status.py [--check]`).
> - 5 Ekim 11:50: R01 uzman incelemesi başlamadı (WAL 0 bayt, 3 Ekim 03:55'ten beri). Ablasyonun ham etiket kolu yazıldı:
>   dal `feat/raw-label-arm` (`cdce2ff`, çalışma ağacı `/Users/student2/dq-raw-arm`), `loop-train --round k --labels raw`,
>   hiç mühürlenmez; temiz koşu kimlikleri değişmez. Yerleştirme (kullanıcı: "Bursiyer alıntısıyla yerleştir"): önce bulucu,
>   yerleşmeyen referans bursiyerin metinde birebir geçen alıntısıyla, o da yoksa düşer ve sayılır (önkayıt eki). **Dal R01
>   incelemesi bitince birleştirilir** (ekran ana ağaçtan çalışıyor), sonra R01 yayın listesiyle gerçek sınama, sonra
>   `M001` + `M001_raw`.
> - **5 Ekim 12:11 — `M001_raw` ARKADA EĞİTİLİYOR** (kullanıcı: "ham kol arkada eğitik halde kalsın"): koşu
>   `dqcheck_loop_849cc99d1f69`, kod dondurulmuş ağaç `/Users/student2/dq-run-m001-raw` (etiket `run/m001-raw-2026-10-05`,
>   `615946e`), gözetmen `loop/supervise_m1_raw.sh` (tmux `gt750-m1raw-train`, günlük `loop/m1_raw_train.log`); 750 belge,
>   1.280 adım. **Ana hattın GPU işini (M001, ikinci hat, R02 tahmini, hakem) başlatmadan önce `touch
>   artifacts/data_quality_checker_gt750_law_article_p4_gtv8/loop/PAUSE_RAW_ARM`, günlükte `DURDU` bekle; ana hat bitince
>   dosyayı sil** (koşu kaldığı yerden sürer). `STOP_RAW_ARM` gözetmeni bitirir. Yayından sonra `loop-raw-check --round 1`.
>   Nokta seçimi motor önerisinden sonra yeniden puanlanmış eğriden (önkayıt eki 2).
> - **5 Ekim 14:58 — HAZIR, R01 YAYININI BEKLİYOR:** seçim kodu `loop-select-rescored` (dal `feat/raw-label-arm`,
>   `1d44198`) ve motor önerisi (dal `feat/engine-valid-numbers`, `44ebbfd`, çalışma ağacı `/Users/student2/dq-engine-apply`;
>   commit edilmiş kodla yeniden ölçüldü: P4 M000 @550 0,9429, P0 @571 0,9443; dalın 30 testi geçiyor; iki dalın deneme
>   birleştirmesi çakışmasız, süit 1.113). Uzman R01'e başlıyor. **R01'den sonraki sıra:** (1) `loop-errors --round 1` +
>   kayıt betiği (mevcut motorla) → (2) yayın → (3) ekran kapatılır (kullanıcıya sor) → (4) iki dal `feat/gt750-loop`'a
>   birleştirilir → (5) `loop-rescore` M000 + `loop-select-rescored` (beklenen @550 0,9429) ve R01 hata raporu yeni motorla →
>   (6) `loop-raw-check --round 1` → (7) `PAUSE_RAW_ARM`, `DURDU` → `M001` → (8) `M001_raw` bitince `loop-rescore` +
>   `loop-select-rescored`.
> - **5 Ekim 23:31 — TEKRARLI SATIRLAR:** tur belgelerinin eğitim hedefi iki kolda da her referansı bir kez taşır (dq
>   `5605ac1`, `feat/raw-label-arm`; önkayıt eki 3 ner `0908cad`). Bu yüzden `M001_raw` yeniden başlatıldı: **güncel koşu
>   `dqcheck_loop_02d8059bfbad`** (etiket `run/m001-raw-2026-10-05b`, aynı tmux ve gözetmen). Eski koşu
>   `dqcheck_loop_849cc99d1f69` 725. adımda durduruldu, kullanılmaz. Uzman ekranda tekrarlı satırı elle silmek zorunda değil.
> - **6 Ekim 00:15 — EĞİTİM VERİSİ DENETİMİ:** tur belgelerinin referansları pencereye bulucunun bütün yazılı/çıkarsanmış
>   geçişleriyle yerleşir (dq `cb39c18`; kullanıcı "Yalnız tur belgeleri"; önkayıt eki 4 ner `0717c85`; kaynak ner
>   `sources/training_data_audit_2026-10-06.json`). **Güncel `M001_raw`: `dqcheck_loop_3331f64287a1`** (etiket
>   `run/m001-raw-2026-10-06c`, 1.362 adım). `dqcheck_loop_02d8059bfbad` 25'te duraklatıldı, kullanılmaz. Denetimde başka
>   sorun yok (213/413, hedef alanları, sızıntı, satır sınırı, kapsama).
> - **6 Ekim 10:58 — UZMAN EKRANI YENİ ARAYÜZLE YENİDEN BAŞLATILDI** (dq `4f48556`, `feat/review-ui` ana dala ileri alındı; tmux
>   `gt750la-serve` aynı betikle yeniden kuruldu). Vurgulama düzeltildi (model satırları da, her geçiş, "metinde bulunamadı"),
>   iki sütunlu düzen. Kararlar korundu: inceleme satırlarının parmak izi önce/sonra aynı (62 bitmiş). Sahte uyuşmazlık
>   kontrolü: yok (yeniden yönlendirme yapılmadı). Arayüz geliştirmesi için deneme ekranı: `scripts/dev_review_screen.py
>   --root <boş geçici klasör> --port 5057` (uydurma belgeler). **dq push EDİLMEDİ:** uzak depo herkese açık ve dal GT v6–v9'u
>   (test cevap anahtarı dahil) ekliyor; karar kullanıcıda. ner `origin`'e push edildi.
> - **6 Ekim 12:09 — EKRAN İKİNCİ SÜRÜM** (dq `68dbd01`): hazırlama uyarıları düz Türkçe ("Bursiyer listesi notları"), geçiş
>   gezgini, kaydırma işaretleri, "yalnız farklar", seçerek düzeltme (kaydetme açık tıklama), maddesiz/numarasız etiketleri.
>   Parmak izi önce/sonra aynı (62 bitmiş). Arayüz testleri: `tests/test_review_ui_browser.py` (gerçek tarayıcı, uydurma parti
>   `tests/review_fixture.py`).
> - **7 Ekim 13:36 — R01 SONRASI TEK DAL `integrate/after-r01` (`59c3c63`, worktree `/Users/student2/dq-after-r01`):**
>   ham kol/eğitim verisi düzeltmeleri + motor önerisi + **madde aralığı kuralı** (kullanıcı: "473 ila 475" = 473, 474, 475, her
>   madde ayrı satır; motor ve eğitim verisi; M001'den itibaren altın da). Süit 1.172. Yukarıdaki sıradaki "(4) iki dal
>   birleştirilir" adımı artık bu dalın birleştirilmesidir. `M001_raw` 6 Ekim 22:00'de bitti (`dqcheck_loop_3331f64287a1`, ön
>   değer @300 F1 0,9341); yeniden eğitilmez. Ekranda "aralığı aç" düğmesi canlıda (`9cf6079`).
> - **7 Ekim 13:47 — HATA RAPORUNDA TAMAMLAYICILIK** (dq `7096b06`): uzmanın doğrusunu kim buldu (bursiyer/model/hakem/birleşik/
>   yalnız uzman), taşınan 23 karar ayrı (kaynak ve süre ölçüleri). R01 kaydında `complementarity.new_decisions` ve
>   `expert_workload.*_new` kullanılmalı; kayıt betiği sütunları hazır.
> - **7 Ekim 23:55 — R01 BİTTİ VE YAYINLANDI; M001 BELLEK BEKLİYOR.**
>   - Uzman kuyruğu bitti (134 karar). Yayın `release_23061e3c0de900d2`; tur 1 `adjudicated`.
>   - Ekran kapatıldı (kullanıcı; tmux `gt750la-serve` yok). Parmak izi `74676a38…` değişmedi.
>   - `integrate/after-r01` birleştirildi (`b54164d`).
>   - **Yayın sonrası doğru düzeltmeleri** (`322171c`): `releases/R01/truth_corrections.json`. Temiz kolun eğitim verisi ve
>     hata raporu bunu okur; yayın değişmez; komutlar `loop-correct-ranges`, `loop-correct-rows`.
>     - "Kuralla düzelt": aralık kuralından önceki 4 kararda 57 ara madde eklendi.
>     - "Önerileri uygula": bulucunun yerleştiremediği 7 doğru satırı düzeltildi; satırlar
>       `artifacts/…/batches/R01/row_fixes_2026-10-07.json`.
>   - M000 yeni motorla @550 0,9429. `M001_raw` nihai seçimi @300 0,9341. Ham koşu = yayın.
>   - R01 geçerli F1: bursiyer 0,8926, model 0,9287, hakem 0,9055 (eski motorla 0,9304 / 0,9107 / 0,9312; ayrıştırma ner LEDGER
>     7 Ekim 23:55).
>   - **`M001` = `dqcheck_loop_df9a27694e6a`**, dondurulmuş ağaç `/Users/student2/dq-run-m001` (etiket `run/m001-2026-10-07`).
>     - Gözetmen `loop/supervise_m1.sh` (tmux `gt750-m1-train`, günlük `loop/m1_train.log`); `STOP_M1` durdurur.
>     - Boş bellek ≥ 45 GB olunca başlar; 23:53'te 32 GB, makinede başka bir kullanıcının modeli çalışıyor. Ham kolun tepe
>       belleği 66,7 GB'tı.
>   - **M001'den sonra:**
>     1. ikinci hat `loop-train --round 1 --view full_documents` ve `loop-compare --round 1`;
>     2. R02 tahmini (`loop-predict`);
>     3. `M002_raw`.
>   - **Açık:**
>     - dq push (uzak depo herkese açık; karar kullanıcıda);
>     - ner `CANLI_DURUM.md` üreticisi 4 Ekim'de kalmış.

> **2 Ekim 2026, 12:42 — TEK KARŞILAŞTIRMA MOTORU. Referans listelerini karşılaştıran her şey
> `src/data_quality_checker/reference_matching.py`** (kullanıcı: "döngü motoru ve değerlendiriciyi farklı iki şey gibi
> görmeyelim"). Kullanıcılar: doğrulama F1'i (`benchmark/reference/matching.py` yalnız yeniden dışa aktarır), yönlendirme,
> uzman ekranı, dergi tablosu, hata atfı/kappalar, referans denetimi, hakem ölçüleri. Kurallar:
> - ad yedeği (numarasız satır, adın belgede/kanonda tek numarası varsa onu alır; ad iki numaraya karşılık geliyorsa
>   adla eşleşir);
> - aynı listede maddesi de olan kanunun yalnız kanun satırı fazladır: gerçekte iddia değil, adayda mazur (kullanıcı
>   "yalnız aynı listede" seçti);
> - 213/413 iki taraftan; tp+fn = gerçeğin referans sayısı.
>
> Yeni bir karşılaştırma yazma; `compare_with_policy` / `compare_reference_lists` kullan. Gardiyan
> `tests/test_one_reference_comparison.py`. M000 (P4, GT v8) `dqcheck_loop_88cf7872b3ed` 12:42'de başladı (tmux
> `gt750v8-train`); R01 zinciri `gt750v8-chain`. P0'ın doğrulama değeri (tek motor, v8) 0,9443.

> **2 Ekim 2026 — DÖNGÜ PROMPT P4 + GT v8 İLE SABİTLENDİ. Önce bunu oku** (1 Ekim kutusu aşağıda, hâlâ geçerli).
>
> - **Ayar:** `configs/presets/gt750_law_article_loop.json` = prompt **P4** (`loop.prompt_variant =
>   law-article-p4-combined`), **GT v8** (`data/ground_truth/gt_v8_gt750_law_article_2026-10-02/`, salt okunur),
>   kökler `data/sensitive/data_quality_checker_gt750_law_article_p4_gtv8`, `artifacts/data_quality_checker_gt750_law_article_p4_gtv8`.
>   Önceki hâli (P0, GT v7, eski kökler) bayt bayt `gt750_law_article_loop_p0_gtv7.json`; M000 v2
>   (`dqcheck_loop_5ba971453375`, @571) o kökte, dokunulmadı. M000 v2'ye bir şey koşacaksan **o** ayarla.
> - **Prompt taraması bitti** (önkayıt `ner-project/Journal/experiments/dq_loop_gt750/prompt_screen_preregistration.json`):
>   aşama 1 (temel model, 100 eğitim belgesi) P1 0,852 > P4 0,825 > … > P0 0,746 → aşama 2'ye P0, P1, P4; aşama 2
>   (150. adım, 50 doğrulama) **P4 0,918** > P0 0,903 > P1 0,863 → P4 (fark anlamsız: +0,015 [−0,020, +0,049]).
>   Kayıt sırasında bulunan karışıklık: 1.536 jetonluk eğitim bağlamı promptu da taşıyor → P4'te 3.657 pencere / 915 adım
>   (P0 2.284 / 571), pencere başına metin 200 jeton (P0 416). Bu yüzden **doğrulama kuralı önceden yazıldı**
>   (`prompt_screen_confirmation_preregistration.json`): P4'ün tam epoch M000'ı GT v8'de M000 v2'den (**v8'de @571
>   0,9424**, `.../law_article_p4_gtv8/loop/rescored/2026-10-02_v2_M000_P0_at_gt_v8`) düşükse prompt P0'a döner ve M000
>   P0 ile v8'de yeniden eğitilir.
> - **GT v8** = v7 + denetimin 15 düzeltmesi (ner `fba4ef8`): 7 tablo/tarife bölümü satırı düştü (§3), 3 değiştiren
>   kanun satırı eklendi (§14; biri doğrulamada, doc 455), 5 tireli aralık "ila" yazımına çekildi (§4, kimlik aynı).
>   Eğitim 2.412, doğrulama 249 satır; test bayt bayt v7 (okunmadı).
> - **Yeni komut** `loop-rescore --run-id R --label L --reason …`: koşunun doğrulama noktaları ayarın GT'siyle
>   yeniden puanlanır (`loop/rescored/L/`), koşunun kendi değerlendirmeleri değişmez.
> - **Çalışanlar / sıradakiler** (yeni kökte `loop/`): `supervise_m0.sh` (M000, P4, v8; günlük `m0_train.log`),
>   `r01_chain.sh` (M000 bitince R01 tahmin → 23 kararın taşınması → hakem → sıfırdan kanıt → ekran 5056),
>   `serve_R01.sh`. Kanıt betiği artık kökleri ayardan okur (P0 kopyası `_p0_gtv7.py`). P4'ün v7 üstündeki devam koşusu
>   (`dqcheck_loop_911f6e96a80f`) yalnız eşit epoch payı noktası için 250. adıma kadar sürdü, hiçbir karar vermez.
> - **İkinci hat** (tam belge M000) ve `loop-compare --round 0`: R01 zincirinden sonra, yeni ayarla.

> **1 Ekim 2026 — REFERANS = KANUN NO + MADDE. Önce bunu oku.** Kullanıcı kararı: fıkra ve bent
> işten tamamen çıktı. Ne modelin ve hakemin promptunda, ne eğitim hedeflerinde, ne değerlendiricide,
> eşleyicide, yönlendiricide, ekranda ya da ölçülerde geçer; fıkra/bent farkı yüzünden hiçbir bayrak
> kalkmaz. Plan: `/Users/student2/ner-project/docs/superpowers/plans/2026-10-01-kanun-madde-only.md`.
>
> - **Yeni döngü (v2):** `--config configs/presets/gt750_law_article_loop.json`, kökler
>   `data/sensitive/data_quality_checker_gt750_law_article`, `artifacts/data_quality_checker_gt750_law_article`.
>   Altın veri **GT v7** (`data/ground_truth/gt_v7_gt750_law_article_2026-10-01/`, salt-okunur; v6'dan
>   `ner-project/scripts/gt750/reduce_law_article.py` ile; 4.356 satır → 3.561; birleşen satırların bütün
>   alıntıları `source_texts`'te). Bölme (SPLIT.json bayt bayt aynı), havuz, 4×250+191 paket, belge anahtarı
>   (`data/sensitive/data_quality_checker_gt750/keys/gt750_loop.key`) aynı.
> - **Eski döngü (v1, fıkra/bent) ARŞİV:** `gt750_loop.json` ve `*_gt750` kökleri dokunulmadan duruyor
>   (M000 v1, R01 v1 tahminleri, 23 uzman kararı, 172 hakem kararı). Kod: etiket `archive/gt750-v1-fikra-bent`.
>   Veritabanı yedeği `data/sensitive/archive_v1_fikra_bent_2026-10-01/` (salt-okunur). GT v6 salt-okunur.
>   Codex'in commit edilmemiş işi dal `archive/codex-route-refresh-wip-2026-10-01`.
> - **Kimlik kuralı (her motor):** (kanun_no, madde); numara yoksa katlanmış ad. `kanun_ad` etiket.
>   Maddeye yapışık alt parça kesilir ("13/a", "17/4-a", "10 (2)" → madde); eklenmiş madde harfi büyük
>   kalır ("32/A", iki kez uygulamak sonucu değiştirmez). Kanun düzeyi satır, o kanunun maddesi varsa düşer.
>   Kovalar: YEŞİL / KIRMIZI / KARANTİNA — **SARI yok**. Gardiyan: `tests/test_no_paragraph_anywhere.py`.
> - **Model:** prompt `few-shot-cot-v3-en-compact-recall-law-article-v1` (üç alan), görünüm `lawarticle-v12`.
>   **M000 v2** `dqcheck_loop_5ba971453375` 1 Ekim 20:32'de başladı (tmux `gt750la-train`, gözetmen
>   `.../loop/supervise_m0.sh`, günlük `m0_train.log`); 23 parça × 25 adım + doğrulama, ~10 sa.
>   **Eğitim sürerken şu dosyalara DOKUNMA** (çalıştırıcı ve değerlendirici her doğrulamada baştan yükler;
>   devam ederken sözleşme değişmesin): atomic, checkpoints, config, constants, contracts, errors,
>   fake_training, fingerprints, g0, g0_finalize, g0_training, heartbeat, locking, mlx_compute, mlx_stateful,
>   mlx_worker, normalization, performance, preparation, processing, reference_policy, router, safezip,
>   storage, text, context_windows, loop_train, loop_sources, loop_ops; benchmark/reference/{evaluate,matching}.py;
>   annotator/reference/common.py.
> - **PROMPT TARAMASI (kullanıcı kararı 1 Ekim ~21:40) — R01'den önce:** "küçük alt kümelerde prompt denemesi,
>   en iyisiyle her şey sabit". Yöntem iki aşama (kullanıcı seçti); önkayıt
>   `ner-project/Journal/experiments/dq_loop_gt750/prompt_screen_preregistration.json`, plan §8. Kod dalı
>   `feat/prompt-screen` (worktree `/Users/student2/dq-prompt-screen`, `b0fdd83`): `g0.PROMPTS` (şimdiki + 6 aday),
>   `g0.loop_prompt(config)` (`loop.prompt_variant`), `loop-train --stop-at N`, `loop-prompt-screen`. Bekleyen betik
>   tmux `gt750la-prompt` (`.../loop/prompt_screen.sh`, günlük `prompt_screen.log`): eğitim bitince dalı birleştirir →
>   testler → aşama 1 (temel model, 100 eğitim belgesi, 7 prompt) → aşama 2 (en iyi 2 aday, kendi ayarlarıyla 150 adım;
>   şimdiki promptun karşılığı M000 v2 @150). **Kazananı karşılaştırıp raporlamak ve R01 zincirini o promptla yeniden
>   kurmak elle yapılacak.** Ana çalışma ağacını temiz tut (birleştirme bozulmasın).
> - **Zincir DURDURULDU (21:43, prompt kararı yüzünden; betik duruyor) (tmux `gt750la-chain`, `.../loop/r01_chain.sh`, günlük `r01_chain.log`):** eğitim bitince
>   `loop-predict --round 1` → `loop-carry-decisions --round 1 --from-config configs/presets/gt750_loop.json`
>   (uzmanın 23 kararı kanun+madde izdüşümüyle, olay geçmişi, asıl zaman ve `carried_from` ile; önce iki partinin
>   aynı belgeler olduğu kanıtlanır) → `loop-judge --round 1` (yeni hakem promptu, aday sözleşmesi v2) →
>   sıfırdan kanıt `data/sensitive/e2e_proof_2026-10-01/e2e_journal_metrics_law_article.py`
>   (`output_law_article.txt`) → tam tutarsa ekran 5056 (tmux `gt750la-serve`, `.../loop/serve_R01.sh`).
>   Adım düşerse zincir durur.
> - **İkinci hat DURDURULDU (prompt seçiminden sonra kurulacak) (tmux `gt750la-second`):** zincir bitince tam belge M000 v2 (`--view full_documents`), sonra
>   `loop-compare --round 0` → `.../loop/m0_compare.json`.
> - **Liste imi düzeltmesi (`b5a07e6`, eğitim sürerken, bilinçli):** "-1/1" maddesi artık madde 1; @25 eski kuralla,
>   @50+ yeni kuralla puanlandı → eğitim bitince iki eğri de son kuralla yeniden puanlanmalı.
> - **Biçim karşılaştırması:** eski M000'ın doğrulama eğrisi yeni değerlendirici + GT v7 ile yeniden puanlandı
>   (`.../loop/rescored/2026-10-01_v1_M000_at_law_article/`; @400 0,9272 → 0,9314, temel 0,7815 → 0,8067).
>   M000 v2 bitince eşleştirilmiş bootstrap ile kıyaslanacak (yalnız doğrulama).
> - **Bulucu:** alt parça sıralaması kalktı (doc_405'te 3065/1'i kararname maddesine koyuyordu). Gardiyan GT v7'de:
>   3.561'in 3.368'i altın alıntıda (eski bulucu kanun+madde düzeyinde 3.376). Yer değiştiren 10 kimliğin test dışı
>   8'i okundu, hepsi aynı kanun+maddenin alıntılanmamış başka gerçek bir geçişinde. **Açık (M001 öncesi):**
>   yayımlanan belgelerde bulucu tek yer veriyor; bütün açık geçişleri vermek doğruluğu ölçülmeden açılmayacak
>   (ek geçişlerin %47–58'i bir altın alıntıya denk geliyor, gerisi elle okunmadı).
> - **Kaldırılanlar:** döngünün kullanmadığı, fıkra/bent isteyen prompt ve çıkarıcı kopyaları
>   (`annotator/reference/{llm_prompt,gt_generation_prompt,llm,llm_common,llm_providers,regex,spacy,langextract_provider,gemini_aiplatform}.py`,
>   `benchmark/reference/run.py`) bu depodan çıktı; ner-project'te duruyorlar (ASYU bildirisinin yeniden üretimi için).
>   Q36-P1 (arşiv deneyi) kilidi eski prompta bağlı → artık reddeder (bilinçli).
> - **Kayıt:** `ner-project/Journal/experiments/dq_loop_gt750/` artık v2'yi yazar; v1 kaydı `archive_v1_fikra_bent/`.


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
>   `loop-judge --round N` (hakemi sabitler + çalıştırır + `judged` işaretler),
>   `loop-mark --round N --stage adjudicated`, `loop-open-test --confirm-open-test`
>   (yalnız son turdan sonra). Uzman/sürüm: `serve`, `release` `--batch-id R0k --generation M00(k-1)`.
>   Döngüde `pilot-judges`/`judge-lock` reddedilir (pilot yok).
> - Başlama kapısı doğrulamada (M000 > eğitilmemiş temel model); **test en sona kadar kapalı**.
> - Puanlamada `213/413` tahminden de düşer (loop ayarında `canonical_evaluate`).
> - **Durum 25 Eylül 17:45:** M000 hazır (400. adım; doğrulama F1 0,927, temel model 0,782 —
>   kapı geçti; `artifacts/data_quality_checker_gt750/loop/rounds/round_000_model.json`).
>   R01: 250 belge hazır, M000 ile tahmin edildi (GREEN 108 / YELLOW 40 / RED 102).
>   **Pilot kaldırıldı (29 Eylül, aşağıda).** Sıra: `loop-judge --round 1` (29 Eyl 19:15'te başlatıldı,
>   tmux `gt750-judge`, 172 belge) ‖ uzman 172 belgelik kuyruğu karara bağlar (hakemi beklemez) →
>   `release --batch-id R01 --generation M000` → `loop-mark --round 1 --stage adjudicated` → `loop-train --round 1`.
> - Uzman ekranı: `http://127.0.0.1:5056/` — **anahtarsız, giriş sayfası yok** (kullanıcı kararı 1 Ekim,
>   dq `acac446`; paylaşımlı makinede 127.0.0.1'e herkesin ulaşabileceği söylendi). Başlatma:
>   tmux `gt750-serve` ← `artifacts/data_quality_checker_gt750/loop/serve_R01.sh` (oturum imzası
>   `keys/serve_env.sh`). Uzaktan: `ssh -L 5056:127.0.0.1:5056 …`. 5055'teki ekran **eski** M01 turunundur
>   (o anahtarlı kalır).
> - Belge kimliği anahtarı: `data/sensitive/data_quality_checker_gt750/keys/gt750_loop.key` —
>   R01…R05'in hepsinde aynısı kullanılmalı.
> - **Üç kaynaklı hata (26 Eylül):** `loop-errors --round N` bursiyer / model / hakemi uzmanın
>   kararına karşı puanlar (kaçırılan, fazladan, detay = doğru madde yanlış fıkra/bent; P/R/F1) → `artifacts/data_quality_checker_gt750/loop/rounds/round_NNN_errors.json`;
>   `loop-mark --stage adjudicated` bunu kendisi yazar.
> - **Hakem pilotu KALDIRILDI (kullanıcı kararı 29 Eylül, dq `88e57a3`):** "pilot ve hakemi denetlemek
>   uzmana yük; hakemin hataları zaten ölçülüyor; hakem kesinlikle Gemma 4". Döngüde hakem önceden
>   sabit (`fix_loop_judge`: yalnız `gemma-4-31b-it-optiq-4bit`, yerel MLX), uzmanın incelediği her
>   belgede çalışır (kırmızı + sarı + yeşil örneklem), **uzmana hiç gösterilmez** (bölüm, düğme, J
>   kısayolu yok; `judge_override` reddedilir) → her karar kör; olay `judge_fixed`, hata raporunda
>   `judge_visible_to_expert=false`, tüm kararlar `before_judge_lock`'ta. Kuyruk R01 187 → 172.
>   Sürüm, hakemin incelenen her belgede geçerli kararını ister. Pilotun 60 geçerli kararı yeniden
>   kullanılır. Eski (pilotlu, M01) düzen değişmedi.
> - **Uzman ekranı gerçek tarayıcıyla sınandı (26 Eylül, canlı R01'in kopyasında, 19/19):**
>   her düğme ve kısayol yazdığını söylediği şeyi yazıyor. Düzeltilenler: kısayollar Cmd/Ctrl ile
>   ya da düzeltme paneli açıkken karar göndermiyor (önceden Cmd+M "model doğru" kaydediyordu);
>   elle yazılan teknik JSON artık kayboluyor değil kaydediliyor; çift tık tek karar; gerekçesiz
>   erteleme sayfada uyarı; ham JSON hata sayfaları Türkçe sayfa oldu; yedek bir kez başarısız
>   olunca sonraki sayfa yedeği tamamlıyor (önceden yeniden başlatmaya kadar 500). **Karar süresi**
>   artık sayfanın açık kaldığı süre (`seconds_on_page`, olay kaydında); önceden belgenin
>   hazırlandığı andan beri geçen süreydi. Tarayıcı testi: `tests/test_hitl_browser.py`.
> - **Mekanizma denetimi (27 Eylül; 1. tur canlının kopyasında uçtan uca, gerçek komutlarla):**
>   `release` sabit "G0" arıyordu → hiçbir tur sürüme çıkamazdı; artık
>   `release --batch-id R0k --generation M00(k-1)`. M001 eğitimi sürümü `releases/<parti>/`
>   altında arıyordu, gerçek sürüm `releases/<parti>/<sürüm_kimliği>/` → artık mühürlü sürüm
>   özetleriyle doğrulanıp okunuyor. **Tur modeli koruması:** loop ayarında `serve`,
>   `pilot-judges`, `release` R0k için yalnız M(k-1)'i kabul eder. R01 = GT750 M000
>   (`dqcheck_loop_519d9b43e168` @400, eğitim = mühürlü 500; test/doğrulama hiç görülmedi).
> - **Kullanıcı kuralı (27 Eylül): bursiyer `source_text` hiçbir karşılaştırmaya/karara girmez**
>   (yalnız vurgulama); 213/413 her karşılaştırmada iki taraftan düşer. Ekran farkı, hata atfı
>   ("dayanak" satırı kaldırıldı), yeşil örneklem katmanı ve kanal seçimi (artık HTML varsa HTML)
>   alıntıdan arındırıldı; R01 yeşil örneklemi yeniden seçildi (eskisi
>   `green_audit.superseded_2026-09-27_source_text_stratum.json`).
> - **Bulucu + madde yazımı (27 Eylül, `e9a3cc8`):** bulucu alıntısız yeniden yazıldı (GT750:
>   %92,7 altın alıntıda, 3 ret, yazıma %100 dayanıklı); "geçiçi/GEÇİCİ/Geçici Madde N", "ve
>   izleyen/devamı", "Değişik N", "466-469" iki motorda tek kimlik (bent harfleri katlanmaz).
>   Evaluator değişti → M000 doğrulaması yeniden puanlandı
>   (`artifacts/data_quality_checker_gt750/loop/rescored/2026-09-27_article_spelling/`; seçim ve
>   kapı aynı). R01 eski kuralla yönlendirildi (yeni kuralla 2 kırmızı yeşil olurdu; bırakıldı).
>   M001'de her sürüm referansının penceresi bulucuyla bulunur.
> - **Bulucu sağlamlaştırma (28 Eylül, `5798798`):** GT750 altın alıntı üstünde 4.040 → 4.077 (%93,6);
>   ret 3 (tablo/tarife, satır almaz); yazım değişikliklerine dayanıklı (madde 3.861/3.861, alan
>   33.481/33.519); 17 → 12 sn. Kanun okuma: parantezli kısaltma, tireli ad, önüne yapışan sözcük,
>   değiştiren kanun yalnız "Kanunla/ile", "Kanununa eklenen" alan kanunun, "Bazı Kanunlar … KHK'de
>   Değişiklik" tek kanun, "2007/13033 sayılı BKK", "520-541" aralık, "Geçici 19 ve 20". **Gardiyan:**
>   `tests/test_reference_locator_gt750.py` (altındaki 4.077 referans altında kalır; değişiklik bilinçli
>   olmalı → fikstürü yeniden üret). Denetim araçları ve önce/sonra fotoğrafları
>   `data/sensitive/locator_audit_2026-09-28/` (`diff.py` kanıt gücüyle sınıflar). M000'ı etkilemez (eğitim
>   belgeleri kendi doğrulanmış alıntısını kullanır); M001'den itibaren iki hatta aynı. R01 ret: bursiyer
>   11, M000 16, tam belge 16 — hepsi yanlış/biçimsiz referans veya tablo satırı.
>   **Yan bulgu (karar bekliyor):** ortak `normalize_law_number` "2007/13033"ü "2007" yapar → aynı yılın iki
>   kararı yönlendirmede tek kimlik; puanlama motoru ham değeri kullanıyor olabilir (motor uyumu testinde yok).
> - **Açık karar (kullanıcı):** bulucunun yerleştiremediği referans — canlı R01'de retlerin
>   neredeyse hepsi gerçekten yanlış referans. `round_training_documents` şimdilik uzman listesiyle
>   durur (yanlış veri üretmez).
> - **Eğitim biçimi ablasyonu (28 Eylül) — döngü PENCERELİ sürer:** M000 iki biçimde eğitildi:
>   pencereli (1.536 token; mühürlü M000, `dqcheck_loop_519d9b43e168` @400, doğrulama F1 0,927) ve
>   tam belge (`dqcheck_loop_02fd08a7a9d1` @125, 0,920; doğrusal dikkat parça-paralel WY formülüyle,
>   `mlx_memory.py`; en uzun belge 7.109 token 24,2 GB; 2 sa 22 dk). Eşleştirilmiş bootstrap %95
>   [−0,026, +0,014] → anlamlı fark yok; doğrulama F1 kuralı pencereliyi seçti. **Kullanıcı kararı:**
>   döngü pencereli sürer, M001+ M000 gibi (stok yineleme) eğitilir; `loop_train.training_view_mode`
>   varsayılanı `windows`, tam belge yalnız `"loop": {"training_view": "full_documents"}` ile.
>   Kanıt `artifacts/data_quality_checker_gt750/loop/candidates/round_000_ablation.json`; tam belge
>   adayı mühürlenmedi. M000 ve R01 tahminleri olduğu gibi geçerli.
> - **İKİ HAT, HER TUR (kullanıcı kararı 28 Eylül — "tam belge modelini her zaman sakla"):** tam belge
>   modeli ikinci hat olarak tutulur ve her turda **aynı temizlenmiş veriyle** eğitilir; mühürlenmez,
>   hiçbir partiyi tahmin etmez (Ar-Ge: ileride kullanılabilir). Tur k'nın sırası: sürüm +
>   `loop-mark --stage adjudicated` → `loop-train --round k` (pencereli, mühürlü M00k) →
>   `loop-predict --round k+1` (uzman işe başlayabilir) → `loop-train --round k --view full_documents`
>   (ikinci hat; uzman çalışırken koşabilir, GPU'yu `loop-judge` ile çakıştırma) →
>   `loop-compare --round k` (iki hat + her hattın kendi önceki turuna göre kazancı, yalnız doğrulama).
>   İkinci hattın kaydı `loop/candidates/round_NNN_full_documents.json` (+ `_registry.json`, adaptör
>   sha256); herhangi bir partide `process_batch(registry_path=...)` ile koşturulabilir. Mühürleme
>   yalnız pencereli görünümü kabul eder. Tam belge M000'ın R01 tahminleri:
>   `data/sensitive/gt750_whatif_full_root/` (kopya; merak ölçümü, makaleye girmez).
>   **Önkayıt:** test en sonda açılınca iki son model de puanlanır, yalnız raporlanır (seçim değişmez).
> - **Dergi tablosu (1 Ekim, dq `850673e` + `4d7c34e`):** `round_NNN_errors.json` → `reference_table`
>   (kaynak başına bulunan / hatalı / kaçırılan / detay, referans + hüküm düzeyi, %95 aralık; kapsam `reviewed` ve
>   `batch` = + yeşil uzlaşı, yalnız yeşil denetim geçince); `loop-report` → `loop/rounds/batch_trend.json` (paket başına
>   F1, önceki pakete fark, turların doğrulama F1'i); `loop-mark --stage adjudicated` ikisini de yazar. Her uzman kararı
>   olayı tam listeyi + önceki hâli taşır (karar geçmişi eksiksiz). Sıfırdan kanıt (kopya, 172 belge, her eylem türü,
>   bağımsız yeniden sayım) 20/20: `data/sensitive/e2e_proof_2026-10-01/`. Journal: `metrics/reference_table.csv`,
>   `metrics/batch_trend.csv`.
> - **KAYIT (makalenin kaynağı):** her aşamadan sonra
>   `cd /Users/student2/ner-project && /opt/llm-lab/.venv/bin/python Journal/analysis/scripts/dq_loop_gt750_record.py`
>   + `Journal/experiments/dq_loop_gt750/LEDGER.md`'ye giriş. Makine yükü örnekleyicisi
>   tmux `gt750-load`'da hep açık olmalı (`Journal/analysis/scripts/dq_loop_machine_load.sh`).
> - GPU paylaşımlı: M000'ın ilk 3 saatini student4'ün toplu değerlendirmesi yedi (Metal
>   bellek yetersizliği). Uzun eğitimler `artifacts/data_quality_checker_gt750/loop/supervise_m0.sh`
>   gibi GPU boşken deneyen bir gözetmenle koşulmalı.


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
