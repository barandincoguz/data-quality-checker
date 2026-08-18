# DQCheck Runbook

## 1. Yazılım kabulü

Repo kökünden:

```bash
cd /Users/student2/data-quality-checker
pytest -q
dqcheck --help
```

Bu adım model indirmez ve dış servise çağrı yapmaz.

## 2. G0 yazılım preflight

```bash
PYTHONPATH=src /opt/llm-lab/.venv/bin/python -m data_quality_checker \
  train-bootstrap --generation G0
```

Beklenen durum `software_preflight_passed_compute_pending` ve
`long_run_started=false` değerleridir. Canonical GT, example bank ve split
fingerprint'i değişmişse devam etmeyin.

## 3. Gerçek MLX kabul smoke'u

Bu adım pinned modeli indirebilir ve GPU/birleşik bellek kullanır. Uzun eğitim
değildir. Terminal kaybına karşı ayrı bir tmux session kullanın:

```bash
tmux new-session -s dqcheck-g0
cd /Users/student2/data-quality-checker
dqcheck train-bootstrap --generation G0 --execute
```

İzlenecek dosyalar ilgili run dizinindeki `compute_acceptance/heartbeat.json`
ve public `preflight.json` dosyasıdır. Preflight; exact revision, tokenizer,
no-think, sequence/memory, iki-update train, adapter generation ve gerçek
failure-resume eşdeğerliği geçmeden `long_run_allowed=true` yazmaz.

Bu makinede inference/forward-only validation bağlamı `12288`, backward eğitim
bağlamı `1536` olarak kilitlidir. Eğitim girdisi kaynak dokümanı kırpmaz;
`train_context_1536_manifest.json` içindeki text/reference coverage kapılarını
geçen örtüşmeli pencereleri kullanır. Run config'teki training-view SHA ile
manifest/output SHA'ları uyuşmadan resume reddedilir.

## 4. G0 development ve LR pilotları

Önce iki adayın kilitli sözleşmesini compute başlatmadan inceleyin:

```bash
dqcheck train-g0 --run-id <run-id> --candidate lr5e-5 --target-updates 50
dqcheck train-g0 --run-id <run-id> --candidate lr1e-4 --target-updates 50
dqcheck train-g0 --run-id <run-id> --candidate lr2.5e-5 --target-updates 25
```

Çıktıda blocker olmamalı; `max_sequence_length=1536`, tam validation bağlamı
`12288` ve training-view fingerprint'i görülmelidir. LR adayları yalnız
`peak_learning_rate` alanında farklı olmalıdır. Gerçek koşuları aynı anda
başlatmayın; ortak device lock ikinci yazıcıyı reddeder.

Her adayı ayrı bir `tmux` oturumunda sıralı çalıştırın:

```bash
tmux -L dqcheck-qwen35 new-session -s dqcheck-g0-lr5
cd /Users/student2/data-quality-checker
dqcheck train-g0 --run-id <run-id> --candidate lr5e-5 \
  --target-updates 50 --execute
```

Kesinti sonrası aynı komuta `--resume` eklenir. Aday dizinindeki `state.json` ve
`heartbeat.json` izlenir. Akış `25 update → full-state checkpoint → validation-50
→ 50 update → full-state checkpoint → validation-50` sırasını uygular. Her
validation belgesi atomik yazılır; aynı fingerprint'le resume tamamlanan belgeyi
yeniden üretmez.

Kabul edilen Mac Studio sözleşmesinde full-document inference/validation
`12288` token, backward eğitim `1536` tokendır. Güncel full max-10 görünümünde
`394` train belgesi lossless örtüşmeli/dense-replay pencerelerle `3726` satıra
dönüşür. Manifest hem metin hem gold
referans coverage'ını `complete` olarak doğrulamazsa eğitim başlamaz; bu akış
sessiz truncation değildir.

Güncel v9 trajectory'sinde seçilen `lr2.5e-5` önce `25`, sonra `50/75`
validation kapılarından geçirilir. Üçü de sağlıklıysa aynı stateful trajectory
bir training-view epoch olan `932=ceil(3726/4)` update'e uzatılır:

```bash
dqcheck train-g0 --run-id <run-id> --candidate lr2.5e-5 \
  --target-updates 932 --execute --resume
```

`932`, 3726 satırlık kilitli görünümü gradient accumulation `4` ile bir kez
tüketir. En iyi checkpoint her 25 update'teki generation metriğinden seçilir;
ara sağlık kapısı düşerse uzun hedefe devam edilmez.

Checkpoint seçimi validation-50 üzerinde `core-F1 → docwise@1.0 → recall →
validation loss` sırasındadır. Validation ve sealed test rolleri birleştirilmez:
toplam 100 holdout belgenin ilk 50'si model/hiperparametre seçimine, son 50'si
tek-seferlik sonuç kontrolüne ayrılmıştır. Test kapısı geçmeden final refit yoktur.

Canlı izleme:

```bash
tmux -L dqcheck-qwen35 attach -t <session>
jq '{status,stage,elapsed_seconds,last_successful_unit,last_error}' \
  Fine-Tuning/runs/<run-id>/development/<candidate>/heartbeat.json
```

### 4.1 2026-07-24 tarihsel ilk G0 pilot sonucu

Run `dqcheck_g0_qwen3_5_9b_ae56ead042b4` için iki 50-update pilotu tamamlandı.
`lr5e-5`, validation-50'de core F1 `0.4574` ve strict docwise `2/50`
üretti; `lr1e-4` core F1 `0.0000` ve strict docwise `0/50` kaldı.
Seçilen aday `lr5e-5`tir. Devam komutu:

```bash
PYTHONPATH=src /opt/llm-lab/.venv/bin/python -m data_quality_checker \
  train-g0 --run-id dqcheck_g0_qwen3_5_9b_ae56ead042b4 \
  --candidate lr5e-5 --target-updates 295 --execute --resume
```

Bu komut tarihsel run içindir; güncel v9 window/loop-recovery trajectory'sini
başlatmak için kullanılmaz.

### 4.2 Güncel validation fallback kontratı

Primary full-document generation her zaman önce ve `12288` input / `2048`
output sınırlarıyla çalışır. Yalnız primary parse-fail veya length-runaway
olursa belge `1024` tokenizer token + `256` overlap ile eksiksiz taranır; her
pencere en fazla `1024` token üretir. `covered_tokens == document_tokens`
olmadan fallback başarılı sayılmaz.

Bir pencere generation sınırına ulaşırsa recovery yalnız şu koşulların tamamında
çalışır: raw çıktı geçerli tam JSON object prefix içerir, terminalde art arda en
az iki aynı legal tuple vardır ve eksik suffix kullanılmadan prefix array olarak
kapatılabilir. Tam object'ler `(kanun_no, kanun_ad, madde, fikra, bent)` ile
dedup edilir; primary/raw window çıktıları provenance içinde korunur. Bu koşullar
yoksa belge hata olarak kalır. Sealed test bu development akışında açılmaz.

## 5. Batch hazırlama

Gerçek ZIP'leri repoya kopyalamayın. En az 32 byte HMAC anahtarı kullanın:

```bash
dqcheck prepare \
  --annotation-zip /izinli/yol/annotations.zip \
  --document-pool-zip /izinli/yol/documents.zip \
  --hmac-key-file /izinli/yol/dqcheck-hmac.key
```

`READY.json` oluşmadan process veya HITL açılmaz. Quarantine sayısını ve public
manifest checksum'larını kontrol edin.

Eski bir batch prepare sırasında anotatör kimliğini taşımadıysa HITL'den önce
özel attribution sidecar'ını üretin:

```bash
dqcheck import-attribution \
  --batch-id <batch-id> \
  --annotation-zip /izinli/yol/annotations.zip
```

Komut tüm batch belgelerini kaynak ZIP ile bire bir eşleştirmeden çıktı yazmaz.
Kullanıcı adları public artefaktlara değil yalnız
`data/sensitive/data_quality_checker/batches/<batch-id>/annotation_attribution.json`
dosyasına `0600` izinle ve atomik olarak yazılır.

## 6. Process, judge ve HITL

```bash
dqcheck process --prepared-batch <batch-id> --generation G0 --resume

# Ham predictionları değiştirmeden policy reroute preflight
dqcheck reroute --batch-id <batch-id>

# Preflight doğrulandıktan sonra backup + atomik apply
dqcheck reroute --batch-id <batch-id> --apply

dqcheck pilot-judges --batch-id <batch-id> --allow-external-judge
```

Varsayılan policy `ignore_vuk_213_article_413_v1`dir. Metin modele eksiksiz
verilmeye devam eder; yalnız normalize `(kanun_no=213, madde=413)` referans
görünümü router, judge, HITL ve release sınırında iki taraftan simetrik olarak
çıkarılır. Reroute bütün prediction dosyalarının checksum'unu doğrular, eski ve
yeni bucket sayılarını ayrı kaydeder ve review başlamışsa apply'ı reddeder.

Pilot uzman incelemesi tamamlandıktan sonra judge açıkça kilitlenir:

```bash
dqcheck judge-lock --batch-id <batch-id> \
  --model qwen3.5:397b --reason '<ölçülebilir seçim gerekçesi>'
dqcheck pilot-judges --batch-id <batch-id> --allow-external-judge
```

İkinci `pilot-judges` çağrısı kilitli judge için gerekli production coverage'ı
tamamlar.

HITL sunucusu:

```bash
export DQCHECK_SESSION_SECRET='<en-az-32-karakter>'
export DQCHECK_ACCESS_TOKEN='<en-az-32-karakter>'
dqcheck serve --batch-id <batch-id> --port 5055
```

Sunucu yalnız `127.0.0.1:5055` üzerinde dinler.
HITL ekranında insan anotasyonu ve G0 model çıktısı sabit, açık sütunlarda
gösterilir; körleme yoktur. İnsan sütununda export içindeki `completed_by`
(yoksa `last_editor`) kullanıcı adı gösterilir.

### 6.1 HITL karar yedekleri

Her review mutasyonu, ilgili `expert_review_updated` olayıyla aynı SQLite
transaction'ında yazılır. Ardından, aynı batch için process/thread lock altında
tam SQLite online backup alınır; integrity, foreign key, review-state
fingerprint'i, checksum ve son review event id doğrulanmadan istek başarılı
sayılmaz. En yeni beş doğrulanmış snapshot korunur.

Servis başlangıcında eksik veya geride kalmış snapshot canlı SQLite durumuna
otomatik yetiştirilir. `LATEST.json` ya da işaret ettiği snapshot bozuksa servis
fail-closed açılmaz. Ana review commit'i tamamlanmış fakat snapshot yazımı
başarısız olmuşsa API `503 durability_pending` ve `review_saved=true` döndürür;
form ekranı kararın kaydedildiğini ve tekrar gönderilmemesi gerektiğini açıkça
gösterir.

Canlı kontrol ve kurtarma provası:

```bash
dqcheck review-backup create --batch-id <batch-id>
dqcheck review-backup verify --batch-id <batch-id>
dqcheck review-backup status --batch-id <batch-id>
dqcheck review-backup restore-smoke --batch-id <batch-id>
```

`restore-smoke`, `LATEST` snapshot'ını hassas yedek kökü altında geçici ve
izole bir veritabanına geri yükler; canlı veritabanını değiştirmez. Başarılı
çıktıda `status=passed`, `integrity_check=ok`, foreign-key ihlali `0`, aynı
review fingerprint'i ve aynı son event id görülmelidir. Snapshot yolu
`data/sensitive/data_quality_checker/review_backups/<batch-id>/` altındadır ve
Git'e girmez. Aynı-disk snapshot'ı süreç/uygulama hatalarına karşı korur; disk
veya makine kaybı için ayrıca şifreli off-host yedek politikası gerekir.

## 7. Release

```bash
dqcheck status --batch-id <batch-id>
dqcheck release --batch-id <batch-id>
```

Release; prediction checksum, zorunlu review, deferred kayıt, GREEN audit ve
kilitli judge coverage kapılarından biri eksikse durur. Mevcut release üzerine
yazılmaz.

## 8. Q36-P1 kontrollü SFT preflight
 
Q36-P1, G0'dan bağımsızdır. Repo kökünden:

```bash
cd /Users/student2/data-quality-checker
dqcheck q36-p1 preflight
```

Komut pinned Qwen3.6-27B snapshotını ve tokenizerı kullanarak prompt/model/data
fingerprintlerini, GREEN split-leakage sınıflandırmasını ve üç evaluation
evreninin ortak token bütçesini ölçer. `green_audit.status=passed` olmadan
`--build-data` fail-closed durur; eğitim veya inference başlatmaz.

Audit review, bu runbook'un 6. bölümündeki yerel HITL ve backup sözleşmesini
kullanır. Audit geçtikten sonra training source exportu:

```bash
dqcheck q36-p1 preflight --build-data
```

Prediction/gold production görünümü:

```bash
dqcheck q36-p1 views --predictions <raw.json> --output-dir <views-dir>
dqcheck q36-p1 gold-view --source <gold.json-or-dir> --output <filtered.json-or-dir>
```

Tam Q36 sözleşmesi ve audit sonrası compute kapıları:
`tasks/q36_p1_controlled_sft_2026_08_04/{PLAN,RUNBOOK}.md`.

## Kurtarma

- Aynı input/config/model fingerprint'iyle komutu yeniden çalıştırın.
- Process için `--resume` kullanın; doğrulanmış belge çıktıları atlanır.
- `running` heartbeat bayatsa önce PID/host bilgisini kontrol edin.
- Checksum veya fingerprint uyuşmazlığında dosyayı elle düzeltmeyin; yeni run
  kimliği oluşturun ve olayı ROADMAP/status kaydına yazın.
- Tek doğrulanmış checkpoint'i silmeyin veya üzerine yazmayın.
- HITL kararları için önce `review-backup verify`, ardından
  `review-backup restore-smoke` çalıştırın; doğrulama başarısızsa canlı SQLite
  veya snapshot üzerinde elle değişiklik yapmayın.
