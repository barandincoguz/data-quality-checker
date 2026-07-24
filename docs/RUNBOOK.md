# DQCheck Runbook

## 1. Yazılım kabulü

Repo kökünden:

```bash
cd data-quality-checker-weak-learning-program
/opt/llm-lab/.venv/bin/python -m pytest -q
PYTHONPATH=src /opt/llm-lab/.venv/bin/python -m data_quality_checker --help
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
cd /Users/student2/ner-project/data-quality-checker-weak-learning-program
PYTHONPATH=src /opt/llm-lab/.venv/bin/python -m data_quality_checker \
  train-bootstrap --generation G0 --execute
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
cd /Users/student2/ner-project
PYTHONPATH=data-quality-checker-weak-learning-program/src \
  /opt/llm-lab/.venv/bin/python -m data_quality_checker \
  train-g0 --run-id <run-id> --candidate lr5e-5 \
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

## 6. Process, judge ve HITL

```bash
dqcheck process --prepared-batch <batch-id> --generation G0 --resume
dqcheck pilot-judges --batch-id <batch-id> --allow-external-judge
```

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

## 7. Release

```bash
dqcheck status --batch-id <batch-id>
dqcheck release --batch-id <batch-id>
```

Release; prediction checksum, zorunlu review, deferred kayıt, GREEN audit ve
kilitli judge coverage kapılarından biri eksikse durur. Mevcut release üzerine
yazılmaz.

## Kurtarma

- Aynı input/config/model fingerprint'iyle komutu yeniden çalıştırın.
- Process için `--resume` kullanın; doğrulanmış belge çıktıları atlanır.
- `running` heartbeat bayatsa önce PID/host bilgisini kontrol edin.
- Checksum veya fingerprint uyuşmazlığında dosyayı elle düzeltmeyin; yeni run
  kimliği oluşturun ve olayı ROADMAP/status kaydına yazın.
- Tek doğrulanmış checkpoint'i silmeyin veya üzerine yazmayın.
