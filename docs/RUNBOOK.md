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

## 4. Batch hazırlama

Gerçek ZIP'leri repoya kopyalamayın. En az 32 byte HMAC anahtarı kullanın:

```bash
dqcheck prepare \
  --annotation-zip /izinli/yol/annotations.zip \
  --document-pool-zip /izinli/yol/documents.zip \
  --hmac-key-file /izinli/yol/dqcheck-hmac.key
```

`READY.json` oluşmadan process veya HITL açılmaz. Quarantine sayısını ve public
manifest checksum'larını kontrol edin.

## 5. Process, judge ve HITL

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

## 6. Release

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
