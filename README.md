# Data Quality Checker

Türkçe hukuk belgelerindeki insan anotasyonlarını, canonical-only Qwen3.5-9B
çıktıları ve uzman incelemesiyle karşılaştıran yerel bir kalite kontrol akışıdır.

Akış:

```text
ZIP hazırlama -> G0 prediction -> semantic router -> judge -> HITL -> release
```

## Güvenlik sınırı

- Ham belge kimlikleri, seçilen metinler, model/judge cevapları ve uzman
  kararları yalnız `data/sensitive/data_quality_checker/` altında tutulur.
- `artifacts/data_quality_checker/` yalnız HMAC kimlikleri, checksum'lar,
  sayımlar ve reason-code içeren redakte manifestler içindir.
- ZIP dosyaları topluca extract edilmez. Güvensiz yollar, symlink, şifreli
  entry, duplicate ad, aşırı boyut ve compression ratio reddedilir.
- Canonical eğitim kaynağı yalnız
  `data/ground_truth/gt_v3_triangulated_2026-05-15/validated/` dizinidir.

## Çalıştırma

Repo kökünden, paket kurulumu yapmadan:

```bash
export PYTHONPATH="$PWD/data-quality-checker-weak-learning-program/src"
DQPY=/opt/llm-lab/.venv/bin/python

$DQPY -m data_quality_checker --help
$DQPY -m data_quality_checker train-bootstrap --generation G0
```

Editable kurulum istenirse:

```bash
/opt/llm-lab/.venv/bin/python -m pip install -e \
  './data-quality-checker-weak-learning-program[test,compute]'
dqcheck --help
```

Temel batch akışı:

```bash
dqcheck prepare \
  --annotation-zip /izinli/yol/annotations.zip \
  --document-pool-zip /izinli/yol/documents.zip \
  --hmac-key-file /izinli/yol/dqcheck-hmac.key

dqcheck process --prepared-batch <batch-id> --generation G0 --resume
dqcheck pilot-judges --batch-id <batch-id> --allow-external-judge
dqcheck serve --batch-id <batch-id>
dqcheck release --batch-id <batch-id>
```

`serve` yalnız `127.0.0.1` üzerinde açılır. Session secret ve erişim token'ı
ortam değişkenlerinden verilmelidir; ayrıntılar [runbook](docs/RUNBOOK.md)
içindedir.

## G0 compute sınırı

`train-bootstrap` varsayılan olarak canonical veriyi, split'i, checksum'ları ve
fake full-state resume testini hazırlar; model indirmez. `--execute` gerçek MLX
train/generate ve failure-resume kabul smoke'larını çalıştırır, fakat uzun G0
development eğitimi başlatmaz. Uzun koşu ancak bu preflight geçtiğinde ve ayrı
bir runbook adımıyla başlatılmalıdır.

Gerçek adapter mühürlenmeden `process` komutu G0 registry eksikliği nedeniyle
durur. Testlerdeki `--fake-backend` seçenekleri yalnız fixture kabulü içindir ve
help çıktısında bilerek gösterilmez.

## Test

```bash
cd data-quality-checker-weak-learning-program
/opt/llm-lab/.venv/bin/python -m pytest -q
```

Varsayılan testler model yüklemez. Gerçek MLX kabulü ayrı compute kapısıdır.

