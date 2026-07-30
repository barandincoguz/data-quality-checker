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

dqcheck import-attribution \
  --batch-id <batch-id> \
  --annotation-zip /izinli/yol/annotations.zip

dqcheck process --prepared-batch <batch-id> --generation G0 --resume
dqcheck reroute --batch-id <batch-id>
dqcheck reroute --batch-id <batch-id> --apply
dqcheck pilot-judges --batch-id <batch-id> --allow-external-judge
dqcheck serve --batch-id <batch-id>
dqcheck release --batch-id <batch-id>
```

`serve` yalnız `127.0.0.1` üzerinde açılır. Session secret ve erişim token'ı
ortam değişkenlerinden verilmelidir; ayrıntılar [runbook](docs/RUNBOOK.md)
içindedir.

HITL ekranı insan anotasyonu ile G0 model çıktısını açık etiketlerle gösterir;
körleme uygulanmaz. `import-attribution`, eski hazırlanmış batch'lerde eksik
olan `completed_by`/`last_editor` bilgisini kaynak ZIP'ten belge kimliğiyle
eşleyip yalnız hassas veri kökündeki `0600` izinli sidecar'a yazar.

Her HITL kararı, review satırı ve `expert_review_updated` audit event'i aynı
SQLite transaction'ında commit edildikten sonra tam veritabanının doğrulanmış
online snapshot'ı alınır. Snapshot doğrulanmadan başarılı API cevabı veya form
redirect'i verilmez. En son beş doğrulanmış durum korunur; UI güncel korunan
karar sayısını gösterir. Operasyon komutları:

```bash
dqcheck review-backup create --batch-id <batch-id>
dqcheck review-backup verify --batch-id <batch-id>
dqcheck review-backup status --batch-id <batch-id>
dqcheck review-backup restore-smoke --batch-id <batch-id>
```

Bu snapshot'lar `data/sensitive/data_quality_checker/review_backups/` altında
ve ana veritabanıyla aynı makinededir. Yanlışlıkla silme/bozulma ve süreç
çökmesine karşı kurtarma noktası sağlar; makine veya disk kaybına karşı
off-host yedek yerine geçmez.

`reroute` varsayılan olarak salt preflight'tır. Sealed ham predictionları veya
belge metnini değiştirmeden sürümlü reference-policy görünümünü hesaplar.
`--apply`, prediction checksum/coverage ve pending-review kapıları geçtikten
sonra doğrulanmış SQLite backup alır ve yeni router bucket'larını tek
transaction'da uygular.

## G0 compute sınırı

`train-bootstrap` varsayılan olarak canonical veriyi, split'i, checksum'ları ve
fake full-state resume testini hazırlar; model indirmez. `--execute` gerçek MLX
train/generate ve failure-resume kabul smoke'larını çalıştırır, fakat uzun G0
development eğitimi başlatmaz.

Uzun development koşusunun kilitli planını compute başlatmadan görmek için:

```bash
dqcheck train-g0 \
  --run-id dqcheck_g0_qwen3_5_9b_<fingerprint> \
  --candidate lr5e-5 --target-updates 50
```

`train-g0` gerçek koşuyu 25-update segmentlere böler. Her segmentten sonra
full-state checkpoint'i doğrular, validation-50 generation ve canonical metrik
değerlendirmesini tamamlar. `--execute` yalnız yeşil compute preflight ve `tmux`
içinde çalışır; `--resume` tamamlanmış segment/validation kayıtlarını atlar.
İki başlangıç adayı yalnız peak LR bakımından farklıdır: `5e-5` ve `1e-4`.

Mac Studio compute kabulünde tam belge inference/validation bağlamı `12288`,
backward eğitim bağlamı `1536` token olarak doğrulandı. Uzun train belgeleri
sessizce kırpılmaz: örtüşmeli ve fingerprint'li doğal pencereler, belge başına
en fazla bir boş pencere ve çok-referanslı dense-replay satırlarıyla `394`
belge `3726` eğitim satırına dönüşür; tek replay hedefi en fazla `10` referans
taşır. Candidate metin coverage'ı negative
sampling öncesi, gold referans coverage'ı ise eğitim görünümünde eksiksiz
olmadan koşu açılmaz.

Hiperparametre seçimi validation-50 üzerinde yapılır. Ayrı test-50 seçim
sonrasında yalnız bir kez açılır; bu iki rol birlikte 100 holdout belgeyi
oluşturur. Test sonucu görülmeden final 494-belge refit başlamaz.

Gerçek adapter mühürlenmeden `process` komutu G0 registry eksikliği nedeniyle
durur. Testlerdeki `--fake-backend` seçenekleri yalnız fixture kabulü içindir ve
help çıktısında bilerek gösterilmez.

## Test

```bash
cd data-quality-checker-weak-learning-program
/opt/llm-lab/.venv/bin/python -m pytest -q
```

Varsayılan testler model yüklemez. Gerçek MLX kabulü ayrı compute kapısıdır.
