# Reference Benchmark Pipeline

> **Yorumlayıcı uyarısı.** Bu dosyadaki komut blokları `venv/bin/python` yazar,
> fakat bu repo paylaşımlı bir lab ortamında koşuyor ve **o yol her makinede
> bulunmaz** (bu Mac Studio'da yoktur; çalışan yorumlayıcı
> `/opt/llm-lab/.venv/bin/python`). Çözümleme sırası ve gerekçe: `AGENTS.md` §8.
> Ayrıca `evaluate.py`'yi **dosya yolu** ile çağırıyorsan
> `PYTHONPATH=<repo kökü>` zorunludur; `python -m benchmark.reference.evaluate`
> biçimi ise cwd repo kökü olduğu sürece PYTHONPATH gerektirmez.

> **2026-08-03:** `--matching-engine optimal` **kabul edildi**; post-audit'in
> açtığı iki blocker (uppercase `32/A` idempotence ve satır-sırası duyarlılığı)
> kapandı ve dört kabul kapısı geçti. `optimal` motoru artık kendi ön işleme
> zincirini sahiplenir; `--reference-postprocess` ve `--core-reference-view`
> yalnız `greedy` yolunu biçimlendirir. Dondurulmuş `greedy` koşuları
> etkilenmemiştir (Run `021` ile 9/9 bit düzeyinde). Kanıt:
> `artifacts/2026-08-03_evaluator-matching-semantics-audit/POST_AUDIT.md`.

Bu yapı, `gemini_annotate.py` ile uyumlu `references` formatında regex, spaCy ve LLM tabanlı annotator'lar ile evaluator sunar.

## Package Structure

- `annotator/reference/common.py`
  - Ortak normalizasyon, canonical kanun adı/numarası eşlemeleri, alias çözümleme.
- `annotator/reference/regex.py`
  - Regex baseline annotator (`sayılı + kanun`, kısaltma ve anafora destekli).
- `annotator/reference/spacy.py`
  - spaCy (`Matcher` + `PhraseMatcher`) baseline annotator.
- `annotator/reference/llm.py`
  - Provider-agnostic LLM annotator. Resmi aktif Gemini benchmark prompt varyanti `few-shot-cot-v3-en`dir; `zero-shot`, `few-shot-cot`, `few-shot-cot-v2`, `en-baseline`, `en-negative-guarded` ve `en-recall-recovery` tarihsel/deneysel comparator olarak korunur.
- `annotator/reference/llm_providers.py`
  - LLM provider interface + registry.
- `annotator/reference/llm_prompt.py`
  - Provider-neutral prompt builder.
- `annotator/reference/llm_common.py`
  - Ortak JSON repair, normalization ve output contract enforcement.
- `benchmark/reference/evaluate.py`
  - Ground truth karşılaştırması; `core_law_article_strict`, `docwise_core_accuracy`, `overall_reference` ve tanısal metrikleri üretir.
- `benchmark/reference/run.py`
  - Toplu annotate + evaluate orchestrator.

> **Note (2026-05-01 onward — `2026-05-01_common500` snapshot):** From the `2026-05-01_common500` snapshot onward, the filename for the `gemini-few-shot-en` method is `gemini-few-shot-en-predictions.json` (no count suffix). Historical snapshots retain `…-350-annotations.json` / `…-250-annotations.json` for traceability. The mapping between historical and canonical method IDs lives in `benchmark/predictions/method_registry.json` under each method's `historical_aliases` field; cross-snapshot tooling (e.g. an old-vs-new leaderboard comparator) should reconcile via that registry.

## Fair_344 Benchmark (2026-04-29) — HISTORICAL / SUPERSEDED

> **Superseded (2026-05-03 onward).** The paper-facing canonical benchmark
> is now `fair_494` (500-doc corpus minus the 6 exemplars), NOT `fair_344`.
> The ASYU paper uses GT v2:
> `benchmark/results/per_method/015-common500-benchmark-eight-method-newgt-v2/`
> (2026-05-03). The latest GT v3 cleaned core-F1 result is
> `benchmark/results/per_method/017-common500-benchmark-eight-method-gt-v3-cleaned/`
> (2026-06-10/11). The latest **NEW METRIC performance** run with
> `docwise_core_accuracy` legacy-view is
> `benchmark/results/per_method/019-common500-benchmark-nine-method-gt-v3-cleaned-new-metric/`
> (2026-06-20; Qwen2.5-14B LoRA v1 fine-tune included). The latest canonical-set row-view postprocess run is
> `benchmark/results/per_method/020-common500-benchmark-nine-method-gt-v3-cleaned-canonical-set-new-metric/`
> (2026-06-27; evaluator `--reference-postprocess canonical_set`). The core-set audit run is
> `benchmark/results/per_method/021-common500-benchmark-nine-method-gt-v3-cleaned-core-set-new-metric/`
> (2026-06-30; evaluator `--reference-postprocess canonical_set --core-reference-view core_set`). The `fair_344` artifacts below are historical provenance
> only — do not cite them as the current benchmark.

## Current Fair_494 Artefacts (2026-08-03)

Current benchmark interpretation must keep six artefact roles separate:

- ASYU paper-canonical frozen GT v2: `benchmark/results/per_method/015-common500-benchmark-eight-method-newgt-v2/`
- GT v3 cleaned primary core-F1 benchmark: `benchmark/results/per_method/017-common500-benchmark-eight-method-gt-v3-cleaned/`
- NEW METRIC legacy-view performance (`docwise_core_accuracy@1.0`, fine-tune dahil): `benchmark/results/per_method/019-common500-benchmark-nine-method-gt-v3-cleaned-new-metric/`
- NEW METRIC canonical-set row-view performance (`docwise_core_accuracy@1.0`, `--reference-postprocess canonical_set --core-reference-view row`): `benchmark/results/per_method/020-common500-benchmark-nine-method-gt-v3-cleaned-canonical-set-new-metric/`
- NEW METRIC core-set audit (`docwise_core_accuracy@1.0`, `--reference-postprocess canonical_set --core-reference-view core_set`): `benchmark/results/per_method/021-common500-benchmark-nine-method-gt-v3-cleaned-core-set-new-metric/`
- **Dual-engine sealed comparison** (`--matching-engine greedy` vs `optimal`, same Run `021` predictions, both GT snapshots): `benchmark/results/per_method/023-common500-benchmark-nine-method-dual-engine-sealed/` — paper-facing. Run `022` is the superseded provisional snapshot and is not cited.

`017`, `019`, `020`, and `021` use the same `fair_494` family: 500 corpus documents minus the six few-shot exemplar IDs `[1, 10, 16, 18, 36, 77]`. In `017`, the primary `core_law_article_strict` F1 winner is DeepSeek v3.2 Cloud (F1 `0.8825`). In `019` and `020`, the strict document-wise row-view `docwise_core_accuracy` winner is Qwen3.5 397B Cloud (`0.4069`, 201/494 documents passed). In `021`, the core-set audit `docwise_core_accuracy` winner is Qwen3.5 397B Cloud (`0.5182`, 256/494 documents passed), and the core-set core-F1 winner is DeepSeek v3.2 Cloud (F1 `0.9012`). These metric-specific and view-specific rankings should not be treated as a conflict. **All leader statements above are `greedy`-engine results.** Run `023` scores the same predictions with `--matching-engine optimal`, where the core-F1 winner is DeepSeek v3.2 Cloud (F1 `0.9348`) and the core-set `docwise_core_accuracy` winner changes to DeepSeek v3.2 Cloud (`0.5931`, 293/494) instead of Qwen3.5 397B Cloud. Ranking is engine-sensitive, so every leader claim must name `matching_engine` alongside the metric and `core_reference_view`.

Ground truth source depends on the workflow:

- Active annotation workflow: `data/annotations/doc_*.json` (350 docs on 2026-06-16).
- GT v3 cleaned benchmark snapshot: `data/ground_truth/gt_v3_triangulated_2026-05-15/validated/` (500 docs; `fair_494` after exemplar exclusion).
- GT v4 article-normalized snapshot: `data/ground_truth/gt_v4_article_normalized_2026-08-03/validated/` — valid **only** with `--matching-engine optimal`; identical results to GT v3 under that engine.

Current 9-method fair_494 benchmark includes: regex, spaCy, Gemini few-shot-cot-v3-en, Qwen3.5 397B Cloud, DeepSeek v3.2 Cloud, Gemma 4 31B Cloud, Mistral Large 3 675B Cloud, LangExtract + Gemini 2.5 Flash, and Qwen2.5-14B LoRA v1 local fine-tune.

Journal-facing use:

- Active journal manuscript workspace: `Journal/`.
- Locked journal input artifacts: `Journal/reproducibility/input_artifacts.lock.json`.
- Journal claim register: `Journal/evidence/claims/CLAIMS_REGISTER.csv`.
- Generated paper tables: `Journal/analysis/generated/tables/`.
- Regenerate journal tables from live JSON with `make -C Journal tables`.
- Run `make -C Journal audit` before promoting evaluator numbers into manuscript text.

Historical `fair_344` benchmark provenance (2026-04-29):

- directory: `benchmark/results/per_method/012-common344-benchmark-seven-method-single-status-approved/`
- JSON: `reference_eval_common344_seven_methods_approved_2026-04-29.json`
- CSV: `reference_eval_common344_seven_methods_approved_2026-04-29.csv`
- error buckets: `reference_eval_common344_seven_methods_approved_error_buckets_2026-04-29.csv`

This is a seven-method `fair_344` benchmark: 350 corpus documents minus the six
few-shot exemplar IDs in
`benchmark/reference/example_sets/few_shot_cot_v3_en_doc_ids.json`.
Historical fair_344 ground truth used `data/annotations/` in a single-status
approved regime. This is not the current journal/fair_494 GT source.

Generation command:

```bash
venv/bin/python -m benchmark.reference.evaluate \
  --predictions \
    benchmark/predictions/snapshots/2026-04-09_common350/regex_reference_annotations.json \
    benchmark/predictions/snapshots/2026-04-09_common350/spacy_reference_annotations.json \
    benchmark/predictions/snapshots/2026-04-09_common350/gemini-few-shot-en-350-annotations.json \
    benchmark/predictions/snapshots/2026-04-09_common350/ollama_few-shot-en-qwen3.5-397b.json \
    benchmark/predictions/snapshots/2026-04-09_common350/langextract_few_shot_cot_v3_en_gemini_2_5_flash_aiplatform_v2_reference_annotations.json \
    benchmark/predictions/snapshots/2026-04-09_common350/ollama_few_shot_cot_v3_en_gemma4_31b_cloud_reference_annotations.json \
    benchmark/predictions/snapshots/2026-04-28_common350/ollama_few_shot_cot_v3_en_mistral_large_3_675b_cloud_reference_annotations.json \
  --ground-truth-dir data/annotations \
  --docs data/test_data.json \
  --exclude-doc-ids-file benchmark/reference/example_sets/few_shot_cot_v3_en_doc_ids.json \
  --gt-mode approved_only \
  --per-doc \
  --json-report benchmark/results/per_method/012-common344-benchmark-seven-method-single-status-approved/reference_eval_common344_seven_methods_approved_2026-04-29.json \
  --csv-report benchmark/results/per_method/012-common344-benchmark-seven-method-single-status-approved/reference_eval_common344_seven_methods_approved_2026-04-29.csv \
  --error-bucket-csv benchmark/results/per_method/012-common344-benchmark-seven-method-single-status-approved/reference_eval_common344_seven_methods_approved_error_buckets_2026-04-29.csv
```

## Çıktı Formatı

Her doküman için:

```json
{
  "doc_id": 1,
  "ozelge_no": "...",
  "status": "success",
  "references": [
    {
      "type": "kanun_madde_referansi",
      "kanun_no": "213",
      "kanun_ad": "Vergi Usul Kanunu",
      "madde": "mukerrer 298",
      "fikra": "A",
      "bent": "1",
      "source_text": "..."
    }
  ]
}
```

LLM-specific behavior:
- Shared LLM outputs now apply document-level compaction.
- Aynı exact `kanun_no/kanun_ad/madde/fikra/bent` tuple belge içinde tekrar ederse yalnız 1 row tutulur.
- Aynı kanun için belge içinde en az bir spesifik `madde/fikra/bent` tuple varsa ekstra generic law-only row baskılanır.
- Bu politika shared LLM stack ve Qwen mirror notebook'ları için geçerlidir; regex/spaCy baseline'ları ayrı değerlendirilir.

## Metrik Özeti

Evaluator şu ana metrik yüzeylerini üretir:

| Metrik | Seviye | Alanlar / kaynak | Kullanım |
| --- | --- | --- | --- |
| `core_law_article_strict` | Referans satırı, mikro P/R/F1 | `kanun_no`, `kanun_ad`, `madde` | Birincil core-F1 leaderboard |
| `docwise_core_accuracy` | Doküman pass/fail | Dokümanın `core_law_article_strict` TP/FP/FN sonucu | **NEW METRIC performance**; strict belge-bazlı başarı |
| `overall_reference` | Referans satırı, mikro P/R/F1 | `kanun_no`, `kanun_ad`, `madde`, `fikra`, `bent` | İkincil / tanısal |

`docwise_core_accuracy` mevcut core eşleştirme mantığını değiştirmez. Her doküman için core F1 hesaplanır ve `--docwise-threshold` eşiğine göre pass/fail verilir. Varsayılan eşik `1.0` olduğu için doküman ancak tüm eligible core gold referansları eşleşirse ve ekstra yanlış core prediction yoksa geçer.

Metrik kaynakları:

- Genel metrik kataloğu: `docs/context/METRICS_GUIDE.md`
- Canonical ayrıntılı NEW METRIC rehberi: `docs/context/NEW_METRIC_TECHNICAL_GUIDE.md`
- Sunum ve anlatım rehberi: `docs/context/NEW_METRIC_PRESENTATION_GUIDE.md`

## Çalıştırma Komutları

Regex annotator:

```bash
venv/bin/python -m annotator.reference.regex --input data/test_data.json --output benchmark/predictions/regex_reference_annotations.json
```

spaCy annotator:

```bash
venv/bin/python -m annotator.reference.spacy --input data/test_data.json --output benchmark/predictions/spacy_reference_annotations.json
```

LLM annotator (Gemini zero-shot):

```bash
GEMINI_API_KEY=... venv/bin/python -m annotator.reference.llm \
  --input data/test_data.json \
  --output benchmark/predictions/gemini_zero_shot_reference_annotations.json \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --prompt-variant zero-shot \
  --run-summary benchmark/results/llm_run_summary.json
```

LLM annotator (Gemini few-shot-cot-v3-en, resmi aktif Gemini prompt varyanti):

```bash
GEMINI_API_KEY=... venv/bin/python -m annotator.reference.llm \
  --input data/test_data.json \
  --doc-ids-file benchmark/reference/example_sets/gold_holdout_clean_doc_ids.json \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --prompt-variant few-shot-cot-v3-en \
  --temperature 0.0 \
  --run-summary artifacts/benchmark/prompt_optimization/few_shot_cot_v3_en/final_candidate/holdout133_run_summary.json
```

Not:
- Repo içi global default model bu fazda değiştirilmedi.
- İlk resmi LLM benchmark koşularında model override ile `gemini-2.5-flash-lite` kullanılmalıdır.
- Resmi aktif Gemini benchmark prompt varyanti `few-shot-cot-v3-en`dir.
- `few-shot-cot`, `zero-shot`, `few-shot-cot-v2`, `en-baseline`, `en-negative-guarded` ve `en-recall-recovery` tarihsel/deneysel comparator olarak korunur; yeni official rerun hedefi gibi ele alinmamalidir.
- Shared LLM prompt/policy su anda belge-seviyesi iki ek kural uygular:
  - ayni kanun icin en az bir spesifik `madde/fikra/bent` tuple'i varsa generic law-only row tutulmaz
  - ayni `kanun_no / kanun_ad / madde / fikra / bent` tuple'i belge icinde tekrar etse bile yalniz 1 kez tutulur
- Aktif/current fair_494 kiyas seti: `regex`, `spaCy`, `Gemini few-shot-cot-v3-en`, `Qwen3.5 397B via Ollama Cloud`, `DeepSeek v3.2 via Ollama Cloud`, `LangExtract + Gemini 2.5 Flash`, `Gemma 4 31B Cloud via Ollama Cloud`, `Mistral Large 3 675B via Ollama Cloud`
- `Gemini zero-shot` ve `Gemini few-shot-cot` tarihsel comparator olarak korunur; yeni veri genislemelerinde varsayilan guncel Gemini rerun hedefi degildir
- Acik kaynak Qwen2.5-7B benchmark sonuclari tarihsel kayittir: `benchmark/results/per_method/004-open-source-llm-reference-benchmark/`
- Qwen2.5-7B prediction dosyasi tarihsel comparator icindir: `benchmark/predictions/open-source-llm-predictions/qwen2_5_7b_few_shot_cot_v3_en_reference_annotations.json`
- Qwen2.5-7B Kaggle notebook tarihsel comparator icindir: `notebooks/kaggle_qwen2_5_7b_benchmark.ipynb` (standalone, 2xT4 GPU, 4-bit quantized)
- `--temperature` ve `--run-label` ayni prompt varyanti icin deneysel kosulari dosya adi cakismasi olmadan ayirmak icindir.

Evaluator (tek/çoklu prediction):

```bash
venv/bin/python -m benchmark.reference.evaluate \
  --predictions benchmark/predictions/regex_reference_annotations.json benchmark/predictions/spacy_reference_annotations.json \
  --ground-truth-dir data/ground_truth/gt_v3_triangulated_2026-05-15/validated \
  --docs data/test_data.json \
  --docwise-threshold 1.0 \
  --json-report benchmark/results/reference_eval.json \
  --csv-report benchmark/results/reference_eval.csv
```

`--docwise-threshold` opsiyoneldir ve varsayılanı `1.0`dır. Daha toleranslı doküman pass/fail değerlendirmesi için örn. `--docwise-threshold 0.8` kullanılabilir. Bu eşik doküman-level core F1 üzerine uygulanır. `--reference-postprocess legacy` tarihsel Run `019` davranışını korur; `--reference-postprocess canonical_set` Run `020` gibi normalize + compact + full legal identity tuple dedup görünümüyle skor üretir. `--core-reference-view row` varsayılandır ve Run `019`/`020` row-view core matching davranışını korur. `--core-reference-view core_set`, reference postprocess sonrası yalnız core/docwise metrikleri için doküman içinde normalize `(kanun_no, kanun_ad, madde)` identity'lerini gold ve prediction tarafında deduplicate eder; Run `021` bu görünümle üretilmiştir.

JSON çıktısında her method result bloğu şu alanı içerir:

```json
{
  "docwise_core_accuracy": {
    "threshold": 1.0,
    "accuracy": 0.4068825910931174,
    "passed_doc_count": 201,
    "failed_doc_count": 293,
    "total_docs": 494
  }
}
```

`--per-doc` kullanılırsa `docwise_core_per_doc` listesi de yazılır (`doc_id`, `core_f1`, `tp`, `fp`, `fn`, `passed`). CSV çıktısında her yöntem için ayrıca `metric=docwise_core_accuracy` satırı üretilir.

Evaluator hold-out-clean kiyas icin belirli doc id'leri eval disi birakabilir:

```bash
venv/bin/python -m benchmark.reference.evaluate \
  --predictions benchmark/predictions/regex_reference_annotations.json benchmark/predictions/spacy_reference_annotations.json benchmark/predictions/gemini_zero_shot_reference_annotations.json benchmark/predictions/gemini_few_shot_cot_reference_annotations.json benchmark/predictions/gemini_few_shot_cot_v3_en_reference_annotations.json \
  --ground-truth-dir data/annotations \
  --docs data/test_data.json \
  --exclude-doc-ids-file benchmark/reference/example_sets/few_shot_cot_holdout_doc_ids.json \
  --json-report benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_holdout_gt_with_few_shot_cot_v3_en.json \
  --csv-report benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_holdout_gt_with_few_shot_cot_v3_en.csv
```

> **Note (2026-04-29):** Aktif ground truth artik tek statu uretir (`status: "approved"`). Yeni paper-facing kosular icin `--gt-mode approved_only` kullan. `approved_plus_draft` ve `--with-drafts true` backward-compatible amacli korunur, ama mevcut GT uzerinde her iki mode de ayni skoru hesaplar.

`approved_only` mode:

```bash
venv/bin/python -m benchmark.reference.evaluate \
  --predictions benchmark/predictions/langextract_few_shot_cot_v3_en_gemini_2_5_flash_aiplatform_v2_reference_annotations.json \
  --ground-truth-dir data/annotations \
  --docs data/test_data.json \
  --gt-mode approved_only
```

Backward-compatible alias:

```bash
venv/bin/python -m benchmark.reference.evaluate \
  --predictions benchmark/predictions/langextract_few_shot_cot_v3_en_gemini_2_5_flash_aiplatform_v2_reference_annotations.json \
  --ground-truth-dir data/annotations \
  --docs data/test_data.json \
  --with-drafts true
```

Toplu çalıştırma:

```bash
venv/bin/python -m benchmark.reference.run \
  --input data/test_data.json \
  --methods regex spacy llm \
  --evaluate \
  --ground-truth-dir data/annotations \
  --gt-mode approved_only \
  --docwise-threshold 1.0 \
  --llm-provider gemini \
  --llm-model gemini-2.5-flash-lite \
  --llm-prompt-variant zero-shot \
  --json-report benchmark/results/reference_eval.json \
  --csv-report benchmark/results/reference_eval.csv
```

Belirli bir doc id alt kumesi uzerinde benchmark kosmak icin:

```bash
venv/bin/python -m benchmark.reference.run \
  --input data/test_data.json \
  --input-doc-ids-file benchmark/reference/example_sets/prompt_optimization_challenge_doc_ids.json \
  --methods llm \
  --evaluate \
  --ground-truth-dir data/annotations \
  --llm-provider gemini \
  --llm-model gemini-2.5-flash-lite \
  --llm-prompt-variant few-shot-cot-v3-en \
  --llm-temperature 0.1 \
  --llm-run-label temp_0p1
```

Sabit gold subset dosyalari:

- `benchmark/reference/example_sets/gold_annotated_doc_ids.json`
  - tarihsel prompt-optimization gold subset (`139`)
- `benchmark/reference/example_sets/gold_holdout_clean_doc_ids.json`
  - tarihsel few-shot hold-out-clean benchmark evreni (`133`)
- `benchmark/reference/example_sets/prompt_optimization_challenge_doc_ids.json`
  - prompt iterasyonlarinda hizli hata odakli challenge set (`18`)

Bu subset dosyalari tarihsel prompt optimizasyonu icindir; guncel fair_494 leaderboard evreni olarak okunmamalidir.

## Güncel NEW METRIC Artefaktları — 2026-06-30

`020-common500-benchmark-nine-method-gt-v3-cleaned-canonical-set-new-metric/` klasörü, GT v3 cleaned `fair_494` evreninde dokuz yöntemin canonical-set row-view **NEW METRIC performance** sonucunu taşır. `021-common500-benchmark-nine-method-gt-v3-cleaned-core-set-new-metric/` klasörü aynı kaynakları `--core-reference-view core_set` ile skorlayan core-set audit artefaktıdır. `019-common500-benchmark-nine-method-gt-v3-cleaned-new-metric/` ise frozen legacy-view provenance artefaktıdır.

- Evren: `fair_494` = 500 doküman - few-shot exemplar `[1, 10, 16, 18, 36, 77]`
- GT: `data/ground_truth/gt_v3_triangulated_2026-05-15/validated/`
- Prediction sources: `benchmark/predictions/snapshots/2026-05-01_common500/` ve `benchmark/predictions/qwen14b_ft_v1_fair494_reference_annotations.json`
- Metrik: `docwise_core_accuracy@1.0`
- Run `020` row-view NEW METRIC winner: Qwen3.5 397B Cloud, `0.4069` (201/494 doc pass)
- Run `020` row-view primary core-F1 winner aynı koşuda: DeepSeek v3.2 Cloud, F1 `0.8825`
- Run `020` fine-tune: Qwen2.5-14B LoRA v1, `docwise_core_accuracy=0.3482` (172/494), core F1 `0.8254`, core precision `0.8941`
- Run `021` core-set NEW METRIC winner: Qwen3.5 397B Cloud, `0.5182` (256/494 doc pass)
- Run `021` core-set core-F1 winner: DeepSeek v3.2 Cloud, F1 `0.9012`
- Run `021` fine-tune: Qwen2.5-14B LoRA v1, `docwise_core_accuracy=0.4514` (223/494), core-set precision `0.8978`

Artefaktlar:

- canonical-set row-view JSON: `benchmark/results/per_method/020-common500-benchmark-nine-method-gt-v3-cleaned-canonical-set-new-metric/reference_eval_common500_nine_methods_gt_v3_cleaned_canonical_set_NEW_METRIC_docwise_2026-06-27.json`
- canonical-set row-view leaderboard: `benchmark/results/per_method/020-common500-benchmark-nine-method-gt-v3-cleaned-canonical-set-new-metric/leaderboard_NEW_METRIC_docwise_core_accuracy_canonical_set_2026-06-27.csv`
- core-set audit: `benchmark/results/per_method/021-common500-benchmark-nine-method-gt-v3-cleaned-core-set-new-metric/reference_eval_common500_nine_methods_gt_v3_cleaned_core_set_NEW_METRIC_docwise_2026-06-30.json`
- core-set audit leaderboard: `benchmark/results/per_method/021-common500-benchmark-nine-method-gt-v3-cleaned-core-set-new-metric/leaderboard_NEW_METRIC_docwise_core_accuracy_core_set_2026-06-30.csv`
- frozen legacy-view JSON: `benchmark/results/per_method/019-common500-benchmark-nine-method-gt-v3-cleaned-new-metric/reference_eval_common500_nine_methods_gt_v3_cleaned_NEW_METRIC_docwise_2026-06-20.json`
- Journal-generated tables: `Journal/analysis/generated/tables/`

## 2026-03-29 Fair_194 / Common_200 hizali snapshot — **DEPRECATED / HATALI**

> **2026-06-20 not, 2026-06-30 ek ayrim:** Bu snapshot deprecated/hatali kabul edilir. Predictions filtrelenmeden birakildi ama ground truth `approved_only` olarak filtrelendi; sonucta draft gold satirlarina denk gelen dogru tahminler false positive olarak sayildi ve metrikler suni olarak duştu. `fair_194` artefaktlari yalnizca tarihsel kayit olarak korunur, leaderboard veya benchmark referansi olarak kullanilmaz. Aktif `fair_494` ailesinde ASYU paper GT v2 (`015`), GT v3 cleaned primary core-F1 (`017`), frozen legacy-view NEW METRIC (`019`), canonical-set row-view NEW METRIC (`020`) ve core-set NEW METRIC audit (`021`) ayrımı korunmalıdır. `fair_344` ailesi tarihsel provenance'tir.

Yeni gold dosyalari geldikten sonra canonical all-method kiyas icin ayri bir hizali snapshot uretilmistir.

- prediction snapshot evreni:
  - `benchmark/reference/example_sets/common_200_doc_ids_2026-03-29.json`
  - `benchmark/predictions/snapshots/2026-03-29_common200/`
- fair evaluation evreni:
  - `benchmark/reference/example_sets/fair_194_doc_ids_2026-03-29.json`
  - `fair_194 = common_200 - {1,10,16,18,36,77}`
- audit:
  - `benchmark/results/per_method/004-open-source-llm-reference-benchmark/eval_input_audit_2026-03-29.json`
- final aligned evaluation:
  - `benchmark/results/per_method/004-open-source-llm-reference-benchmark/reference_eval_fair194_all_methods_2026-03-29.json`
  - `benchmark/results/per_method/004-open-source-llm-reference-benchmark/reference_eval_fair194_all_methods_2026-03-29.csv`
  - `benchmark/results/per_method/004-open-source-llm-reference-benchmark/reference_eval_fair194_all_methods_error_buckets_2026-03-29.csv`
- current active benchmark evaluation:
  - `benchmark/results/per_method/004-open-source-llm-reference-benchmark/reference_eval_fair194_current_methods_2026-03-30.json`
  - `benchmark/results/per_method/004-open-source-llm-reference-benchmark/reference_eval_fair194_current_methods_2026-03-30.csv`
  - `benchmark/results/per_method/004-open-source-llm-reference-benchmark/reference_eval_fair194_current_methods_error_buckets_2026-03-30.csv`

Bu hizali snapshot'ta few-shot ailesi icindeki kazanan:

- `gemini_few_shot_cot_v3_en_reference_annotations`
  - `core_law_article_strict` F1: `0.6874`
  - `overall_reference` F1: `0.6138`

Yakindan takip eden historical comparator:

- `gemini_few_shot_cot_reference_annotations`
  - `core_law_article_strict` F1: `0.6863`
  - `overall_reference` F1: `0.5899`

## Veri genisleme politikasi

`data/test_data.json` ileride buyurse few-shot ailesi icin resmi refresh kurali su sekilde uygulanmalidir:

- varsayilan resmi few-shot rerun adayi yalniz `gemini_few_shot_cot_v3_en`dir
- `gemini_few_shot_cot` historical comparison artefakti olarak korunur
- `gemini_zero_shot` historical comparison artefakti olarak korunur
- yeni veri genislemelerinde `gemini_few_shot_cot` varsayilan zorunlu rerun hedefi degildir
- yeni veri genislemelerinde `gemini_zero_shot` varsayilan zorunlu rerun hedefi degildir
- all-method evaluation yine ancak ayni prediction evreni uzerinden, audit dosyasi ile dogrulanarak yapilmalidir
- few-shot doc'lari iceren eval evreni ile fair hold-out-clean evreni etiketlenmeden karistirilmamalidir
- guncel yasayan benchmark yorumu icin `fair_494` esas alinmalidir (ASYU/GT v2: `benchmark/results/per_method/015-common500-benchmark-eight-method-newgt-v2/`; GT v3 cleaned primary core-F1: `benchmark/results/per_method/017-common500-benchmark-eight-method-gt-v3-cleaned/`; frozen legacy-view NEW METRIC: `benchmark/results/per_method/019-common500-benchmark-nine-method-gt-v3-cleaned-new-metric/`; canonical-set row-view NEW METRIC: `benchmark/results/per_method/020-common500-benchmark-nine-method-gt-v3-cleaned-canonical-set-new-metric/`; core-set audit NEW METRIC: `benchmark/results/per_method/021-common500-benchmark-nine-method-gt-v3-cleaned-core-set-new-metric/`). `fair_344` ailesi (`012-common344-benchmark-seven-method-single-status-approved/`) ve `fair_194` artefaktlari tarihsel/deprecated olarak isaretlenmistir
- `benchmark/results/per_method/008-common344-benchmark-withDrafts/`, `benchmark/results/per_method/009-common344-benchmark/` ve `benchmark/results/per_method/010-common344-benchmark-mistral-large-3-675b/` tarihsel provenance kayitlaridir; yeni paper-facing kaynak olarak kullanilmamalidir

## Metrikler

Ana benchmark karari artik `core_law_article_strict` uzerinden verilmelidir.

### Primary metric: `core_law_article_strict`

Bu metric fine-tune hedefini merkeze alir: model bir referansi ancak gerekli kanun kimligi ile birlikte `madde`yi dogru kurarsa basarili sayilir.

Gold referanslar normalize edildikten sonra su sekilde siniflanir:

- `triplet_required`
  - `kanun_no + kanun_ad + madde` dolu
  - prediction'da uc alanin da exact dogru olmasi zorunludur
- `no_pair_required`
  - `kanun_no + madde` dolu, `kanun_ad` bos
  - prediction'da `kanun_no + madde` exact dogruysa match olur
  - prediction'daki ekstra `kanun_ad` match'i bozmaz, ama diagnostic olarak raporlanir
- `ad_pair_required`
  - `kanun_ad + madde` dolu, `kanun_no` bos
  - prediction'da `kanun_ad + madde` exact dogruysa match olur
  - prediction'daki ekstra `kanun_no` match'i bozmaz, ama diagnostic olarak raporlanir

Gold'da `madde` bossa:

- `core_law_article_strict` evreninden cikarilir
- `core_exclusion_diagnostics` altinda raporlanir
- legacy `overall_reference` icinde kalmaya devam eder

Prediction tarafinda:

- `madde` bossa `invalid_core_prediction_no_madde_count`
- `madde` dolu ama hem `kanun_no` hem `kanun_ad` bossa `invalid_core_prediction_no_law_count`
- Bu iki durum precision'i dusuren core-FP yukudur

Primary metric ciktisi:

- `Precision`
- `Recall`
- `F1`
- `TP`
- `FP`
- `FN`

### Matching engine: `--matching-engine {greedy,optimal}` (2026-08-03)

Yukarida anlatilan tier merdiveni **`greedy`** motorudur ve varsayilandir. Run
`015`-`021` bu motorla uretilmistir; bit duzeyinde korunur.

**`optimal`** motoru (`benchmark/reference/matching.py`) yeni semantigi uygular.
Run `023`teki dört kabul kapısı (Run `021` greedy eşitliği, 9 yöntem × 30 seed
tam-payload sıra bağımsızlığı, 494 belgede idempotence, optimal altında GT
v3==GT v4) geçtiği için aşağıdaki davranış pipeline için doğrulanmıştır:

| Konu | `greedy` | `optimal` |
|---|---|---|
| Eslestirme | asamali first-fit | min-cost max-flow, **maksimum kardinalite** |
| Satir sirasi | sonucu degistirebilir | Run `023`te tam payload üzerinde end-to-end degismez |
| `madde` bos gold | sessizce atilir | **law-level referans** olarak skorlanir |
| `madde` bos prediction | otomatik FP | law-level gold ile eslesebilir |
| Kanun kimligi | tek yonlu tier + fallback | **simetrik unification** |
| Dedup | `core_set` yalniz madde'li satirlar | madde'siz satirlar dahil + subsumption |
| `13/a` kodlamasi | `madde="13/a"` ile literal | `madde=13 + bent=a` ile denk |
| `Mükerrer 257` | `mükerrer 257`'den farkli | ayni |

Unification kurali: iki satir, **ikisinde de dolu** olan hicbir kanun alaninda
celismiyorsa ve en az bir dolu alanda uyusuyorsa uyumludur. Boylece yalniz
`kanun_ad` veren dogru bir prediction artik FP degildir.

Kanun adi olarak literal `"Kanun"` (normalizer placeholder'i) kimlik sayilmaz.

**Madde kodlama normalizasyonu.** `optimal` motoru karsilastirma oncesi:

- `13/a` -> `madde=13, bent=a` (yalniz **kucuk harf** son ek). Buyuk harf ayri
  maddedir: `32/A`, `38/A`, `39/A`, `14/A`, `3/A`, `3/B` bolunmez -- GT v3'te bu
  satirlarin yarisi kendi `fikra`sini tasir, bent olsalar bu imkansizdir.
- `Mükerrer 257` = `mükerrer 257`, `Ek 1` = `ek 1`, `Geçici 2` = `geçici 2`
- `466-469` = `466 ila 469`

Dedup **madde duzeyindedir** (`core_law_article_strict` tanimi geregi): VUK
`17/4-a` ile `17/1-b` tek core referanstir; bolme sonrasi KDV `13/a` ile `13/d`
de oyle. Bent bilgisi kaybolmaz, `soft_extension_diagnostics` icinde olculur.

Ek rapor blogu: `optimal_matching_diagnostics`.

Tarihsel Run `022`de görülen no-bent `32/A` idempotence ve survivor sıra
duyarlılığı blocker'ları Run `023` öncesi giderildi. Run `022` sadece tarihsel
provisional kayıttır; Run `023`ün tam-payload kabul kapıları bu davranışı
doğrular.

Ornek kullanim:

```bash
python3 -m benchmark.reference.evaluate \
  --predictions benchmark/predictions/<method>.json \
  --ground-truth-dir data/ground_truth/gt_v3_triangulated_2026-05-15/validated \
  --docs data/test_data.json \
  --docwise-threshold 1.0 \
  --reference-postprocess canonical_set \
  --core-reference-view core_set \
  --matching-engine optimal
```

Bu bolumun ustundeki ornek komutlar `--matching-engine` bayragi eklenmeden once
yazilmistir ve varsayilan `greedy` motorunu kullanir. Dokuz yontemli muhurlu
iki-motorlu karsilastirma: `benchmark/results/per_method/023-common500-benchmark-nine-method-dual-engine-sealed/`. Run `022` superseded provisional snapshot'tır ve atıf verilmez.

### Soft extension diagnostics

`fikra` ve `bent` icin artik weighted ikinci skor yoktur.

Bunun yerine yalniz `core_law_article_strict` TP olan eslesmeler icinde su tanisallar uretilir:

- `fikra_exact_when_gold_present_count/rate`
- `fikra_missing_when_gold_present_count/rate`
- `fikra_wrong_nonempty_when_gold_present_count/rate`
- `fikra_extra_when_gold_empty_count/rate`
- `bent_exact_when_gold_present_count/rate`
- `bent_missing_when_gold_present_count/rate`
- `bent_wrong_nonempty_when_gold_present_count/rate`
- `bent_extra_when_gold_empty_count/rate`
- `full_extension_exact_count/rate`

Bu blok JSON raporda `soft_extension_diagnostics` altinda yer alir; leaderboard sirasina girmez.

### Secondary / legacy metrics

- `overall_reference`
  - `(kanun_no, kanun_ad, madde, fikra, bent)` strict tuple match
  - tarihsel uyumluluk ve eski snapshotlarla kiyas icin korunur
- Entity bazli (reference-aligned): `kanun_no`, `kanun_ad`, `madde`, `fikra`, `bent`
  - Per-entity TP/FP/FN matched/unmatched tuple'lardaki non-empty field'lardan turetilir
  - Bos alanlar (orn. `fikra=""`) per-entity sayimina dahil edilmez

Onemli not:

- Evaluator ham metinden anafora cozmez.
- `bu kanun`, `mezkur kanun`, `ayni kanun` gibi atiflar prediction/gold JSON icinde zaten resolve edilmis alanlar olarak gelmelidir.
- Evaluator yalniz normalize edilmis `kanun_no / kanun_ad / madde / fikra / bent` alanlarini karsilastirir.

### Granüler Hata Analizi (Error Buckets)

Unmatched referanslar greedy best-match pairing ile eşleştirilir ve şu bucket tiplerine sınıflandırılır:

| Bucket | Açıklama |
|--------|----------|
| `partial_4of5` | 5 field'dan 4'ü doğru (near-miss) |
| `partial_3of5` | 5 field'dan 3'ü doğru |
| `wrong_subfields` | Kanun doğru (no+ad) ama madde/fıkra/bent hatalı |
| `wrong_law` | Kanun eşleşmiyor |
| `spurious` | Gold'da hiç karşılığı yok |
| `missed` | Prediction'da hiç karşılığı yok |

Her bucket FP ve FN tarafı olarak ayrı raporlanır (ör. `fp_partial_4of5`, `fn_missed`).
`wrong_fields` listesi hangi alanların hatalı olduğunu gösterir.

### Partial Match İstatistikleri

- `total_unmatched_pairs`: Greedy eşleştirilen pair sayısı
- `avg_overlap_score`: Ortalama overlap skoru (0-5)
- `overlap_distribution`: Kaç pair hangi skorda
- `near_miss_count`: 4+ overlap'li (neredeyse doğru) pair sayısı

### Document-Level Diagnostics

- Belge bazında F1 dağılımı: min, max, mean, median, p25, p75
- Perfect-F1 ve Zero-F1 belge sayıları
- En kötü 10 belgenin detaylı metrik listesi
- `--per-doc` flag ile belge başına tam metrik JSON çıktısı

## Sağlamlık Kontrolleri

- Prediction dosyalarındaki `status` dağılımı raporlanır.
- `status=error` olan `doc_id` listesi görünür şekilde yazdırılır.
- Duplicate `doc_id` tespiti yapılır.
- Missing prediction ve missing ground truth dokümanları raporlanır.

## Yeni Karsilastirma ve Hard-Case Opsiyonlari

Orchestrator artik baseline delta ve hard-case raporlarini tek kosuda uretebilir:

```bash
venv/bin/python -m benchmark.reference.run \
  --input data/test_data.json \
  --methods regex spacy \
  --evaluate \
  --ground-truth-dir data/annotations \
  --baseline-report benchmark/results/per_method/001-regex-spacy-accuracy/reference_eval_baseline.json \
  --doc-ids-file benchmark/results/per_method/001-regex-spacy-accuracy/hard_case_doc_ids.json \
  --emit-default-reports \
  --feature-tag 001-regex-spacy-accuracy
```

Bu kosu ile su artefaktlar otomatik uretilir:
- `reference_eval_improved.json/.csv`
- `error_buckets.csv`
- `hard_case_eval.json/.csv`
- `hard_case_error_buckets.csv`
- `llm_run_summary.json` (`llm` seciliyse)

## 003-llm-reference-benchmark

Frozen pre-LLM baseline snapshot:

```bash
venv/bin/python -m benchmark.reference.run \
  --input data/test_data.json \
  --methods regex spacy \
  --evaluate \
  --ground-truth-dir data/annotations \
  --json-report benchmark/results/per_method/003-llm-reference-benchmark/frozen_pre_llm_reference_eval.json \
  --csv-report benchmark/results/per_method/003-llm-reference-benchmark/frozen_pre_llm_reference_eval.csv \
  --error-bucket-csv benchmark/results/per_method/003-llm-reference-benchmark/frozen_pre_llm_error_buckets.csv
```

Gemini zero-shot benchmark snapshot:

```bash
GEMINI_API_KEY=... venv/bin/python -m benchmark.reference.run \
  --input data/test_data.json \
  --methods llm \
  --evaluate \
  --ground-truth-dir data/annotations \
  --llm-provider gemini \
  --llm-model gemini-2.5-flash-lite \
  --llm-prompt-variant zero-shot \
  --json-report benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_with_llm.json \
  --csv-report benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_with_llm.csv \
  --error-bucket-csv benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_with_llm_error_buckets.csv \
  --llm-run-summary benchmark/results/per_method/003-llm-reference-benchmark/llm_run_summary.json
```

Gemini few-shot CoT benchmark snapshot:

```bash
GEMINI_API_KEY=... venv/bin/python -m benchmark.reference.run \
  --input data/test_data.json \
  --methods llm \
  --evaluate \
  --ground-truth-dir data/annotations \
  --llm-provider gemini \
  --llm-model gemini-2.5-flash-lite \
  --llm-prompt-variant few-shot-cot \
  --exclude-doc-ids-file benchmark/reference/example_sets/few_shot_cot_holdout_doc_ids.json \
  --json-report benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_with_llm_few_shot_cot.json \
  --csv-report benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_with_llm_few_shot_cot.csv \
  --error-bucket-csv benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_with_llm_few_shot_cot_error_buckets.csv \
  --llm-run-summary benchmark/results/per_method/003-llm-reference-benchmark/llm_run_summary_few_shot_cot.json
```

Archived historical experiment: Gemini few-shot CoT v2 subset temperature sweep

Not:

- `few-shot-cot-v2` aktif makale omurgasinin parcasi degildir.
- Bu varyanta ait mevcut artefaktlar `artifacts/archive/benchmark_cleanup_2026-03-18/` altina tasinmistir.
- Asagidaki komutlar yalniz reproducibility amaciyla tutulur; tekrar kosulursa ayni dosya aileleri yeniden uretilebilir.

Komut:

```bash
GEMINI_API_KEY=... venv/bin/python -m benchmark.reference.run \
  --input data/test_data.json \
  --input-doc-ids-file benchmark/reference/example_sets/few_shot_cot_v2_sweep_subset_doc_ids.json \
  --methods llm \
  --evaluate \
  --ground-truth-dir data/annotations \
  --llm-provider gemini \
  --llm-model gemini-2.5-flash-lite \
  --llm-prompt-variant few-shot-cot-v2 \
  --llm-temperature 0.0 \
  --llm-run-label temp_0p0 \
  --json-report artifacts/benchmark/llm_sweeps/few_shot_cot_v2/reference_eval_temp_0p0.json \
  --csv-report artifacts/benchmark/llm_sweeps/few_shot_cot_v2/reference_eval_temp_0p0.csv \
  --error-bucket-csv artifacts/benchmark/llm_sweeps/few_shot_cot_v2/reference_eval_temp_0p0_error_buckets.csv \
  --llm-run-summary artifacts/benchmark/llm_sweeps/few_shot_cot_v2/llm_run_summary_temp_0p0.json
```

Ayni komut `--llm-temperature 0.1 --llm-run-label temp_0p1` ve `--llm-temperature 0.2 --llm-run-label temp_0p2` icin tekrar edilir.

## English Prompt Optimization Workflow

Amaç:

- `200` belgelik tam kosu yerine yalniz gold evreninde prompt iterasyonu yapmak
- ana KPI olarak `133` belgeli hold-out-clean F1'i optimize etmek
- mevcut `few-shot-cot` ve `few-shot-cot-v2` snapshotlarini bozmadan yeni English control prompt ailesini denemek

Prompt varyantlari:

- `en-baseline`
- `en-negative-guarded`
- `en-recall-recovery`
- `few-shot-cot-v3-en`

Not:

- Bu ailede prompt dili İngilizcedir, ancak model Türkçe hukuk metnini okur.
- `kanun_ad`, `source_text` ve orijinal yüzey biçimleri çevrilmez.
- Model sabit tutulur: `gemini-2.5-flash-lite`
- Temperature sabit tutulur: `0.0`

Challenge set üzerinde hizli iterasyon:

```bash
GEMINI_API_KEY=... venv/bin/python -m benchmark.reference.run \
  --input data/test_data.json \
  --input-doc-ids-file benchmark/reference/example_sets/prompt_optimization_challenge_doc_ids.json \
  --methods llm \
  --evaluate \
  --ground-truth-dir data/annotations \
  --llm-provider gemini \
  --llm-model gemini-2.5-flash-lite \
  --llm-prompt-variant en-baseline \
  --llm-temperature 0.0 \
  --json-report artifacts/benchmark/prompt_optimization/few_shot_cot_v3_en/en_baseline/challenge_eval.json \
  --csv-report artifacts/benchmark/prompt_optimization/few_shot_cot_v3_en/en_baseline/challenge_eval.csv \
  --error-bucket-csv artifacts/benchmark/prompt_optimization/few_shot_cot_v3_en/en_baseline/challenge_error_buckets.csv \
  --llm-run-summary artifacts/benchmark/prompt_optimization/few_shot_cot_v3_en/en_baseline/challenge_run_summary.json
```

Ayni komut sirasiyla su varyantlar icin tekrar edilir:

- `en-negative-guarded`
- `en-recall-recovery`
- `few-shot-cot-v3-en`

`139` gold belge uzerinde diagnostic kosu:

```bash
GEMINI_API_KEY=... venv/bin/python -m benchmark.reference.run \
  --input data/test_data.json \
  --input-doc-ids-file benchmark/reference/example_sets/gold_annotated_doc_ids.json \
  --methods llm \
  --evaluate \
  --ground-truth-dir data/annotations \
  --llm-provider gemini \
  --llm-model gemini-2.5-flash-lite \
  --llm-prompt-variant few-shot-cot-v3-en \
  --llm-temperature 0.0 \
  --json-report artifacts/benchmark/prompt_optimization/few_shot_cot_v3_en/final_candidate/gold139_eval.json \
  --csv-report artifacts/benchmark/prompt_optimization/few_shot_cot_v3_en/final_candidate/gold139_eval.csv \
  --error-bucket-csv artifacts/benchmark/prompt_optimization/few_shot_cot_v3_en/final_candidate/gold139_error_buckets.csv \
  --llm-run-summary artifacts/benchmark/prompt_optimization/few_shot_cot_v3_en/final_candidate/gold139_run_summary.json
```

`133` hold-out-clean official comparison kosusu:

```bash
GEMINI_API_KEY=... venv/bin/python -m benchmark.reference.run \
  --input data/test_data.json \
  --input-doc-ids-file benchmark/reference/example_sets/gold_holdout_clean_doc_ids.json \
  --methods llm \
  --evaluate \
  --ground-truth-dir data/annotations \
  --llm-provider gemini \
  --llm-model gemini-2.5-flash-lite \
  --llm-prompt-variant few-shot-cot-v3-en \
  --llm-temperature 0.0 \
  --json-report artifacts/benchmark/prompt_optimization/few_shot_cot_v3_en/final_candidate/holdout133_eval.json \
  --csv-report artifacts/benchmark/prompt_optimization/few_shot_cot_v3_en/final_candidate/holdout133_eval.csv \
  --error-bucket-csv artifacts/benchmark/prompt_optimization/few_shot_cot_v3_en/final_candidate/holdout133_error_buckets.csv \
  --llm-run-summary artifacts/benchmark/prompt_optimization/few_shot_cot_v3_en/final_candidate/holdout133_run_summary.json
```

Kazanan aday resmi benchmarka tasinacaksa canonical full comparison:

```bash
venv/bin/python -m benchmark.reference.evaluate \
  --predictions benchmark/predictions/regex_reference_annotations.json benchmark/predictions/spacy_reference_annotations.json benchmark/predictions/gemini_zero_shot_reference_annotations.json benchmark/predictions/gemini_few_shot_cot_reference_annotations.json benchmark/predictions/gemini_few_shot_cot_v3_en_reference_annotations.json \
  --ground-truth-dir data/annotations \
  --docs data/test_data.json \
  --doc-ids-file benchmark/reference/example_sets/gold_holdout_clean_doc_ids.json \
  --json-report benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_holdout_gt_with_few_shot_cot_v3_en.json \
  --csv-report benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_holdout_gt_with_few_shot_cot_v3_en.csv \
  --error-bucket-csv benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_holdout_gt_with_few_shot_cot_v3_en_error_buckets.csv
```

Canonicalizasyon kosulu:

1. `holdout133 overall_reference F1 > 0.669882`
2. `bent F1 >= 0.627737`
3. yeni sistematik `kanun_ad` surface drift gorulmuyor

Temperature secim kurali:

1. en yuksek `overall_reference` F1
2. esitlikte en yuksek `bent` F1
3. yine esitlikte daha dusuk `FP`
4. yine esitlikte daha dusuk temperature

Secim yardimci komutu:

```bash
python scripts/benchmark/select_best_llm_temperature.py \
  --sweep-dir artifacts/benchmark/llm_sweeps/few_shot_cot_v2 \
  --output artifacts/benchmark/llm_sweeps/few_shot_cot_v2/temperature_selection_summary.json
```

Kazanan temperature ile canonical full `few-shot-cot-v2` benchmark snapshot:

```bash
GEMINI_API_KEY=... venv/bin/python -m benchmark.reference.run \
  --input data/test_data.json \
  --methods llm \
  --evaluate \
  --ground-truth-dir data/annotations \
  --llm-provider gemini \
  --llm-model gemini-2.5-flash-lite \
  --llm-prompt-variant few-shot-cot-v2 \
  --llm-temperature <winner_temperature> \
  --json-report benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_with_llm_few_shot_cot_v2.json \
  --csv-report benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_with_llm_few_shot_cot_v2.csv \
  --error-bucket-csv benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_with_llm_few_shot_cot_v2_error_buckets.csv \
  --llm-run-summary benchmark/results/per_method/003-llm-reference-benchmark/llm_run_summary_few_shot_cot_v2.json
```

Canonical v2 hold-out-clean leaderboard:

```bash
venv/bin/python -m benchmark.reference.evaluate \
  --predictions benchmark/predictions/regex_reference_annotations.json benchmark/predictions/spacy_reference_annotations.json benchmark/predictions/gemini_zero_shot_reference_annotations.json benchmark/predictions/gemini_few_shot_cot_reference_annotations.json benchmark/predictions/gemini_few_shot_cot_v2_reference_annotations.json \
  --ground-truth-dir data/annotations \
  --docs data/test_data.json \
  --exclude-doc-ids-file benchmark/reference/example_sets/few_shot_cot_holdout_doc_ids.json \
  --json-report benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_holdout_gt_with_few_shot_cot_v2.json \
  --csv-report benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_holdout_gt_with_few_shot_cot_v2.csv \
  --error-bucket-csv benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_holdout_gt_with_few_shot_cot_v2_error_buckets.csv
```

Not:

- Yukaridaki `few-shot-cot-v2` ciktilarinin mevcut kopyalari arsivlenmistir.
- Aktif makale hatti icin esas alinmasi gereken few-shot aileleri `few-shot-cot` ve `few-shot-cot-v3-en`'dir.

Adil hold-out-clean leaderboard:

```bash
venv/bin/python -m benchmark.reference.evaluate \
  --predictions benchmark/predictions/regex_reference_annotations.json benchmark/predictions/spacy_reference_annotations.json benchmark/predictions/gemini_zero_shot_reference_annotations.json benchmark/predictions/gemini_few_shot_cot_reference_annotations.json \
  --ground-truth-dir data/annotations \
  --docs data/test_data.json \
  --exclude-doc-ids-file benchmark/reference/example_sets/few_shot_cot_holdout_doc_ids.json \
  --json-report benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_holdout_gt_with_few_shot_cot.json \
  --csv-report benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_holdout_gt_with_few_shot_cot.csv \
  --error-bucket-csv benchmark/results/per_method/003-llm-reference-benchmark/reference_eval_holdout_gt_with_few_shot_cot_error_buckets.csv
```

Not:
- `zero-shot` current-GT leaderboard'i ayridir ve overwrite edilmemelidir.
- `few-shot-cot` varyanti held-out example doc id'leri (`1, 10, 16, 18, 36, 77`) eval disi birakilarak raporlanmalidir.
- `few-shot-cot-v2` tarihsel/arsivlenmis varyanttir; fairness kurali aynidir ama aktif benchmark omurgasina dahil edilmez.
- Prediction dosya adi varyanta gore otomatik belirlenir:
  - `gemini_zero_shot_reference_annotations.json`
  - `gemini_few_shot_cot_reference_annotations.json`
  - `gemini_few_shot_cot_v3_en_reference_annotations.json`
- Tarihsel/arsivlenmis varyantlar yeniden uretilirse:
  - `gemini_few_shot_cot_v2_reference_annotations.json`
  - deneyselse `gemini_few_shot_cot_v2_temp_0p1_reference_annotations.json` gibi `run_label` suffix'i alabilir

### 180K Olcek Icin spaCy Parametreleri

`run` komutu artik spaCy isleme parametrelerini disaridan alir:

```bash
venv/bin/python -m benchmark.reference.run \
  --input data/test_data.json \
  --methods spacy \
  --spacy-batch-size 512 \
  --spacy-n-process 2
```

Buyuk veri hacminde once bu iki parametre icin throughput sweep yapip sabit bir profil secin.

## 002-spacy-spanruler-resolver

spaCy extractor icin candidate+resolver tabanli yeni akis:

```bash
# Baseline
venv/bin/python -m benchmark.reference.run \
  --input data/test_data.json \
  --methods spacy \
  --spacy-batch-size 256 \
  --spacy-n-process 1 \
  --evaluate \
  --ground-truth-dir data/annotations \
  --json-report benchmark/results/per_method/002-spacy-spanruler-resolver/spacy_baseline.json \
  --csv-report benchmark/results/per_method/002-spacy-spanruler-resolver/spacy_baseline.csv

# Improved (delta)
venv/bin/python -m benchmark.reference.run \
  --input data/test_data.json \
  --methods spacy \
  --spacy-batch-size 256 \
  --spacy-n-process 1 \
  --evaluate \
  --ground-truth-dir data/annotations \
  --baseline-report benchmark/results/per_method/002-spacy-spanruler-resolver/spacy_baseline.json \
  --json-report benchmark/results/per_method/002-spacy-spanruler-resolver/spacy_improved.json \
  --csv-report benchmark/results/per_method/002-spacy-spanruler-resolver/spacy_improved.csv
```
