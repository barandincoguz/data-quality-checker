# Tamamlanmış plan ve tasarım kayıtları

Bu depo bir süre her iş için ayrı bir plan dosyası tuttu. Hepsi tamamlandı, ve
tamamlanmış bir plan dosyası artık yalnızca gürültü — ne olduğunu anlatır ama
nerede durduğunu anlatmaz. Dosyalar silindi; her birinin ne yaptığı ve işin
hangi kodda yaşadığı aşağıda. Tam metinleri git geçmişinde duruyor.

Silme commit'i: bu dosyanın eklendiği commit.

## Bu depoda yaşayan işler

| Plan | İşin yaşadığı yer |
| --- | --- |
| `2026-09-02-dq-loop-split-and-training` | `loop_split.py`, `loop_batches.py`, `loop_selection.py` |
| `2026-09-02-gemini-judge-provider` | `judges.py` — `GeminiJudgeProvider` ve sağlayıcı kaydı |
| `2026-09-03-dq-loop-bindings` | `loop_bindings.py` — `predict_step` / `route_step` / `judge_step` |
| `2026-09-03-dq-loop-round-state-machine` | `loop_rounds.py` — `ROUND_STAGES`, `advance_round` |
| `2026-09-03-dq-loop-round-training` | `loop_train.py` — `train_round` |
| `2026-09-03-dq-loop-runner` | `loop_runner.py` (CLI `loop-status` oldu, `loop-run` değil) |
| `2026-09-03-dq-loop-wiring-unblock` | `cli.py` — `--generation` zorunlu (commit `16fc992`) |
| `2026-09-03-local-gemma-judge` | `judges.py` — `MlxJudgeProvider`, pilotun varsayılan hakemi |
| `2026-09-04-loop-metrics` | `loop_metrics.py` — `round_metrics`, kappa fonksiyonları |
| `2026-09-05-reference-normalization-defects` | `normalization.py` — `_fold`, ordinal ve madde/bent ayrımı (commit `f2bb4e3`) |
| `2026-09-07-source-text-off-model-surfaces` | 9/9 görev, commit `11b766d`..`dbddc5d`; yeni `reference_locator.py` |

## Aşılmış iş

| Plan | Ne oldu |
| --- | --- |
| `2026-09-06-m0-baseline-and-source-text-removal` | M₀ (v10 @ 325) kuruldu, sonra tur 1 geri sarmasıyla terk edildi (commit `a82c141`). Yerine M₀′ (v11 @ 425). |

## Başka depoda yaşayan işler

Aşağıdakilerin kodu ve sonuçları `ner-project` deposunda; burada yalnız plan
dosyaları duruyordu ve bu depoda hiçbir izleri yok.

| Plan / tasarım | Nerede |
| --- | --- |
| `2026-07-25-cot-enumeration-recall-probe` (+ tasarımı) | `ner-project` — `Fine-Tuning/experiments/2026-07-25_cot_enum_probe/` |
| `2026-07-27-two-call-cot-recall-probe` (+ tasarımı) | `ner-project` — `arms/two_call_cot`, `run_two_call.py` |

## Tutulan tasarım belgesi

`specs/2026-07-24-hitl-review-ux-design.md` **silinmedi**, çünkü uygulanmamış
bir tasarımı anlatıyor ve bunu bilmek gerekiyor. Belgenin başına bir uyarı
eklendi: tarif ettiği körlenmiş A/B protokolü kurulmadı.
