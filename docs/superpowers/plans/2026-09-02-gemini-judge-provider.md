# Gemini Judge Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Gemini as a selectable LLM-as-judge alongside the two existing Ollama-cloud judges, so the DQ-Loop can use it as the third opinion in each round.

**Architecture:** `judges.py` already defines a `JudgeProvider` protocol with a single `judge(*, model, payload) -> (raw, meta)` method, and two implementations (`FakeJudgeProvider`, `OllamaJudgeProvider`). We add a third implementation that calls Google's `generateContent` REST endpoint with stdlib `urllib`, plus a model→provider registry and a factory so the pilot and green-audit paths stop hardcoding `OllamaJudgeProvider()`. The strict output contract in `_validate_judge_result` is unchanged — Gemini output must satisfy exactly the same gates.

**Tech Stack:** Python 3.11, stdlib only (`urllib.request`, `json`, `os`, `time`), pytest 9, ruff, mypy.

## Global Constraints

- **No new runtime dependency.** `requirements.txt` is `Flask>=3.1,<4` and stays that way. HTTP is stdlib `urllib.request`, matching `OllamaJudgeProvider`.
- **Fail closed.** A missing API key raises `JudgeProviderUnavailable`, never a silent skip. Network/parse failures raise `JudgeProviderUnavailable`.
- **The judge is not an authority.** Its output is a third opinion; it never defines truth. No code path may promote a judge verdict to a final decision without expert review.
- **Output contract is shared.** Gemini results pass through the existing `_validate_judge_result(payload, document_text)` unchanged: verdict in `{A, B, TIE, NEITHER}`, `candidate_errors` with `A` and `B` lists, `final_references` validated by `validate_reference_list` and filtered by `apply_reference_policy`, every `source_text` and every `evidence` span must occur in the document under `evidence_match_mode`.
- **External calls stay behind consent.** `run_judge_pilot` and the green-audit path already refuse to run without `allow_external_judge=True`; Gemini inherits this.
- **Reference policy default is `ignore_vuk_213_article_413_v1`** (`reference_policy.py:12`) and is applied inside the validator. Do not add a second filter.
- **Model id is configuration, not a constant in code paths.** `GEMINI_JUDGE_MODEL` env var, default `"gemini-3.1-pro"`. The default string is unverified against the live API — Task 1 Step 7 verifies it before the model is locked for production use.
- Formatting: `ruff format` and `ruff check` must pass. Type checking: `mypy src` must introduce **no new** errors — the tree carries 37 pre-existing ones on `main`; compare counts, do not require a clean run.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/data_quality_checker/judges.py` (modify) | Add `GeminiJudgeProvider`, `judge_model_providers()`, `resolve_judge_provider`; replace two hardcoded `OllamaJudgeProvider()` sites; widen `lock_judge` validation |
| `src/data_quality_checker/cli.py` (modify) | Add `--judge-models` to `pilot-judges` |
| `src/data_quality_checker/commands.py` (modify) | Pass `judge_models` through to `run_judge_pilot` |
| `tests/test_gemini_judge_provider.py` (create) | Unit tests for the provider: request shape, response parsing, failure modes |
| `tests/test_judge_provider_registry.py` (create) | Unit tests for the registry and factory |
| `tests/test_judges.py` (modify) | Extend: `lock_judge` accepts the Gemini model; pilot honours a model-set override |
| `docs/CLI_REFERENCE.md` (modify) | Document `--judge-models` and the Gemini env vars |

---

### Task 1: `GeminiJudgeProvider`

**Files:**
- Modify: `src/data_quality_checker/judges.py` (add after `OllamaJudgeProvider`, which ends around line 245)
- Test: `tests/test_gemini_judge_provider.py`

**Interfaces:**
- Consumes: `JudgeProvider` protocol (`judges.py:39`), `JudgeProviderUnavailable` (`judges.py:33`).
- Produces: `GeminiJudgeProvider` class with `__init__(self) -> None` and `judge(self, *, model: str, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]`; module constants `DEFAULT_GEMINI_JUDGE_MODEL: str`, `JUDGE_PROMPT: str`, and the accessor `gemini_judge_model() -> str` (the accessor shape lands via Task 1's review fix).

- [ ] **Step 1: Write the failing test file**

Create `tests/test_gemini_judge_provider.py`:

```python
from __future__ import annotations

import io
import json
from typing import Any

import pytest

from data_quality_checker.judges import GeminiJudgeProvider, JudgeProviderUnavailable


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _gemini_body(text: str) -> bytes:
    return json.dumps(
        {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    ).encode("utf-8")


def test_judge_returns_model_text_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(_gemini_body('{"verdict":"A"}'))

    monkeypatch.setattr("data_quality_checker.judges.urllib.request.urlopen", fake_urlopen)

    provider = GeminiJudgeProvider()
    raw, meta = provider.judge(model="gemini-3.1-pro", payload={"document": "metin"})

    assert raw == '{"verdict":"A"}'
    assert meta["provider"] == "gemini"
    assert meta["latency_seconds"] >= 0.0
    assert "gemini-3.1-pro:generateContent" in captured["url"]
    assert captured["body"]["generationConfig"]["temperature"] == 0
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "metin" in captured["body"]["contents"][0]["parts"][0]["text"]


def test_missing_api_key_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(JudgeProviderUnavailable):
        GeminiJudgeProvider()


def test_network_failure_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def boom(request: Any, timeout: float | None = None) -> None:
        raise TimeoutError("timed out")

    monkeypatch.setattr("data_quality_checker.judges.urllib.request.urlopen", boom)
    provider = GeminiJudgeProvider()
    with pytest.raises(JudgeProviderUnavailable):
        provider.judge(model="gemini-3.1-pro", payload={"document": "metin"})


def test_empty_candidates_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        return _FakeResponse(json.dumps({"candidates": []}).encode("utf-8"))

    monkeypatch.setattr("data_quality_checker.judges.urllib.request.urlopen", fake_urlopen)
    provider = GeminiJudgeProvider()
    with pytest.raises(JudgeProviderUnavailable):
        provider.judge(model="gemini-3.1-pro", payload={"document": "metin"})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gemini_judge_provider.py -q`
Expected: FAIL — `ImportError: cannot import name 'GeminiJudgeProvider' from 'data_quality_checker.judges'`

- [ ] **Step 3: Extract the shared judge prompt**

In `judges.py`, `OllamaJudgeProvider.judge` builds its prompt inline. Lift it to a module constant so both providers use the identical wording — a different prompt per provider would make the pilot comparison invalid.

Add near the top of `judges.py`, directly after `JUDGE_MODELS`:

```python
JUDGE_PROMPT = (
    "Act as a blind legal-reference adjudicator. Compare candidate A and B "
    "only against the Turkish document. Return JSON with verdict (A, B, TIE, "
    "or NEITHER), candidate_errors {A:[],B:[]}, final_references, evidence, "
    "and reason_codes. Every evidence span must occur in the document.\n\n"
)
```

Then in `OllamaJudgeProvider.judge`, replace the inline literal so it reads:

```python
        prompt = JUDGE_PROMPT + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
```

- [ ] **Step 4: Run the existing judge tests to confirm the extraction changed nothing**

Run: `.venv/bin/python -m pytest tests/test_judges.py -q`
Expected: PASS, same count as before the edit.

- [ ] **Step 5: Implement `GeminiJudgeProvider`**

Add after `OllamaJudgeProvider` in `judges.py`:

```python
DEFAULT_GEMINI_JUDGE_MODEL = "gemini-3.1-pro"


def gemini_judge_model() -> str:
    """Resolve the Gemini judge model id at call time.

    Read on every call rather than bound at import, so a process that sets
    GEMINI_JUDGE_MODEL after this module loads still gets the configured id.
    """
    return os.environ.get("GEMINI_JUDGE_MODEL", DEFAULT_GEMINI_JUDGE_MODEL)


class GeminiJudgeProvider:
    """Google Generative AI adjudicator, stdlib HTTP only.

    Mirrors OllamaJudgeProvider: same prompt, same return shape, same
    fail-closed behaviour. The response is handed back as raw text so
    _validate_judge_result applies the identical contract to every provider.
    """

    def __init__(self) -> None:
        self.base_url = os.environ.get(
            "GEMINI_BASE_URL",
            "https://aiplatform.googleapis.com/v1/publishers/google/models",
        ).rstrip("/")
        self.timeout = float(os.environ.get("GEMINI_TIMEOUT", "500"))
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise JudgeProviderUnavailable("GEMINI_API_KEY is unavailable")
        self.api_key = key

    def judge(self, *, model: str, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        prompt = JUDGE_PROMPT + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        request_payload = json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/{model}:generateContent",
            data=request_payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise JudgeProviderUnavailable(str(exc)) from exc
        candidates = body.get("candidates") or []
        if not candidates:
            raise JudgeProviderUnavailable("gemini returned no candidates")
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        content = "".join(str(part.get("text", "")) for part in parts)
        if not content:
            raise JudgeProviderUnavailable("gemini returned an empty candidate")
        usage = body.get("usageMetadata") or {}
        return content, {
            "latency_seconds": time.perf_counter() - started,
            "cost": None,
            "provider": "gemini",
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
        }
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gemini_judge_provider.py -q`
Expected: PASS — 4 passed.

- [ ] **Step 7: Verify the model id against the live API (manual, one command)**

The default `gemini-3.1-pro` is a guess and must be confirmed before anything is locked. Run:

```bash
curl -s -H "Authorization: Bearer $GEMINI_API_KEY" \
  "https://aiplatform.googleapis.com/v1/publishers/google/models" | head -c 2000
```

Expected: a JSON list containing the exact model id. If `gemini-3.1-pro` is absent, set the real id in `.env` as `GEMINI_JUDGE_MODEL=<real-id>` and record the id in `Journal/experiments/dq_loop/RUNBOOK.md`. **Do not** edit the default in code to an unverified string.

- [ ] **Step 8: Lint, type-check, commit**

```bash
cd /Users/student2/data-quality-checker
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src
git add src/data_quality_checker/judges.py tests/test_gemini_judge_provider.py
git commit -m "feat(judges): add Gemini judge provider behind the shared prompt"
```

---

### Task 2: Model→provider registry and factory

**Files:**
- Modify: `src/data_quality_checker/judges.py` (`JUDGE_MODELS` at line 30; add registry beneath it)
- Test: `tests/test_judge_provider_registry.py`

**Interfaces:**
- Consumes: `FakeJudgeProvider`, `OllamaJudgeProvider`, `GeminiJudgeProvider` from Task 1.
- Produces: `_STATIC_JUDGE_MODEL_PROVIDERS: dict[str, str]`; `judge_model_providers() -> dict[str, str]` resolved at call time; `resolve_judge_provider(model: str, *, fake_backend: bool = False) -> JudgeProvider`; `ContractError` for an unknown model.

- [ ] **Step 1: Write the failing test**

Create `tests/test_judge_provider_registry.py`:

```python
from __future__ import annotations

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.judges import (
    gemini_judge_model,
    judge_model_providers,
    FakeJudgeProvider,
    GeminiJudgeProvider,
    OllamaJudgeProvider,
    resolve_judge_provider,
)


def test_registry_covers_every_pilot_model() -> None:
    from data_quality_checker.judges import JUDGE_MODELS

    for model in JUDGE_MODELS:
        assert model in judge_model_providers()


def test_gemini_model_is_registered() -> None:
    assert judge_model_providers()[gemini_judge_model()] == "gemini"


def test_fake_backend_short_circuits_every_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = resolve_judge_provider(gemini_judge_model(), fake_backend=True)
    assert isinstance(provider, FakeJudgeProvider)


def test_resolves_ollama_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    provider = resolve_judge_provider("qwen3.5:397b")
    assert isinstance(provider, OllamaJudgeProvider)


def test_resolves_gemini_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = resolve_judge_provider(gemini_judge_model())
    assert isinstance(provider, GeminiJudgeProvider)


def test_unknown_model_is_a_contract_error() -> None:
    with pytest.raises(ContractError):
        resolve_judge_provider("not-a-registered-model")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_judge_provider_registry.py -q`
Expected: FAIL — `ImportError: cannot import name 'JUDGE_MODEL_PROVIDERS'`

- [ ] **Step 3: Implement the registry and factory**

In `judges.py`, immediately after the `JUDGE_MODELS` line (currently line 30), add the static half of the registry:

```python
_STATIC_JUDGE_MODEL_PROVIDERS: dict[str, str] = {
    "qwen3.5:397b": "ollama",
    "deepseek-v3.2": "ollama",
}
```

Then, after `gemini_judge_model()` (added in Task 1's fix), add the accessor and
the factory:

```python
def judge_model_providers() -> dict[str, str]:
    """Model id to provider kind, resolved at call time.

    The Gemini entry is keyed on gemini_judge_model(), which reads the
    environment on every call, so the registry follows configuration rather
    than whatever the environment held at import.
    """
    return {**_STATIC_JUDGE_MODEL_PROVIDERS, gemini_judge_model(): "gemini"}


def resolve_judge_provider(model: str, *, fake_backend: bool = False) -> JudgeProvider:
    """Return the provider that serves `model`.

    `fake_backend` short-circuits before any credential is read, so tests and
    dry runs never touch the network or require a key.
    """
    providers = judge_model_providers()
    kind = providers.get(model)
    if kind is None:
        raise ContractError(f"unknown judge model {model!r}; expected one of {sorted(providers)}")
    if fake_backend:
        return FakeJudgeProvider()
    if kind == "gemini":
        return GeminiJudgeProvider()
    return OllamaJudgeProvider()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_judge_provider_registry.py -q`
Expected: PASS — 6 passed.

- [ ] **Step 5: Commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
git add src/data_quality_checker/judges.py tests/test_judge_provider_registry.py
git commit -m "feat(judges): resolve judge provider per model via a registry"
```

---

### Task 3: `--judge-models` override through the pilot

**Files:**
- Modify: `src/data_quality_checker/judges.py` (`run_judge_pilot` and `_run_pilot_impl` signatures; every `JUDGE_MODELS` reference inside `_run_pilot_impl`)
- Modify: `src/data_quality_checker/cli.py` (the `pilot-judges` parser, around line 141)
- Modify: `src/data_quality_checker/commands.py` (`pilot_judges` handler)
- Modify: `docs/CLI_REFERENCE.md`
- Test: `tests/test_judges.py`

**Interfaces:**
- Consumes: `judge_model_providers()` from Task 2.
- Produces: `run_judge_pilot(..., judge_models: tuple[str, ...] | None = None)` and the same keyword on `_run_pilot_impl`. `None` means `JUDGE_MODELS`, so existing callers are unchanged. The pilot summary's `models` mapping is keyed by whatever set actually ran — Task 4 depends on this.

> **Why this comes before provider routing:** `lock_judge` can only lock a model
> that actually ran in the pilot, so the model-set override has to exist before
> a Gemini lock can be tested at all.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_judges.py`:

```python
def test_pilot_honours_a_model_set_override(tmp_path) -> None:
    from data_quality_checker.judges import gemini_judge_model

    config = prepared_processed_fixture(tmp_path)
    summary = run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=("qwen3.5:397b", gemini_judge_model()),
    )
    assert set(summary["models"]) == {"qwen3.5:397b", gemini_judge_model()}


def test_pilot_rejects_an_unregistered_model_in_the_override(tmp_path) -> None:
    config = prepared_processed_fixture(tmp_path)
    with pytest.raises(ContractError):
        run_judge_pilot(
            config=config,
            batch_id="batch",
            allow_external_judge=True,
            provider=FakeJudgeProvider(),
            judge_models=("made-up-model",),
        )
```

Add the `ContractError` import at the top of `tests/test_judges.py`. The file currently has `from data_quality_checker.errors import GateBlocked`; change it to:

```python
from data_quality_checker.errors import ContractError, GateBlocked
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_judges.py -k override -q`
Expected: FAIL — `TypeError: run_judge_pilot() got an unexpected keyword argument 'judge_models'`

- [ ] **Step 3: Thread the parameter through `judges.py`**

Add `judge_models: tuple[str, ...] | None = None` to the signature of both `run_judge_pilot` and `_run_pilot_impl`, and pass it down from the former to the latter.

At the top of `_run_pilot_impl`, immediately after the `allow_external_judge` consent check, resolve and validate it:

```python
    models = tuple(judge_models) if judge_models else JUDGE_MODELS
    unknown = [model for model in models if model not in judge_model_providers()]
    if unknown:
        raise ContractError(
            f"unknown judge models {unknown}; expected from {sorted(judge_model_providers())}"
        )
```

Then replace every `JUDGE_MODELS` reference **inside `_run_pilot_impl`** with `models`. There are four, at roughly lines 326, 335, 337-338 and 348 of the current file:

```python
        "models": list(models),
```
```python
        results: dict[str, dict[str, int]] = {
            model: {"valid": 0, "unavailable": 0, "error": 0} for model in models
        }
        total_latency: dict[str, float] = {model: 0.0 for model in models}
        total_cost: dict[str, float] = {model: 0.0 for model in models}
```
```python
            for model in models:
```

and the summary comprehension near line 478:

```python
                for model in models
```

Leave the module-level `JUDGE_MODELS` constant itself unchanged — it stays the default pair.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_judges.py -q`
Expected: PASS — all prior tests plus the 2 new ones.

- [ ] **Step 5: Add the CLI flag**

In `cli.py`, in the `pilot-judges` parser block, after the `--allow-external-judge` line, add:

```python
    pilot.add_argument(
        "--judge-models",
        default=None,
        help="comma-separated judge model ids; defaults to the two-model pilot pair",
    )
```

In `commands.py`, inside the `pilot_judges` handler, before the `run_judge_pilot(...)` call:

```python
    judge_models = (
        tuple(part.strip() for part in args.judge_models.split(",") if part.strip())
        if getattr(args, "judge_models", None)
        else None
    )
```

and add `judge_models=judge_models` to that call.

- [ ] **Step 6: Verify the CLI end to end with the fake backend**

```bash
cd /Users/student2/data-quality-checker
python sample_data/generate_sample_zips.py
.venv/bin/python -m data_quality_checker --config configs/presets/sample_data.json prepare \
  --annotation-zip sample_data/mock_annotations.zip \
  --document-pool-zip sample_data/mock_documents.zip \
  --batch-id judge_demo --hmac-key-file sample_data/sample_hmac.key
.venv/bin/python -m data_quality_checker --config configs/presets/sample_data.json process \
  --prepared-batch judge_demo --fake-backend
.venv/bin/python -m data_quality_checker --config configs/presets/sample_data.json pilot-judges \
  --batch-id judge_demo --allow-external-judge --fake-backend \
  --judge-models "qwen3.5:397b,gemini-3.1-pro"
```

Expected: exit 0. Then confirm the model set landed in the artifact:

```bash
SUMMARY=$(find . -name judge_pilot_summary.json -mmin -5 | head -1)
.venv/bin/python -c "import json,sys;print(sorted(json.load(open(sys.argv[1]))['models']))" "$SUMMARY"
```

Expected output: `['gemini-3.1-pro', 'qwen3.5:397b']`

- [ ] **Step 7: Document the flag**

In `docs/CLI_REFERENCE.md`, under the `pilot-judges` section, add:

```markdown
| `--judge-models` | Comma-separated judge model ids. Defaults to the two-model pilot pair. Every id must be registered in `judge_model_providers()`. |

**Gemini judge environment:**

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | — | Required. An absent key raises `JudgeProviderUnavailable`. |
| `GEMINI_JUDGE_MODEL` | `gemini-3.1-pro` | Model id used as the Gemini judge. |
| `GEMINI_BASE_URL` | `https://aiplatform.googleapis.com/v1/publishers/google/models` | Endpoint prefix. |
| `GEMINI_TIMEOUT` | `500` | Per-request timeout in seconds. |
```

- [ ] **Step 8: Commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
git add src/data_quality_checker/judges.py src/data_quality_checker/cli.py \
        src/data_quality_checker/commands.py tests/test_judges.py docs/CLI_REFERENCE.md
git commit -m "feat(cli): choose the judge model set for the pilot"
```

---

### Task 4: Route providers per model and allow a Gemini lock

**Files:**
- Modify: `src/data_quality_checker/judges.py` — provider construction in `_run_pilot_impl` (around line 305) and in the green-audit path (around line 640); `lock_judge` validation and `all_metrics` (around lines 742-755)
- Test: `tests/test_judges.py`

**Interfaces:**
- Consumes: `resolve_judge_provider` (Task 2), `judge_models` (Task 3).
- Produces: no new public names. `lock_judge` accepts any key returned by `judge_model_providers()` and computes `pilot_model_metrics` over the models that actually ran in the pilot.

> **The bug this task must avoid.** `lock_judge` currently builds
> `all_metrics = {candidate: ... for candidate in JUDGE_MODELS}` and then reads
> `metrics = all_metrics[model]`. Widening only the validation check would make
> a Gemini lock raise `KeyError`, because Gemini is not in `JUDGE_MODELS`. The
> candidate set must come from the pilot summary instead.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_judges.py`. This mirrors the existing
`test_second_run_after_explicit_lock_executes_production_coverage` flow, because
`lock_judge` requires a completed pilot and a finalized expert review:

```python
def test_lock_judge_accepts_a_gemini_model_that_ran_in_the_pilot(tmp_path) -> None:
    from data_quality_checker.judges import gemini_judge_model

    config = prepared_processed_fixture(tmp_path)
    with Store(config.database_path) as store:
        document = store.list_documents("batch")[0]
        store.set_router_bucket("batch", document["internal_doc_id"], "RED")

    run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=("qwen3.5:397b", gemini_judge_model()),
    )
    with Store(config.database_path) as store:
        review = store.get_review("batch", document["internal_doc_id"])
        assert review is not None
        store.update_review(
            batch_id="batch",
            internal_doc_id=document["internal_doc_id"],
            expected_version=review["row_version"],
            status="finalized",
            action="accept_human",
            final_references=[],
            reason=None,
            reviewer="fixture",
        )

    payload = lock_judge(
        config=config,
        batch_id="batch",
        model=gemini_judge_model(),
        reason="dq-loop production judge",
    )
    assert payload["model"] == gemini_judge_model()
    assert set(payload["pilot_model_metrics"]) == {"qwen3.5:397b", gemini_judge_model()}
    written = json.loads(
        (config.public_root / "batches" / "batch" / "judge_lock.json").read_text(encoding="utf-8")
    )
    assert written["model"] == gemini_judge_model()


def test_lock_judge_rejects_an_unregistered_model(tmp_path) -> None:
    config = prepared_processed_fixture(tmp_path)
    with pytest.raises(ContractError):
        lock_judge(config=config, batch_id="batch", model="made-up-model", reason="test")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_judges.py -k gemini_model_that_ran -q`
Expected: FAIL — `ContractError: model must be one of ('qwen3.5:397b', 'deepseek-v3.2')`

- [ ] **Step 3: Widen `lock_judge` validation and fix the candidate set**

In `lock_judge`, replace:

```python
    if model not in JUDGE_MODELS:
        raise ContractError(f"model must be one of {JUDGE_MODELS}")
```

with:

```python
    if model not in judge_model_providers():
        raise ContractError(f"model must be one of {sorted(judge_model_providers())}")
```

Then, after the `selection` file is loaded and before the `Store` block, derive the
candidate set from the pilot summary so it matches the models that actually ran:

```python
    summary_path = config.public_root / "batches" / batch_id / "judge_pilot_summary.json"
    try:
        pilot_models = tuple(json.loads(summary_path.read_text(encoding="utf-8"))["models"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pilot_models = JUDGE_MODELS
    if model not in pilot_models:
        raise GateBlocked(f"judge {model} did not run in this batch's pilot")
```

and replace the `all_metrics` comprehension:

```python
        all_metrics = {
            candidate: judge_expert_metrics(store, batch_id=batch_id, model=candidate, ids=ids)
            for candidate in pilot_models
        }
```

The later `summary_path` re-read further down `lock_judge` stays as it is; the
variable is already computed here and can be reused rather than re-derived.

- [ ] **Step 4: Run the lock tests**

Run: `.venv/bin/python -m pytest tests/test_judges.py -k lock -q`
Expected: PASS — the two new tests plus the existing lock tests.

- [ ] **Step 5: Route the provider per model in the pilot**

In `_run_pilot_impl`, delete:

```python
    selected_provider: JudgeProvider = provider or (
        FakeJudgeProvider() if fake_backend else OllamaJudgeProvider()
    )
```

and inside the per-model loop, immediately before the judging call, add:

```python
            model_provider = provider or resolve_judge_provider(model, fake_backend=fake_backend)
```

Replace the `selected_provider.judge(...)` call with `model_provider.judge(...)`.
Keeping `provider or` first preserves every existing test that injects a
`FakeJudgeProvider()` or a stub.

- [ ] **Step 6: Apply the identical change in the green-audit path**

The second construction site (around line 640) has the same two lines. Replace
them the same way, resolving per model at the point of use.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — 206 total. That is 191 on `main` plus 15: 4 from Task 1, 1 from
Task 1's review fix, 6 from Task 2, 2 from Task 3, 2 from Task 4.

- [ ] **Step 8: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src
git add src/data_quality_checker/judges.py tests/test_judges.py
git commit -m "feat(judges): resolve providers per model and allow a Gemini lock"
```

---

## Done When

- `GeminiJudgeProvider` exists, is registered, and is reachable from `pilot-judges` and `judge-lock`.
- `.venv/bin/python -m pytest -q` passes with 15 new tests: 191 on `main` -> 206. (4 from Task 1, 1 from Task 1's review fix, 6 from Task 2, 2 from Task 3, 2 from Task 4.)
- `ruff check` passes. `mypy src` reports **no new** errors: the tree already has 37 pre-existing errors on `main`; the count and file list must be unchanged.
- The real model id is confirmed against the live API and recorded in `Journal/experiments/dq_loop/RUNBOOK.md`.
- `requirements.txt` is unchanged.

## Explicitly Out Of Scope

- Round orchestration, batch splitting, training, checkpoint selection — separate plans.
- Any change to `_validate_judge_result`. Gemini meets the existing contract or its output is rejected.
- Promoting a judge verdict to a decision. The judge stays a third opinion; the expert adjudicates.
