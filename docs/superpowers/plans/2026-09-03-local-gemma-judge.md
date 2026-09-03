# Local Gemma 4 31B Judge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Register the locally-cached Gemma 4 31B as a judge provider so the DQ-Loop adjudicates without any cloud API key.

**Architecture:** A fourth `JudgeProvider` implementation (`MlxJudgeProvider`) that loads the revision-pinned local snapshot through `mlx_lm`, lazily, once per process. It returns raw text so the existing `_validate_judge_result` contract applies unchanged. The provider registry gains a static entry; `resolve_judge_provider` becomes an explicit dispatch instead of an Ollama fallthrough.

**Tech Stack:** mlx-lm 0.31.2, huggingface_hub (offline snapshot resolution), Python stdlib.

## Global Constraints

- Judge repo id: `mlx-community/gemma-4-31B-it-OptiQ-4bit` (exact).
- Judge revision: `23616162c5a8f928cac5b21d3e974d1dbc0b9877` (exact, pinned).
- Judge registry model id: `gemma-4-31b-it-optiq-4bit` (exact).
- Gemma thinking-disable sentinel, verbatim: `<|channel>thought\n<channel|>`
- Sampling temperature is exactly 0. Reproducibility is the reason this model was chosen; nothing may make generation nondeterministic.
- Never `pip install`, never create a venv. The interpreter is `/opt/llm-lab/.venv/bin/python`, reached in-repo as `.venv/bin/python`.
- Constructing a provider must NOT load model weights. Tests construct providers; a 21 GB load in `__init__` is a defect.
- `g0.py` is not to be modified.
- Existing judge behaviour for `qwen3.5:397b`, `deepseek-v3.2` and the Gemini entry must not change.

---

### Task 1: Pin the judge model and assert Gemma's thinking sentinel

**Files:**
- Modify: `src/data_quality_checker/constants.py`
- Create: `src/data_quality_checker/judge_model.py`
- Test: `tests/test_judge_model.py`

**Interfaces:**
- Produces: `JUDGE_MODEL_ID`, `JUDGE_MODEL_REVISION`, `JUDGE_MODEL_KEY` in `constants.py`;
  `resolve_judge_snapshot() -> Path` and `assert_gemma_thinking_is_disabled(prompt: str) -> None` in `judge_model.py`.

**Context:** `mlx_compute.py:213` already has `_model_cache_dir()`, which honours `HF_HUB_CACHE`, then `/opt/llm-lab/hf-cache/hub`, then the home cache. Import and reuse it — do not write a second cache resolver. `mlx_compute.assert_thinking_is_disabled` is Qwen-specific (`<think>\n\n</think>\n\n`); this is its Gemma counterpart, and both must keep existing independently.

- [ ] **Step 1: Add the constants**

In `src/data_quality_checker/constants.py`, below the existing `MODEL_ID` / `MODEL_REVISION` / `MINIMUM_MLX_LM` block:

```python
JUDGE_MODEL_ID = "mlx-community/gemma-4-31B-it-OptiQ-4bit"
JUDGE_MODEL_REVISION = "23616162c5a8f928cac5b21d3e974d1dbc0b9877"
JUDGE_MODEL_KEY = "gemma-4-31b-it-optiq-4bit"
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_judge_model.py`:

```python
import pytest

from data_quality_checker.errors import IntegrityError
from data_quality_checker.judge_model import (
    assert_gemma_thinking_is_disabled,
    resolve_judge_snapshot,
)

SENTINEL = "<|channel>thought\n<channel|>"


def test_sentinel_present_is_accepted() -> None:
    assert_gemma_thinking_is_disabled("<|turn>model\n" + SENTINEL)


def test_missing_sentinel_is_rejected() -> None:
    with pytest.raises(IntegrityError):
        assert_gemma_thinking_is_disabled("<|turn>model\n")


def test_open_thought_channel_is_rejected() -> None:
    """A thinking channel that was opened and left open must not pass."""
    with pytest.raises(IntegrityError):
        assert_gemma_thinking_is_disabled("<|turn>model\n<|channel>thought\n")


def test_non_empty_thought_channel_is_rejected() -> None:
    """Content inside the thought channel means thinking is on."""
    with pytest.raises(IntegrityError):
        assert_gemma_thinking_is_disabled(
            "<|turn>model\n<|channel>thought\nreasoning here\n<channel|>"
        )


def test_snapshot_resolves_from_the_lab_cache() -> None:
    snapshot = resolve_judge_snapshot()
    assert snapshot.is_dir()
    assert (snapshot / "config.json").is_file()
    assert snapshot.name == "23616162c5a8f928cac5b21d3e974d1dbc0b9877"


def test_missing_snapshot_raises_judge_provider_unavailable(monkeypatch, tmp_path) -> None:
    from data_quality_checker.judges import JudgeProviderUnavailable

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(JudgeProviderUnavailable):
        resolve_judge_snapshot()
```

Note on the fourth test: a non-empty thought channel ends with the same closing tag as an empty one, so an implementation that only checks `endswith(SENTINEL)` will pass it. That is intended — write the implementation so it rejects content between the tags.

- [ ] **Step 3: Run the tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_judge_model.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_quality_checker.judge_model'`

- [ ] **Step 4: Implement**

Create `src/data_quality_checker/judge_model.py`:

```python
"""Locate and validate the local adjudicator model.

The judge is pinned to one snapshot revision on purpose: a locked judge
whose weights can change is not a fixed instrument, and every round's
adjudication would be measured against a different rater.
"""

from __future__ import annotations

from pathlib import Path

from .constants import JUDGE_MODEL_ID, JUDGE_MODEL_REVISION
from .errors import IntegrityError

GEMMA_THINKING_SENTINEL = "<|channel>thought\n<channel|>"


def assert_gemma_thinking_is_disabled(prompt: str) -> None:
    """Validate Gemma 4's empty, closed no-thinking sentinel.

    Counterpart to mlx_compute.assert_thinking_is_disabled, which encodes
    Qwen3.5's different sentinel. Measured behaviour: with thinking left on,
    Gemma spends its whole budget in the thought channel and returns no JSON
    at all, so this is a correctness gate, not a cosmetic one.
    """

    if not prompt.endswith(GEMMA_THINKING_SENTINEL):
        raise IntegrityError("Gemma thinking-disable sentinel is missing, open, or non-empty")


def resolve_judge_snapshot() -> Path:
    """Return the pinned judge snapshot from the local cache, offline."""

    from huggingface_hub import snapshot_download

    from .judges import JudgeProviderUnavailable
    from .mlx_compute import _model_cache_dir

    try:
        path = snapshot_download(
            repo_id=JUDGE_MODEL_ID,
            revision=JUDGE_MODEL_REVISION,
            cache_dir=str(_model_cache_dir()),
            local_files_only=True,
        )
    except Exception as exc:  # huggingface_hub raises several unrelated types offline
        raise JudgeProviderUnavailable(
            f"judge snapshot {JUDGE_MODEL_ID}@{JUDGE_MODEL_REVISION[:12]} "
            f"is not in the local cache: {exc}"
        ) from exc
    return Path(path).resolve()
```

Note the sentinel check: `endswith` alone passes the non-empty case from Step 2, because a filled thought channel also ends with `<channel|>`. Make it reject that — check that the final `<|channel>thought` occurrence is immediately followed by the closing tag.

- [ ] **Step 5: Run the tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_judge_model.py -q`
Expected: 6 passed

- [ ] **Step 6: Prove the sentinel test discriminates**

Temporarily weaken `assert_gemma_thinking_is_disabled` to `return None`. Re-run. At least three tests must fail. Restore, re-run, all pass. Report both outputs.

- [ ] **Step 7: Commit**

```bash
git add src/data_quality_checker/constants.py src/data_quality_checker/judge_model.py tests/test_judge_model.py
git commit -m "feat(judge): pin the local Gemma judge snapshot and its no-thinking sentinel"
```

---

### Task 2: MlxJudgeProvider and explicit provider dispatch

**Files:**
- Modify: `src/data_quality_checker/judges.py`
- Test: `tests/test_judges.py` (append; do not restructure existing tests)

**Interfaces:**
- Consumes: `JUDGE_MODEL_KEY` from `constants.py`; `resolve_judge_snapshot`, `assert_gemma_thinking_is_disabled` from `judge_model.py`.
- Produces: `MlxJudgeProvider` in `judges.py`; `_STATIC_JUDGE_MODEL_PROVIDERS` gains `"gemma-4-31b-it-optiq-4bit": "mlx"`.

**Context — the existing shape you must match:** `JudgeProvider` is a Protocol with one method, `judge(self, *, model: str, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]`. `OllamaJudgeProvider.judge` builds its prompt as `JUDGE_PROMPT + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))` and returns `(content, {"latency_seconds": ..., "cost": ..., "provider": "ollama"})`. Return raw text: `_validate_judge_result` parses and contract-checks it identically for every provider.

**Measured behaviour of this model on a real 9908-character özelge with 18 references** — use these to choose the token budget, and do not lower it: prompt 5266 tokens, prefill 257 tok/s, generation 24.6 tok/s, 1370 generation tokens to produce a complete verdict, peak memory 26.65 GB, 76.5 s wall for one document.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_judges.py`:

```python
def test_local_gemma_is_registered_as_an_mlx_provider() -> None:
    from data_quality_checker.constants import JUDGE_MODEL_KEY
    from data_quality_checker.judges import judge_model_providers

    assert judge_model_providers()[JUDGE_MODEL_KEY] == "mlx"


def test_resolve_returns_the_mlx_provider_not_an_ollama_fallthrough(monkeypatch) -> None:
    """Regression: dispatch fell through to Ollama for every non-gemini kind."""
    from data_quality_checker.constants import JUDGE_MODEL_KEY
    from data_quality_checker.judges import MlxJudgeProvider, resolve_judge_provider

    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    for index in range(2, 8):
        monkeypatch.delenv(f"OLLAMA_API_KEY_V{index}", raising=False)
    provider = resolve_judge_provider(JUDGE_MODEL_KEY)
    assert isinstance(provider, MlxJudgeProvider)


def test_constructing_the_mlx_provider_does_not_load_weights() -> None:
    """A 21 GB load in __init__ would make every test that builds a provider unusable."""
    from data_quality_checker.judges import MlxJudgeProvider

    provider = MlxJudgeProvider()
    assert provider._model is None
    assert provider._tokenizer is None


def test_mlx_provider_strips_a_code_fence_before_returning(monkeypatch) -> None:
    """Ollama and Gemini enforce JSON server-side; a local model has no such gate."""
    from data_quality_checker.judges import MlxJudgeProvider

    provider = MlxJudgeProvider()
    monkeypatch.setattr(
        provider, "_generate", lambda prompt: ('```json\n{"verdict":"A"}\n```', {})
    )
    content, _ = provider.judge(model="gemma-4-31b-it-optiq-4bit", payload={"x": 1})
    assert content == '{"verdict":"A"}'


def test_unknown_judge_model_still_raises(monkeypatch) -> None:
    from data_quality_checker.judges import resolve_judge_provider
    from data_quality_checker.errors import ContractError

    with pytest.raises(ContractError):
        resolve_judge_provider("no-such-judge-model")
```

- [ ] **Step 2: Run them, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_judges.py -q`
Expected: FAIL — `MlxJudgeProvider` does not exist.

- [ ] **Step 3: Register the model**

In `judges.py`, extend the static registry:

```python
_STATIC_JUDGE_MODEL_PROVIDERS: dict[str, str] = {
    "qwen3.5:397b": "ollama",
    "deepseek-v3.2": "ollama",
    JUDGE_MODEL_KEY: "mlx",
}
```

Import `JUDGE_MODEL_KEY` from `.constants` at the top of the module.

- [ ] **Step 4: Implement the provider**

Add to `judges.py`, after `OllamaJudgeProvider`:

```python
class MlxJudgeProvider:
    """Local adjudicator on the pinned Gemma 4 31B snapshot.

    Weights load lazily and stay loaded: a judged batch is many documents and
    reloading 21 GB per call would dominate the run. Sampling is greedy, and
    the snapshot revision is pinned, so a re-run of the same batch reproduces
    the same verdicts -- which is the property a locked judge needs and a
    cloud endpoint cannot promise.
    """

    def __init__(self) -> None:
        self.max_tokens = int(os.environ.get("MLX_JUDGE_MAX_TOKENS", "2048"))
        self._snapshot = resolve_judge_snapshot()
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise JudgeProviderUnavailable(f"mlx_lm is unavailable: {exc}") from exc
        self._model, self._tokenizer = load(str(self._snapshot))

    def _generate(self, prompt: str) -> tuple[str, dict[str, Any]]:
        self._load()
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        assert_gemma_thinking_is_disabled(text)
        chunks: list[str] = []
        last = None
        for response in stream_generate(
            self._model,
            self._tokenizer,
            text,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=0.0),
        ):
            chunks.append(response.text)
            last = response
        meta = {
            "prompt_tokens": getattr(last, "prompt_tokens", None),
            "generation_tokens": getattr(last, "generation_tokens", None),
            "peak_memory_gb": getattr(last, "peak_memory", None),
            "finish_reason": getattr(last, "finish_reason", None),
        }
        return "".join(chunks), meta

    def judge(self, *, model: str, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        prompt = JUDGE_PROMPT + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        started = time.perf_counter()
        content, meta = self._generate(prompt)
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("```")[1].removeprefix("json").strip()
        return stripped, {
            "latency_seconds": time.perf_counter() - started,
            "cost": 0.0,
            "provider": "mlx",
            "model_revision": JUDGE_MODEL_REVISION,
            **meta,
        }
```

Import `JUDGE_MODEL_REVISION` from `.constants` and the two helpers from `.judge_model`.

- [ ] **Step 5: Make dispatch explicit**

Replace the fallthrough in `resolve_judge_provider`:

```python
    if fake_backend:
        return FakeJudgeProvider()
    if kind == "gemini":
        return GeminiJudgeProvider()
    if kind == "mlx":
        return MlxJudgeProvider()
    if kind == "ollama":
        return OllamaJudgeProvider()
    raise ContractError(f"judge model {model!r} maps to unknown provider kind {kind!r}")
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, count strictly greater than the 384 on `main`.

- [ ] **Step 7: Prove the dispatch test discriminates**

Restore the old `return OllamaJudgeProvider()` fallthrough. Re-run `tests/test_judges.py`. `test_resolve_returns_the_mlx_provider_not_an_ollama_fallthrough` must fail. Restore, re-run, all pass. Report both outputs.

- [ ] **Step 8: Lint and commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
git add -A && git commit -m "feat(judge): adjudicate on the local Gemma 4 31B snapshot"
```

---

### Task 3: End-to-end proof against the real model

**Files:**
- Create: `tests/test_judges_live_mlx.py`

**Context:** Every other test in this repo runs without weights. This one does not — it is the only thing that proves the provider actually adjudicates, and the two prior benchmark runs showed exactly why it is needed: with thinking left on, every layer above still "worked" while producing no JSON at all. Mark it so the default suite stays fast.

- [ ] **Step 1: Write the test**

```python
"""Live adjudication against the pinned local judge. Opt-in: needs 27 GB.

Run with: .venv/bin/python -m pytest tests/test_judges_live_mlx.py -q -m live_mlx
"""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DQCHECK_LIVE_MLX_JUDGE") != "1",
    reason="set DQCHECK_LIVE_MLX_JUDGE=1 to run the live judge test",
)

GT = Path(
    "data/ground_truth/gt_v3_triangulated_2026-05-15/validated/doc_266.json"
)


def test_local_judge_returns_a_contract_valid_verdict() -> None:
    from data_quality_checker.judges import MlxJudgeProvider, _validate_judge_result

    document = json.loads(GT.read_text(encoding="utf-8"))
    references = document["references"]
    corrupted = [dict(reference) for reference in references[:-1]]
    corrupted[0]["madde"] = "5"

    provider = MlxJudgeProvider()
    content, meta = provider.judge(
        model="gemma-4-31b-it-optiq-4bit",
        payload={
            "document_text": document["text"],
            "candidate_a": references,
            "candidate_b": corrupted,
        },
    )
    assert meta["provider"] == "mlx"
    assert meta["finish_reason"] == "stop", "the model ran out of budget mid-verdict"

    result = _validate_judge_result(content, document["text"])
    assert result["verdict"] == "A", "the unperturbed candidate should win"
```

- [ ] **Step 2: Register the marker**

Add `live_mlx` to the `markers` list in the project's pytest configuration if one exists; if the project has no marker configuration, skip this step rather than creating one.

- [ ] **Step 3: Run it live**

Run: `DQCHECK_LIVE_MLX_JUDGE=1 .venv/bin/python -m pytest tests/test_judges_live_mlx.py -q`
Expected: 1 passed, in roughly 80-120 seconds. Report the actual wall time.

- [ ] **Step 4: Confirm the default suite still skips it**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add tests/test_judges_live_mlx.py
git commit -m "test(judge): prove the local judge adjudicates a real document"
```
