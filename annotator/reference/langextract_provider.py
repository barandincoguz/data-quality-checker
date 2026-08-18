"""LangExtract-backed provider for reference extraction."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from annotator.reference.gemini_aiplatform import generate_gemini_content, resolve_gemini_api_key

from annotator.reference.llm_prompt import (
    FEW_SHOT_PROMPT_VARIANTS,
    SUPPORTED_PROMPT_VARIANTS,
    build_langextract_prompt_description,
    build_system_prompt,
    is_few_shot_prompt_variant,
    load_prompt_example_records,
)

DEFAULT_LANGEXTRACT_BACKEND = "gemini"
DEFAULT_OLLAMA_BASE_URL = "https://ollama.com"
DEFAULT_OLLAMA_OPENAI_BASE_URL = "https://ollama.com/v1"
DEFAULT_LANGEXTRACT_TIMEOUT_SECONDS = 500
REFERENCE_TYPE = "kanun_madde_referansi"
SUPPORTED_LANGEXTRACT_BACKENDS = {"gemini", "openai", "ollama"}


def _strip_trailing_slash(value: str) -> str:
    """Return a URL-like string without a trailing slash."""
    return str(value or "").rstrip("/")


def _resolve_timeout_seconds(value: Any) -> int:
    """Resolve a positive request timeout value."""
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return DEFAULT_LANGEXTRACT_TIMEOUT_SECONDS
    return resolved if resolved > 0 else DEFAULT_LANGEXTRACT_TIMEOUT_SECONDS


def _split_langextract_prompt(prompt: str) -> tuple[str, str]:
    """Split the final LangExtract QA prompt into system instruction and user text."""
    text = str(prompt or "")
    question_marker = "\nQ: "
    answer_marker = "\nA:"
    question_start = text.rfind(question_marker)
    if question_start < 0:
        return "", text

    answer_start = text.find(answer_marker, question_start + len(question_marker))
    if answer_start < 0:
        return "", text

    system_prompt = text[:question_start].rstrip()
    user_text = text[question_start + len(question_marker) : answer_start].strip()
    if not system_prompt or not user_text:
        return "", text
    return system_prompt, user_text


@lru_cache(maxsize=1)
def _build_prompt_variant_lookup() -> dict[str, str]:
    """Build an exact system-prompt to prompt-variant lookup table."""
    lookup: dict[str, str] = {}
    for variant in SUPPORTED_PROMPT_VARIANTS:
        system_prompt = build_system_prompt(variant)
        existing_variant = lookup.get(system_prompt)
        if existing_variant and existing_variant != variant:
            raise ValueError(
                "Aynı system prompt birden fazla prompt varyantına eşleniyor: "
                f"{existing_variant!r} ve {variant!r}"
            )
        lookup[system_prompt] = variant
    return lookup


class LangExtractProvider:
    """Provider that routes LangExtract through Gemini, OpenAI, or Ollama backends."""

    name = "langextract"
    _OLLAMA_AUTH_HEADER = "Authorization"
    _OLLAMA_AUTH_SCHEME = "Bearer"

    def __init__(self, **kwargs: Any) -> None:
        self._timeout_seconds = _resolve_timeout_seconds(
            kwargs.get("timeout") or os.environ.get("LANGEXTRACT_TIMEOUT")
        )
        self._ollama_api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
        self._ollama_base_url = _strip_trailing_slash(
            os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        ) or DEFAULT_OLLAMA_BASE_URL
        self._ollama_openai_base_url = _strip_trailing_slash(
            os.environ.get("OLLAMA_OPENAI_BASE_URL", DEFAULT_OLLAMA_OPENAI_BASE_URL)
        ) or DEFAULT_OLLAMA_OPENAI_BASE_URL
        self._default_backend = str(
            os.environ.get("LANGEXTRACT_BACKEND", DEFAULT_LANGEXTRACT_BACKEND)
        ).strip().lower() or DEFAULT_LANGEXTRACT_BACKEND
        if self._default_backend not in SUPPORTED_LANGEXTRACT_BACKENDS:
            supported = ", ".join(sorted(SUPPORTED_LANGEXTRACT_BACKENDS))
            raise ValueError(
                f"Unsupported LANGEXTRACT_BACKEND: {self._default_backend!r}. Supported: {supported}"
            )

        self._lx = self._load_langextract_module()
        self._prompt_variant_by_system_prompt = _build_prompt_variant_lookup()
        self._prompt_descriptions = {
            variant: build_langextract_prompt_description(variant)
            for variant in FEW_SHOT_PROMPT_VARIANTS
        }
        self._examples_by_variant = self._build_langextract_example_banks()

    @staticmethod
    def _load_langextract_module() -> Any:
        """Import LangExtract lazily so initialization fails with a clear message."""
        try:
            import langextract as lx
        except ImportError as exc:
            raise ImportError(
                "LangExtract provider için `langextract[openai]` kurulmalı. "
                "requirements.txt güncellemesini yükleyip tekrar deneyin."
            ) from exc
        return lx

    def _build_langextract_example_banks(self) -> dict[str, tuple[Any, ...]]:
        """Convert repo-owned example JSON files into LangExtract ExampleData banks."""
        banks: dict[str, tuple[Any, ...]] = {}
        for variant in FEW_SHOT_PROMPT_VARIANTS:
            example_records = load_prompt_example_records(variant)
            banks[variant] = tuple(self._build_example_data(record) for record in example_records)
        return banks

    def _build_example_data(self, record: dict[str, Any]) -> Any:
        """Convert one repo example record into LangExtract ExampleData."""
        input_text = str(record.get("input_text", "")).strip()
        if not input_text:
            raise ValueError(f"LangExtract example text boş olamaz: {record.get('id', '<unknown>')}")

        expected_output = record.get("expected_output", {})
        references = expected_output.get("references", []) if isinstance(expected_output, dict) else []
        extractions: list[Any] = []

        for reference in references:
            if not isinstance(reference, dict):
                continue
            source_text = str(reference.get("source_text", "")).strip()
            if not source_text:
                continue
            extractions.append(
                self._lx.data.Extraction(
                    extraction_class=REFERENCE_TYPE,
                    extraction_text=source_text,
                    attributes={
                        "kanun_no": str(reference.get("kanun_no", "")).strip(),
                        "kanun_ad": str(reference.get("kanun_ad", "")).strip(),
                        "madde": str(reference.get("madde", "")).strip(),
                        "fikra": str(reference.get("fikra", "")).strip(),
                        "bent": str(reference.get("bent", "")).strip(),
                    },
                )
            )

        return self._lx.data.ExampleData(text=input_text, extractions=extractions)

    def _detect_prompt_variant(self, system_prompt: str) -> str:
        """Resolve the exact prompt variant from a built system prompt."""
        variant = self._prompt_variant_by_system_prompt.get(str(system_prompt))
        if variant is None:
            raise ValueError("LangExtract provider yalnız repo içi tanımlı system prompt varyantlarını destekler.")
        if not is_few_shot_prompt_variant(variant):
            raise ValueError(
                "LangExtract provider şu an yalnız curated few-shot prompt varyantlarını destekler. "
                f"Gelen varyant: {variant!r}"
            )
        return variant

    def _resolve_backend_and_model(self, model: str) -> tuple[str, str]:
        """Resolve backend choice from the model string or environment default."""
        raw_model = str(model or "").strip()
        if not raw_model:
            raise ValueError("LangExtract provider için model boş olamaz.")

        if "/" in raw_model:
            backend, model_id = raw_model.split("/", 1)
            normalized_backend = backend.strip().lower()
            normalized_model_id = model_id.strip()
            if normalized_backend in SUPPORTED_LANGEXTRACT_BACKENDS and normalized_model_id:
                return normalized_backend, normalized_model_id

        return self._default_backend, raw_model

    def _build_langextract_model(self, *, backend: str, model_id: str, temperature: float) -> Any | None:
        """Build a custom LangExtract model adapter when repo-specific transport is required."""
        if backend != "gemini":
            return None

        from langextract.core import base_model as lx_base_model
        from langextract.core import data as lx_data
        from langextract.core import exceptions as lx_exceptions
        from langextract.core import types as lx_types

        resolved_api_key = resolve_gemini_api_key()

        class RepoGeminiAiplatformLanguageModel(lx_base_model.BaseLanguageModel):
            """LangExtract adapter that reuses the repo's working Gemini REST path."""

            def __init__(self) -> None:
                super().__init__(constraint=lx_types.Constraint(constraint_type=lx_types.ConstraintType.NONE))
                self.model_id = model_id
                self.api_key = resolved_api_key
                self.temperature = float(temperature)
                self.format_type = lx_data.FormatType.JSON

            @property
            def requires_fence_output(self) -> bool:
                """The repo Gemini wrapper requests raw JSON via responseMimeType."""
                return False

            def infer(self, batch_prompts: list[str] | tuple[str, ...], **kwargs: Any):
                merged_kwargs = self.merge_kwargs(kwargs)
                runtime_temperature = float(merged_kwargs.get("temperature", self.temperature))
                for prompt in batch_prompts:
                    system_prompt, user_text = _split_langextract_prompt(str(prompt))
                    try:
                        output = generate_gemini_content(
                            text=user_text,
                            system_prompt=system_prompt,
                            model=self.model_id,
                            temperature=runtime_temperature,
                            api_key=self.api_key,
                        )
                    except Exception as exc:
                        raise lx_exceptions.InferenceRuntimeError(
                            f"Gemini API error: {str(exc)}",
                            original=exc,
                        ) from exc
                    yield [lx_types.ScoredOutput(score=1.0, output=output)]

        return RepoGeminiAiplatformLanguageModel()

    def _build_config(self, *, backend: str, model_id: str, temperature: float) -> Any:
        """Build a LangExtract ModelConfig for built-in backends."""
        provider_kwargs: dict[str, Any] = {
            "temperature": float(temperature),
            "timeout": self._timeout_seconds,
        }

        if backend == "openai":
            base_url = str(os.environ.get("OPENAI_BASE_URL", "")).strip()
            if base_url:
                provider_kwargs["base_url"] = _strip_trailing_slash(base_url)
            return self._lx.factory.ModelConfig(
                model_id=model_id,
                provider="openai",
                provider_kwargs=provider_kwargs,
            )

        if not self._ollama_api_key:
            raise ValueError("LangExtract ollama backend için OLLAMA_API_KEY set edilmemiş.")

        native_ollama_kwargs = {
            **provider_kwargs,
            "model_url": self._ollama_base_url,
            "api_key": self._ollama_api_key,
            "auth_header": self._OLLAMA_AUTH_HEADER,
            "auth_scheme": self._OLLAMA_AUTH_SCHEME,
            "format_type": self._lx.data.FormatType.JSON,
        }
        return self._lx.factory.ModelConfig(
            model_id=model_id,
            provider="ollama",
            provider_kwargs=native_ollama_kwargs,
        )

    def _build_ollama_openai_fallback(self, *, model_id: str, temperature: float) -> Any:
        """Build an OpenAI-compatible fallback config for Ollama Cloud."""
        return self._lx.factory.ModelConfig(
            model_id=model_id,
            provider="openai",
            provider_kwargs={
                "base_url": self._ollama_openai_base_url,
                "api_key": self._ollama_api_key,
                "temperature": float(temperature),
                "timeout": self._timeout_seconds,
            },
        )

    def _run_langextract(
        self,
        *,
        text: str,
        prompt_description: str,
        examples: tuple[Any, ...],
        config: Any | None = None,
        model: Any | None = None,
        fence_output: bool,
    ) -> Any:
        """Run one LangExtract request with a prebuilt provider config."""
        extract_kwargs: dict[str, Any] = {
            "text_or_documents": text,
            "prompt_description": prompt_description,
            "examples": list(examples),
            "use_schema_constraints": False,
            "fence_output": fence_output,
            "batch_length": 1,
            "max_workers": 1,
            "fetch_urls": False,
            "show_progress": False,
        }
        if model is not None:
            extract_kwargs["model"] = model
        elif config is not None:
            extract_kwargs["config"] = config
        else:
            raise ValueError("LangExtract request için model veya config gerekli.")
        prompt_validation_module = getattr(self._lx, "prompt_validation", None)
        prompt_validation_level = getattr(prompt_validation_module, "PromptValidationLevel", None)
        validation_off = getattr(prompt_validation_level, "OFF", None)
        if validation_off is not None:
            extract_kwargs["prompt_validation_level"] = validation_off

        return self._lx.extract(**extract_kwargs)

    @staticmethod
    def _extract_char_bounds(interval: Any) -> tuple[int | None, int | None]:
        """Normalize LangExtract character interval objects into primitive bounds."""
        if interval is None:
            return None, None

        start = getattr(interval, "start_pos", None)
        end = getattr(interval, "end_pos", None)
        if start is None and isinstance(interval, dict):
            start = interval.get("start_pos")
            end = interval.get("end_pos")
        elif start is None and isinstance(interval, (list, tuple)) and len(interval) >= 2:
            start, end = interval[0], interval[1]

        try:
            normalized_start = int(start) if start is not None else None
            normalized_end = int(end) if end is not None else None
        except (TypeError, ValueError):
            return None, None
        return normalized_start, normalized_end

    @classmethod
    def _extraction_sort_key(cls, extraction: Any) -> tuple[int, int, int]:
        """Sort extractions by grounded source order, then extraction index."""
        start_pos, end_pos = cls._extract_char_bounds(getattr(extraction, "char_interval", None))
        extraction_index = getattr(extraction, "extraction_index", None)
        normalized_index = extraction_index if isinstance(extraction_index, int) else 10**9
        return (
            start_pos if start_pos is not None else 10**9,
            end_pos if end_pos is not None else 10**9,
            normalized_index,
        )

    def _serialize_grounded_output(self, *, text: str, result: Any) -> str:
        """Serialize grounded LangExtract extractions into the existing JSON contract."""
        annotated_document = result[0] if isinstance(result, list) else result
        raw_extractions = getattr(annotated_document, "extractions", None) or []
        grounded_references: list[dict[str, str]] = []

        for extraction in sorted(raw_extractions, key=self._extraction_sort_key):
            start_pos, end_pos = self._extract_char_bounds(getattr(extraction, "char_interval", None))
            if start_pos is None or end_pos is None:
                continue
            if start_pos < 0 or end_pos > len(text) or start_pos >= end_pos:
                continue

            attributes = getattr(extraction, "attributes", None) or {}
            if not isinstance(attributes, dict):
                attributes = {}
            grounded_references.append(
                {
                    "type": REFERENCE_TYPE,
                    "kanun_no": str(attributes.get("kanun_no", "")).strip(),
                    "kanun_ad": str(attributes.get("kanun_ad", "")).strip(),
                    "madde": str(attributes.get("madde", "")).strip(),
                    "fikra": str(attributes.get("fikra", "")).strip(),
                    "bent": str(attributes.get("bent", "")).strip(),
                    "source_text": text[start_pos:end_pos],
                }
            )

        return json.dumps({"references": grounded_references}, ensure_ascii=False)

    def generate_content(self, *, text: str, system_prompt: str, model: str, temperature: float = 0.0) -> str:
        """Generate grounded JSON references via LangExtract."""
        variant = self._detect_prompt_variant(system_prompt)
        prompt_description = self._prompt_descriptions.get(variant, "")
        examples = self._examples_by_variant.get(variant, ())
        if not prompt_description:
            raise ValueError(f"LangExtract prompt description oluşturulamadı: {variant}")
        if not examples:
            raise ValueError(f"LangExtract örnek bankası boş: {variant}")

        backend, model_id = self._resolve_backend_and_model(model)
        langextract_model = self._build_langextract_model(
            backend=backend,
            model_id=model_id,
            temperature=temperature,
        )
        primary_config = None if langextract_model is not None else self._build_config(
            backend=backend,
            model_id=model_id,
            temperature=temperature,
        )

        try:
            result = self._run_langextract(
                text=text,
                prompt_description=prompt_description,
                examples=examples,
                config=primary_config,
                model=langextract_model,
                fence_output=(backend == "ollama"),
            )
        except Exception as primary_exc:
            if backend != "ollama":
                raise RuntimeError(
                    f"LangExtract {backend} backend extraction başarısız oldu: {primary_exc}"
                ) from primary_exc

            try:
                result = self._run_langextract(
                    text=text,
                    prompt_description=prompt_description,
                    examples=examples,
                    config=self._build_ollama_openai_fallback(model_id=model_id, temperature=temperature),
                    fence_output=True,
                )
            except Exception as fallback_exc:
                raise RuntimeError(
                    "LangExtract ollama backend extraction başarısız oldu. "
                    f"Native Ollama hatası: {primary_exc} | OpenAI-compatible fallback hatası: {fallback_exc}"
                ) from fallback_exc

        return self._serialize_grounded_output(text=text, result=result)
