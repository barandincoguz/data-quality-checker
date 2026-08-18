"""Provider registry for LLM-based reference extraction."""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from annotator.reference.gemini_aiplatform import generate_gemini_content, resolve_gemini_api_key
from annotator.reference.llm_prompt import DEFAULT_PROMPT_VARIANT

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 500

_NANOSECONDS_PER_SECOND = 1_000_000_000

# Highest OLLAMA_API_KEY_V<n> slot the collector will look for.
MAX_OLLAMA_KEY_SLOT = 12


@dataclass(frozen=True)
class GenerationUsage:
    """What one generation call consumed, as reported by the provider.

    Every field is optional because not every provider reports it. A field the
    provider did not send stays `None`; it is never coerced to `0`, because a
    zero would read as a measurement ("this call used no tokens") and would
    silently understate any cost computed from it.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_duration_seconds: float | None = None
    load_duration_seconds: float | None = None
    prompt_eval_duration_seconds: float | None = None
    eval_duration_seconds: float | None = None
    source: str = "unavailable"

    @classmethod
    def unavailable(cls) -> GenerationUsage:
        """Usage for a provider that reports nothing."""
        return cls()

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_duration_seconds": self.total_duration_seconds,
            "load_duration_seconds": self.load_duration_seconds,
            "prompt_eval_duration_seconds": self.prompt_eval_duration_seconds,
            "eval_duration_seconds": self.eval_duration_seconds,
            "source": self.source,
        }


@dataclass(frozen=True)
class GenerationResult:
    """A provider response together with what it cost to produce."""

    text: str
    usage: GenerationUsage = GenerationUsage()


class LlmProvider(ABC):
    """Abstract interface for model providers used by the benchmark."""

    name = "unknown"

    @abstractmethod
    def generate_content(self, *, text: str, system_prompt: str, model: str, temperature: float = 0.0) -> str:
        """Generate a raw JSON response string from the provider."""

    def generate_with_usage(
        self, *, text: str, system_prompt: str, model: str, temperature: float = 0.0
    ) -> GenerationResult:
        """Generate content and report provider-side token/duration usage.

        The default implementation delegates to `generate_content` and declares
        the usage unavailable, so a provider that cannot report token counts
        keeps working without change. Providers that can report them override
        this method.
        """
        text_out = self.generate_content(
            text=text, system_prompt=system_prompt, model=model, temperature=temperature
        )
        return GenerationResult(text=text_out, usage=GenerationUsage.unavailable())


def resolve_api_key(api_key: str | None = None) -> str:
    """Resolve the implicit or explicit API key from environment."""
    return resolve_gemini_api_key(api_key=api_key)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _ensure_dotenv_loaded() -> None:
    """Load .env file into os.environ if not already loaded (no-op if absent)."""
    path = _PROJECT_ROOT / ".env"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _collect_ollama_api_keys() -> list[str]:
    """Collect all available Ollama API keys from environment variables.

    Automatically loads .env file from the project root so keys don't need
    to be exported manually each session. Checks OLLAMA_API_KEY plus
    OLLAMA_API_KEY_V2 through OLLAMA_API_KEY_V<MAX_OLLAMA_KEY_SLOT> in order.
    Empty/missing slots are silently skipped, so adding more keys later only
    requires setting more env vars.

    The slot list is generated rather than hand-written: a hand-written list
    stopped at V7 while the environment already carried a V8, so one key sat
    unused and long runs exhausted rotation earlier than they had to.
    """
    _ensure_dotenv_loaded()
    env_names = ["OLLAMA_API_KEY"] + [
        f"OLLAMA_API_KEY_V{slot}" for slot in range(2, MAX_OLLAMA_KEY_SLOT + 1)
    ]
    keys: list[str] = []
    for name in env_names:
        val = os.environ.get(name, "").strip()
        if val:
            keys.append(val)
    if not keys:
        raise ValueError(
            "Hiçbir Ollama API key bulunamadı. "
            f"OLLAMA_API_KEY veya OLLAMA_API_KEY_V2..V{MAX_OLLAMA_KEY_SLOT} "
            "environment variable'larından en az birini set edin. "
            "(.env dosyası veya environment variable olarak)"
        )
    return keys


class GeminiProvider(LlmProvider):
    """Gemini adapter for reference extraction."""

    name = "gemini"

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        self._api_key = resolve_api_key(api_key=api_key)

    def generate_content(self, *, text: str, system_prompt: str, model: str, temperature: float = 0.0) -> str:
        return generate_gemini_content(
            text=text,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            api_key=self._api_key,
        )


class OllamaCloudProvider(LlmProvider):
    """Vanilla Ollama Cloud adapter for reference extraction.

    Supports automatic API key rotation: when a key hits its usage limit
    (HTTP 429 or usage-related errors), the provider switches to the next
    available key. Keys are read from OLLAMA_API_KEY, OLLAMA_API_KEY_V2,
    OLLAMA_API_KEY_V3, OLLAMA_API_KEY_V4 environment variables.
    """

    name = "ollama"

    # HTTP status codes that trigger key rotation without needing provider body text.
    _RATE_LIMIT_CODES = {429}
    _FATAL_OLLAMA_ERROR_KEYWORDS = (
        "requires a subscription",
        "upgrade for access",
        "payment required",
        "account suspended",
    )

    def __init__(self, api_key: str | None = None, key_env: str | None = None, **kwargs: Any) -> None:
        self._key_env = str(key_env).strip() if key_env else None
        if self._key_env:
            # A single named key, rotation off. Used to pin a run to one
            # billing tier: if a paid key could rotate onto a free one (or the
            # reverse) after a 403, the run could not say which tier produced
            # its timings, and the cost table would rest on a guess.
            _ensure_dotenv_loaded()
            value = os.environ.get(self._key_env, "").strip()
            if not value:
                raise ValueError(
                    f"{self._key_env} environment variable'ı boş veya tanımlı değil. "
                    "Adı verilen anahtar bulunamadığında sessizce havuza düşülmez."
                )
            self._api_keys: list[str] = [value]
        elif api_key:
            # Explicit key passed — use only that key (no rotation)
            self._api_keys = [api_key]
        else:
            self._api_keys = _collect_ollama_api_keys()
        self._rotation_enabled = self._key_env is None and not api_key
        self._current_key_index = 0
        self._base_url = str(os.environ.get("OLLAMA_BASE_URL", "https://ollama.com")).rstrip("/") or "https://ollama.com"
        timeout = kwargs.get("timeout") or os.environ.get("OLLAMA_TIMEOUT")
        try:
            self._timeout_seconds = int(timeout) if timeout is not None else DEFAULT_OLLAMA_TIMEOUT_SECONDS
        except (TypeError, ValueError):
            self._timeout_seconds = DEFAULT_OLLAMA_TIMEOUT_SECONDS

        key_count = len(self._api_keys)
        if self._key_env:
            print(f"[OllamaCloud] Tek anahtar kullanılıyor: {self._key_env}. Key rotation devre dışı.")
        else:
            print(
                f"[OllamaCloud] {key_count} API key yüklendi. "
                f"Otomatik key rotation {'aktif' if self.rotation_enabled and key_count > 1 else 'devre dışı'}."
            )

    @property
    def _api_key(self) -> str:
        return self._api_keys[self._current_key_index]

    @property
    def api_keys(self) -> list[str]:
        """The keys this provider may use, in rotation order."""
        return list(self._api_keys)

    @property
    def key_env(self) -> str | None:
        """Name of the single environment variable pinned for this run, if any."""
        return self._key_env

    @property
    def rotation_enabled(self) -> bool:
        return self._rotation_enabled

    def key_provenance(self) -> dict[str, Any]:
        """Which key source this run used — names and counts only, never values.

        Goes into the run summary so a cost figure can be traced to the billing
        tier that produced it.
        """
        return {
            "key_env": self._key_env,
            "key_count": len(self._api_keys),
            "rotation_enabled": self._rotation_enabled,
        }

    def _rotate_key(self) -> bool:
        """Rotate to the next available API key. Returns True if a new key is available."""
        if not self._rotation_enabled:
            return False
        next_index = self._current_key_index + 1
        if next_index < len(self._api_keys):
            self._current_key_index = next_index
            print(f"[OllamaCloud] API key rotasyonu: key {next_index + 1}/{len(self._api_keys)} kullanılıyor.")
            return True
        return False

    def _is_rate_limit_error(self, status_code: int, error_body: str) -> bool:
        """Check if the HTTP error indicates a rate/usage limit."""
        body = str(error_body or "").lower()
        if any(keyword in body for keyword in self._FATAL_OLLAMA_ERROR_KEYWORDS):
            return False
        if status_code in self._RATE_LIMIT_CODES:
            return True

        # Some providers return 400/402/403/500 with usage-related messages.
        rate_keywords = ["rate limit", "quota", "usage limit", "limit exceeded", "too many requests"]
        return any(kw in body for kw in rate_keywords)

    def generate_content(self, *, text: str, system_prompt: str, model: str, temperature: float = 0.0) -> str:
        return self.generate_with_usage(
            text=text, system_prompt=system_prompt, model=model, temperature=temperature
        ).text

    def generate_with_usage(
        self, *, text: str, system_prompt: str, model: str, temperature: float = 0.0
    ) -> GenerationResult:
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": float(temperature)},
        }

        data = json.dumps(payload).encode("utf-8")
        # Try current key, then rotate through remaining keys on rate-limit errors
        while True:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }

            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            _ssl_ctx = ssl.create_default_context()
            _ssl_ctx.check_hostname = False
            _ssl_ctx.verify_mode = ssl.CERT_NONE
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self._timeout_seconds,
                    context=_ssl_ctx,
                ) as response:
                    raw_body = response.read().decode("utf-8")
                return self._parse_response_with_usage(raw_body)
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if self._is_rate_limit_error(exc.code, error_body):
                    key_num = self._current_key_index + 1
                    print(f"[OllamaCloud] Key {key_num} kullanım limitine ulaştı (HTTP {exc.code}).")
                    if self._rotate_key():
                        continue  # Retry with next key
                    # All keys exhausted
                    raise RuntimeError(
                        f"Tüm Ollama API key'leri kullanım limitine ulaştı. "
                        f"Toplam {len(self._api_keys)} key denendi."
                    ) from exc
                else:
                    raise RuntimeError(
                        f"Ollama Cloud API çağrısı başarısız oldu (HTTP {exc.code}): {error_body}"
                    ) from exc

    @staticmethod
    def _parse_response(raw_body: str) -> str:
        """Parse Ollama Cloud response and return only the generated text."""
        return OllamaCloudProvider._parse_response_with_usage(raw_body).text

    @staticmethod
    def _duration_seconds(frame: dict, key: str) -> float | None:
        """Convert an Ollama nanosecond duration field to seconds, or None."""
        raw = frame.get(key)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return None
        return float(raw) / _NANOSECONDS_PER_SECOND

    @staticmethod
    def _token_count(frame: dict, key: str) -> int | None:
        """Read an Ollama token counter, or None when the provider omitted it."""
        raw = frame.get(key)
        if not isinstance(raw, int) or isinstance(raw, bool):
            return None
        return raw

    @classmethod
    def _extract_usage(cls, frames: list[dict]) -> GenerationUsage:
        """Pull token counts and durations out of the response frames.

        Under NDJSON streaming the counters arrive on the final frame, so the
        frames are scanned from the end and the first frame carrying a counter
        wins. Absent counters stay None rather than becoming zero.
        """
        for frame in reversed(frames):
            input_tokens = cls._token_count(frame, "prompt_eval_count")
            output_tokens = cls._token_count(frame, "eval_count")
            if input_tokens is None and output_tokens is None:
                continue
            return GenerationUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_duration_seconds=cls._duration_seconds(frame, "total_duration"),
                load_duration_seconds=cls._duration_seconds(frame, "load_duration"),
                prompt_eval_duration_seconds=cls._duration_seconds(frame, "prompt_eval_duration"),
                eval_duration_seconds=cls._duration_seconds(frame, "eval_duration"),
                source="ollama_chat",
            )
        return GenerationUsage.unavailable()

    @staticmethod
    def _parse_response_with_usage(raw_body: str) -> GenerationResult:
        """Parse Ollama Cloud response into generated text plus reported usage.

        Handles:
        - Single JSON object (stream=false)
        - NDJSON streaming (multiple JSON lines)
        - Thinking models where content may be empty but thinking is populated
        - OpenAI-compatible format (choices[0].message.content)
        - /api/generate format (response field)
        """
        lines = [ln.strip() for ln in raw_body.strip().splitlines() if ln.strip()]
        frames: list[dict] = []
        for line in lines:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    frames.append(obj)
            except json.JSONDecodeError:
                continue

        if not frames:
            raise RuntimeError(f"Ollama Cloud response parse edilemedi. Ham yanıt: {raw_body[:500]}")

        if frames[-1].get("error"):
            raise RuntimeError(f"Ollama Cloud API hatası: {frames[-1]['error']}")

        usage = OllamaCloudProvider._extract_usage(frames)

        # NDJSON streaming: pick frame with non-empty content, preferring later frames
        candidate = frames[-1]
        for frame in reversed(frames):
            msg = frame.get("message", {})
            if isinstance(msg, dict) and msg.get("content", "").strip():
                candidate = frame
                break

        # 1. Ollama native: message.content
        message = candidate.get("message", {}) if isinstance(candidate.get("message"), dict) else {}
        content = message.get("content", "").strip()
        if content:
            return GenerationResult(text=content, usage=usage)

        # 2. Thinking model: content empty but full text may be in accumulated frames
        accumulated = "".join(
            f.get("message", {}).get("content", "")
            for f in frames
            if isinstance(f.get("message"), dict)
        ).strip()
        if accumulated:
            return GenerationResult(text=accumulated, usage=usage)

        # 3. OpenAI-compatible: choices[0].message.content
        choices = candidate.get("choices", [])
        if choices and isinstance(choices[0], dict):
            oa_content = choices[0].get("message", {}).get("content", "").strip()
            if oa_content:
                return GenerationResult(text=oa_content, usage=usage)

        # 4. /api/generate format: response field
        gen_response = candidate.get("response", "").strip()
        if gen_response:
            return GenerationResult(text=gen_response, usage=usage)

        raise RuntimeError(
            f"Ollama Cloud response içinde content bulunamadı. "
            f"Ham yanıt (ilk 500 karakter): {raw_body[:500]}"
        )


def _lazy_langextract() -> Any:
    """Lazy import: LangExtract has optional heavier deps; imported only when selected.

    Returns LangExtractProvider, which duck-types LlmProvider rather than inheriting,
    hence the Any annotation.
    """
    from annotator.reference.langextract_provider import LangExtractProvider
    return LangExtractProvider


PROVIDER_REGISTRY: dict[str, Any] = {
    GeminiProvider.name: GeminiProvider,
    OllamaCloudProvider.name: OllamaCloudProvider,
    "langextract": _lazy_langextract,
}


def normalize_provider_name(provider_name: str) -> str:
    """Normalize provider identifiers."""
    normalized = str(provider_name or GeminiProvider.name).strip().lower()
    if normalized not in PROVIDER_REGISTRY:
        supported = ", ".join(sorted(PROVIDER_REGISTRY))
        raise ValueError(f"Unsupported LLM provider: {provider_name!r}. Supported: {supported}")
    return normalized


def get_provider(provider_name: str, **kwargs: Any) -> LlmProvider:
    """Instantiate a provider from the registry."""
    normalized = normalize_provider_name(provider_name)
    entry = PROVIDER_REGISTRY[normalized]
    cls = entry if isinstance(entry, type) else entry()
    return cls(**kwargs)


def _normalize_run_label(value: str) -> str:
    """Normalize optional run labels so filenames remain shell-safe and readable."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return normalized


def build_prediction_filename(
    provider_name: str,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
    run_label: str = "",
) -> str:
    """Return the standardized benchmark prediction filename for an LLM variant."""
    provider = normalize_provider_name(provider_name)
    normalized_variant = str(prompt_variant or DEFAULT_PROMPT_VARIANT).strip().lower().replace("-", "_")
    normalized_run_label = _normalize_run_label(run_label)
    if normalized_run_label:
        return f"{provider}_{normalized_variant}_{normalized_run_label}_reference_annotations.json"
    return f"{provider}_{normalized_variant}_reference_annotations.json"
