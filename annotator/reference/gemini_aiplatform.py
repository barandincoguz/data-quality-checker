"""Shared Gemini REST helper using the repo's Vertex-style aiplatform endpoint."""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def resolve_gemini_api_key(api_key: str | None = None) -> str:
    """Resolve the implicit or explicit Gemini API key from environment."""
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not resolved_key:
        raise ValueError("GEMINI_API_KEY environment variable'ı set edilmemiş.")
    return resolved_key


def extract_gemini_response_text(response: dict[str, Any]) -> str:
    """Extract deterministic text output from a REST JSON response."""
    candidates = response.get("candidates", [])
    if not candidates:
        error_details = response.get("error", "Unknown API Error")
        raise RuntimeError(f"Gemini response içinde candidate bulunamadı. Detay: {error_details}")

    first_candidate = candidates[0]
    content = first_candidate.get("content", {})
    parts = content.get("parts", [])
    text_chunks: list[str] = []

    for part in parts:
        part_text = part.get("text")
        if isinstance(part_text, str):
            text_chunks.append(part_text)

    text = "".join(text_chunks).strip()
    if text:
        return text

    raise RuntimeError("Gemini response text parçaları boş döndü.")


def _build_ssl_context() -> ssl.SSLContext:
    """Return a verification context that survives missing system CA bundles.

    Some local Python installs point OpenSSL at a non-existent default CA file.
    In that case, fall back to certifi when it is available in the active venv.
    """
    env_cafile = os.environ.get("SSL_CERT_FILE", "").strip()
    if env_cafile and Path(env_cafile).is_file():
        return ssl.create_default_context(cafile=env_cafile)

    verify_paths = ssl.get_default_verify_paths()
    default_cafile = verify_paths.cafile or verify_paths.openssl_cafile
    if default_cafile and Path(default_cafile).is_file():
        return ssl.create_default_context()

    try:
        import certifi
    except Exception:
        return ssl.create_default_context()

    certifi_cafile = certifi.where()
    if certifi_cafile and Path(certifi_cafile).is_file():
        return ssl.create_default_context(cafile=certifi_cafile)
    return ssl.create_default_context()


def generate_gemini_content(
    *,
    text: str,
    model: str,
    temperature: float = 0.0,
    system_prompt: str = "",
    api_key: str | None = None,
) -> str:
    """Call the repo's working Gemini REST path and return the raw text output."""
    resolved_api_key = resolve_gemini_api_key(api_key=api_key)
    url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent"

    payload: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": text,
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "responseMimeType": "application/json",
        },
    }
    if str(system_prompt).strip():
        payload["systemInstruction"] = {
            "parts": [
                {
                    "text": system_prompt,
                }
            ]
        }

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": resolved_api_key,
    }

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, context=_build_ssl_context()) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        raise RuntimeError(f"Vertex AI API çağrısı başarısız oldu (HTTP {exc.code}): {error_body}") from exc

    return extract_gemini_response_text(response_data)
