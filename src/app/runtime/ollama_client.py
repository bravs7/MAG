"""Local Ollama client for generation and embeddings."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

try:
    import ollama as ollama_sdk
except Exception:  # pragma: no cover - optional import fallback
    ollama_sdk = None


class OllamaClient:
    def __init__(self, host: str) -> None:
        self.host = host.rstrip("/")
        self._sdk_client = None
        if ollama_sdk is not None:
            self._sdk_client = ollama_sdk.Client(host=self.host)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        seed: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        options = {"temperature": temperature, "top_p": top_p}
        if seed is not None:
            options["seed"] = seed

        request_timeout = 120.0 if timeout_seconds is None else float(timeout_seconds)

        # SDK calls do not expose per-request timeout configuration reliably,
        # so for deterministic eval timeout behavior we force HTTP path.
        if self._sdk_client is not None and timeout_seconds is None:
            try:
                response = self._sdk_client.generate(
                    model=model,
                    prompt=prompt,
                    system=system,
                    options=options,
                    stream=False,
                )
                return str(response.get("response", "")).strip()
            except Exception:
                # Fall through to HTTP fallback.
                pass

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system

        response = _post_json(
            f"{self.host}/api/generate",
            payload,
            timeout_seconds=request_timeout,
        )
        return str(response.get("response", "")).strip()

    def embed_texts(
        self,
        *,
        model: str,
        texts: list[str],
        timeout_seconds: float | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        request_timeout = 120.0 if timeout_seconds is None else float(timeout_seconds)

        # SDK calls do not expose per-request timeout configuration reliably,
        # so for deterministic eval timeout behavior we force HTTP path.
        if self._sdk_client is not None and timeout_seconds is None:
            try:
                # Newer SDKs expose /api/embed as .embed().
                embed_response = self._sdk_client.embed(model=model, input=texts)
                embeddings = embed_response.get("embeddings")
                if embeddings:
                    return [list(map(float, item)) for item in embeddings]
            except Exception:
                pass

            try:
                # Older SDKs expose /api/embeddings one text per request.
                results: list[list[float]] = []
                for text in texts:
                    embed_response = self._sdk_client.embeddings(model=model, prompt=text)
                    vector = embed_response.get("embedding")
                    if vector is None:
                        raise RuntimeError("Missing embedding in Ollama SDK response")
                    results.append(list(map(float, vector)))
                return results
            except Exception:
                pass

        # HTTP fallback: try /api/embed then /api/embeddings
        try:
            response = _post_json(
                f"{self.host}/api/embed",
                {"model": model, "input": texts},
                timeout_seconds=request_timeout,
            )
            embeddings = response.get("embeddings")
            if embeddings:
                return [list(map(float, item)) for item in embeddings]
        except RuntimeError:
            pass

        vectors: list[list[float]] = []
        for text in texts:
            response = _post_json(
                f"{self.host}/api/embeddings",
                {"model": model, "prompt": text},
                timeout_seconds=request_timeout,
            )
            embedding = response.get("embedding")
            if embedding is None:
                raise RuntimeError("Missing embedding in /api/embeddings response")
            vectors.append(list(map(float, embedding)))

        return vectors


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except TimeoutError as exc:
        raise TimeoutError(
            f"Ollama request timeout after {timeout_seconds:.1f}s for {url}"
        ) from exc
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama HTTP error {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise TimeoutError(
                f"Ollama request timeout after {timeout_seconds:.1f}s for {url}"
            ) from exc
        if str(exc.reason).strip().lower() == "timed out":
            raise TimeoutError(
                f"Ollama request timeout after {timeout_seconds:.1f}s for {url}"
            ) from exc
        raise RuntimeError(f"Cannot reach Ollama at {url}: {exc.reason}") from exc
