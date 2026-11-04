"""Provider-agnostic chat completion wrapper.

Supports OpenAI, Anthropic, Google Gemini, and Ollama. All providers are
called with temperature zero and JSON-constrained
output where the API supports it.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

PROVIDER_CHOICES = ["anthropic", "openai", "gemini", "ollama"]


@dataclass(frozen=True)
class CompletionResult:
    text: str
    provider: str
    model: str
    request_id: str | None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    latency_seconds: float
    token_logprobs: list[dict[str, Any]] | None = None

_TEMPERATURE = 0.0


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini-2024-07-18",
    json_output: bool = False,
    max_tokens: int = 4096,
) -> str:
    """Run a chat completion and return only its text."""
    return chat_completion_with_metadata(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        provider=provider,
        model=model,
        json_output=json_output,
        max_tokens=max_tokens,
    ).text


def chat_completion_with_metadata(
    system_prompt: str,
    user_prompt: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini-2024-07-18",
    json_output: bool = False,
    max_tokens: int = 4096,
    json_schema: dict[str, Any] | None = None,
    include_logprobs: bool = False,
    top_logprobs: int = 0,
) -> CompletionResult:
    """Run a completion and retain provider usage and request metadata."""
    provider = provider.lower()
    if include_logprobs and provider != "openai":
        raise ValueError("Token log probabilities are currently supported only for OpenAI.")

    if provider == "openai":
        return _openai(
            system_prompt,
            user_prompt,
            model,
            json_output,
            max_tokens,
            json_schema,
            include_logprobs,
            top_logprobs,
        )
    if provider == "anthropic":
        return _anthropic(system_prompt, user_prompt, model, json_output, max_tokens)
    if provider == "gemini":
        return _gemini(system_prompt, user_prompt, model, json_output, max_tokens, json_schema)
    if provider == "ollama":
        return _ollama(
            system_prompt,
            user_prompt,
            model,
            json_output,
            max_tokens,
            json_schema,
        )
    raise ValueError(
        f"Unsupported provider: {provider!r}. Choose one of {PROVIDER_CHOICES}."
    )


def _openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    json_output: bool,
    max_tokens: int,
    json_schema: dict[str, Any] | None,
    include_logprobs: bool,
    top_logprobs: int,
) -> CompletionResult:
    import openai  # local import to keep optional deps lazy

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if json_schema is not None:
        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "evaluation_response",
                "strict": True,
                "schema": json_schema,
            },
        }
    else:
        response_format = {"type": "json_object" if json_output else "text"}

    args: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": response_format,
    }
    if include_logprobs:
        args["logprobs"] = True
        args["top_logprobs"] = top_logprobs

    is_reasoning = model.startswith(("o", "gpt-5"))
    if is_reasoning:
        args["max_completion_tokens"] = max(max_tokens, 16000)
    else:
        args["temperature"] = _TEMPERATURE
        args["max_tokens"] = max_tokens

    started = time.perf_counter()
    response = client.chat.completions.create(**args)
    latency = time.perf_counter() - started
    usage = response.usage
    token_logprobs = None
    if include_logprobs and response.choices[0].logprobs is not None:
        token_logprobs = [item.model_dump() for item in response.choices[0].logprobs.content or []]
    return CompletionResult(
        text=response.choices[0].message.content or "",
        provider="openai",
        model=response.model,
        request_id=response.id,
        input_tokens=int(usage.prompt_tokens if usage else 0),
        cached_input_tokens=int(
            usage.prompt_tokens_details.cached_tokens
            if usage and usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens
            else 0
        ),
        output_tokens=int(usage.completion_tokens if usage else 0),
        latency_seconds=latency,
        token_logprobs=token_logprobs,
    )


def _anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
    json_output: bool,
    max_tokens: int,
) -> CompletionResult:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if not model.startswith("claude"):
        model = "claude-3-5-haiku-20241022"
    system = system_prompt
    if json_output:
        system = (
            system_prompt
            + "\n\nIMPORTANT: Respond with valid JSON only. "
            "Do not include any text before or after the JSON object."
        )
    started = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=_TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    latency = time.perf_counter() - started
    usage = response.usage
    cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    return CompletionResult(
        text=response.content[0].text,
        provider="anthropic",
        model=response.model,
        request_id=response.id,
        input_tokens=int(usage.input_tokens),
        cached_input_tokens=cached,
        output_tokens=int(usage.output_tokens),
        latency_seconds=latency,
    )


def _gemini(
    system_prompt: str,
    user_prompt: str,
    model: str,
    json_output: bool,
    max_tokens: int,
    json_schema: dict[str, Any] | None,
) -> CompletionResult:
    """Call Gemini via either an API key or a Vertex AI service account."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not project or not creds_path:
            raise RuntimeError(
                "Gemini provider requires either GEMINI_API_KEY (or "
                "GOOGLE_API_KEY), or Vertex AI credentials "
                "(GOOGLE_APPLICATION_CREDENTIALS pointing to a service-account "
                "JSON file, plus GOOGLE_CLOUD_PROJECT)."
            )
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        client = genai.Client(vertexai=True, project=project, location=location)

    if not model.startswith("gemini"):
        model = "gemini-2.5-flash-lite"
    config: dict[str, Any] = {
        "temperature": _TEMPERATURE,
        "max_output_tokens": max_tokens,
    }
    if json_output or json_schema is not None:
        config["response_mime_type"] = "application/json"
    if json_schema is not None:
        config["response_schema"] = json_schema
    combined = f"{system_prompt}\n\n{user_prompt}"
    started = time.perf_counter()
    response = client.models.generate_content(model=model, contents=combined, config=config)
    latency = time.perf_counter() - started
    usage = response.usage_metadata
    return CompletionResult(
        text=response.text or "",
        provider="gemini",
        model=model,
        request_id=getattr(response, "response_id", None),
        input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
        cached_input_tokens=int(getattr(usage, "cached_content_token_count", 0) or 0),
        output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        latency_seconds=latency,
    )


def _ollama(
    system_prompt: str,
    user_prompt: str,
    model: str,
    json_output: bool,
    max_tokens: int,
    json_schema: dict[str, Any] | None,
) -> CompletionResult:
    import ollama

    started = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format=json_schema if json_schema is not None else ("json" if json_output else None),
        options={"temperature": _TEMPERATURE, "num_predict": max_tokens},
    )
    latency = time.perf_counter() - started
    return CompletionResult(
        text=response["message"]["content"],
        provider="ollama",
        model=model,
        request_id=None,
        input_tokens=int(response.get("prompt_eval_count", 0) or 0),
        cached_input_tokens=0,
        output_tokens=int(response.get("eval_count", 0) or 0),
        latency_seconds=latency,
    )
