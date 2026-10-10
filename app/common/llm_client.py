"""LiteLLM client wrapper with structured output, retries, and MockLLM – Rule / Phase 4."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel, ValidationError

from app.common.errors import LLMCallError, LLMCallFailure
from app.common.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "LLMCallError",
    "LLMCallFailure",
    "LLMClient",
    "LLMResponseMetadata",
    "MockLLM",
    "get_llm_usage_summary",
    "record_llm_usage",
]


@dataclass
class LLMResponseMetadata:
    """Metadata for an LLM generation."""

    raw_response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    model: str = ""
    latency_s: float = 0.0
    timestamp: datetime = datetime.now(UTC)


_GLOBAL_LLM_USAGE: dict[str, Any] = {
    "total_tokens": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "estimated_cost_usd": 0.0,
    "calls": 0,
    "by_model": {},
}


def record_llm_usage(meta: LLMResponseMetadata) -> None:
    """Record LLM call metadata into the global in-memory usage summary."""
    _GLOBAL_LLM_USAGE["total_tokens"] += meta.total_tokens
    _GLOBAL_LLM_USAGE["prompt_tokens"] += meta.prompt_tokens
    _GLOBAL_LLM_USAGE["completion_tokens"] += meta.completion_tokens
    _GLOBAL_LLM_USAGE["estimated_cost_usd"] += meta.estimated_cost
    _GLOBAL_LLM_USAGE["calls"] += 1

    m = meta.model or "unknown"
    if m not in _GLOBAL_LLM_USAGE["by_model"]:
        _GLOBAL_LLM_USAGE["by_model"][m] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
    entry = _GLOBAL_LLM_USAGE["by_model"][m]
    entry["calls"] += 1
    entry["prompt_tokens"] += meta.prompt_tokens
    entry["completion_tokens"] += meta.completion_tokens
    entry["total_tokens"] += meta.total_tokens
    entry["cost_usd"] += meta.estimated_cost


def get_llm_usage_summary() -> dict[str, Any]:
    """Return a snapshot of the recorded LLM usage summary."""
    return {
        "total_tokens": _GLOBAL_LLM_USAGE["total_tokens"],
        "prompt_tokens": _GLOBAL_LLM_USAGE["prompt_tokens"],
        "completion_tokens": _GLOBAL_LLM_USAGE["completion_tokens"],
        "estimated_cost_usd": round(_GLOBAL_LLM_USAGE["estimated_cost_usd"], 4),
        "calls": _GLOBAL_LLM_USAGE["calls"],
        "by_model": {
            k: {
                "calls": v["calls"],
                "prompt_tokens": v["prompt_tokens"],
                "completion_tokens": v["completion_tokens"],
                "total_tokens": v["total_tokens"],
                "cost_usd": round(v["cost_usd"], 4),
            }
            for k, v in _GLOBAL_LLM_USAGE["by_model"].items()
        },
    }


class LLMClient:
    """Asynchronous client for LLM completions using LiteLLM."""

    def __init__(
        self,
        model: str | None = None,
        timeout_s: int = 60,
        max_retries: int = 2,
    ) -> None:
        self.model = model or os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[T] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> tuple[T | None, LLMResponseMetadata]:
        """Execute a completion with retries, timeout, and structured output parsing."""
        target_model = model or self.model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        last_raw_response: str = ""
        attempts = 0

        actual_temp = temperature
        if "gemini" in target_model.lower() and actual_temp < 1.0:
            actual_temp = 1.0

        kwargs: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": actual_temp,
        }

        # Use response_format schema if supported
        if response_schema is not None:
            kwargs["response_format"] = response_schema

        for attempt in range(self.max_retries + 1):
            attempts += 1
            start_time = time.perf_counter()
            try:
                # Enforce timeout
                completion = await asyncio.wait_for(
                    litellm.acompletion(**kwargs),
                    timeout=float(self.timeout_s),
                )
                latency = time.perf_counter() - start_time

                content = completion.choices[0].message.content or ""
                last_raw_response = content

                # Token & cost attribution
                usage = getattr(completion, "usage", None)
                p_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                c_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                t_tokens = getattr(usage, "total_tokens", 0) if usage else 0

                try:
                    cost = float(litellm.completion_cost(completion_response=completion))
                except Exception:
                    cost = 0.0

                meta = LLMResponseMetadata(
                    raw_response=content,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=t_tokens,
                    estimated_cost=cost,
                    model=target_model,
                    latency_s=latency,
                    timestamp=datetime.now(UTC),
                )
                record_llm_usage(meta)

                if response_schema is None:
                    return None, meta

                # Parse & validate schema
                # Clean possible markdown code fences if model returned them
                cleaned_content = self._clean_json_markdown(content)
                parsed = response_schema.model_validate_json(cleaned_content)
                return parsed, meta

            except TimeoutError as exc:
                last_error = exc
                logger.warning(
                    "LLM call timed out after %ds (attempt %d/%d)",
                    self.timeout_s,
                    attempt + 1,
                    self.max_retries + 1,
                )
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "LLM output validation error: %s (attempt %d/%d)",
                    exc,
                    attempt + 1,
                    self.max_retries + 1,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM error: %s (attempt %d/%d)",
                    exc,
                    attempt + 1,
                    self.max_retries + 1,
                )

            if attempt < self.max_retries:
                await asyncio.sleep(0.5 * (2**attempt))

        # Automatic failover to backup model (e.g., Groq <-> Gemini)
        fallback_model = os.environ.get("FALLBACK_LLM_MODEL")
        if fallback_model and fallback_model != target_model:
            logger.warning(
                "Primary model %s failed after %d attempts (%s). Failing over to %s...",
                target_model,
                attempts,
                last_error,
                fallback_model,
            )
            fb_kwargs = dict(kwargs)
            fb_kwargs["model"] = fallback_model
            start_time = time.perf_counter()
            try:
                completion = await asyncio.wait_for(
                    litellm.acompletion(**fb_kwargs),
                    timeout=float(self.timeout_s),
                )
                latency = time.perf_counter() - start_time
                content = completion.choices[0].message.content or ""
                usage = getattr(completion, "usage", None)
                p_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                c_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                t_tokens = getattr(usage, "total_tokens", 0) if usage else 0

                try:
                    cost = float(litellm.completion_cost(completion_response=completion))
                except Exception:
                    cost = 0.0

                meta = LLMResponseMetadata(
                    raw_response=content,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=t_tokens,
                    estimated_cost=cost,
                    model=fallback_model,
                    latency_s=latency,
                    timestamp=datetime.now(UTC),
                )
                record_llm_usage(meta)

                if response_schema is None:
                    return None, meta

                cleaned_content = self._clean_json_markdown(content)
                parsed = response_schema.model_validate_json(cleaned_content)
                return parsed, meta
            except Exception as fb_exc:
                logger.error("Fallback model %s also failed: %s", fallback_model, fb_exc)

        raise LLMCallError(
            f"LLM completion failed after {attempts} attempts: {last_error}",
            raw_response=last_raw_response,
            attempts=attempts,
        )

    @staticmethod
    def _clean_json_markdown(text: str) -> str:
        """Strip markdown code block fences if present."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return text


class MockLLM(LLMClient):
    """Mock LLM returning scripted responses or raising errors for unit tests."""

    def __init__(
        self,
        canned_responses: list[Any] | None = None,
        timeout_s: int = 60,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model="mock", timeout_s=timeout_s, max_retries=max_retries)
        self._canned: list[Any] = list(canned_responses or [])
        self.call_history: list[dict[str, Any]] = []
        self.call_count: int = 0

    def add_response(self, response: Any) -> None:
        """Enqueue a response or exception."""
        self._canned.append(response)

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[T] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> tuple[T | None, LLMResponseMetadata]:
        attempts = 0
        last_error: Exception | None = None
        last_raw = ""

        for attempt in range(self.max_retries + 1):
            attempts += 1
            self.call_count += 1
            self.call_history.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "attempt": attempt + 1,
                }
            )

            resp_item = "{}" if not self._canned else self._canned.pop(0)

            if isinstance(resp_item, Exception):
                last_error = resp_item
                logger.warning("MockLLM simulating failure: %s", resp_item)
                if attempt < self.max_retries:
                    continue
                break

            if isinstance(resp_item, BaseModel):
                raw_str = resp_item.model_dump_json()
                parsed = resp_item
            elif isinstance(resp_item, dict):
                raw_str = json.dumps(resp_item)
                parsed = response_schema.model_validate(resp_item) if response_schema else None
            else:
                raw_str = str(resp_item)
                cleaned = self._clean_json_markdown(raw_str)
                try:
                    parsed = (
                        response_schema.model_validate_json(cleaned) if response_schema else None
                    )
                except Exception as exc:
                    last_error = exc
                    last_raw = raw_str
                    if attempt < self.max_retries:
                        continue
                    break

            meta = LLMResponseMetadata(
                raw_response=raw_str,
                prompt_tokens=150,
                completion_tokens=50,
                total_tokens=200,
                estimated_cost=0.0001,
                model="mock",
                latency_s=0.05,
                timestamp=datetime.now(UTC),
            )
            record_llm_usage(meta)
            return parsed, meta

        raise LLMCallError(
            f"MockLLM call failed after {attempts} attempts: {last_error}",
            raw_response=last_raw,
            attempts=attempts,
        )
