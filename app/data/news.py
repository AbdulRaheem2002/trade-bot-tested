"""News provider interface and sanitization – Rule / Phase 2.

Provides:
- NewsProvider: Abstract base interface
- NullNewsProvider: Default no-op provider (used when news_enabled=false)
- sanitize_headline: Strips HTML, cuts to 200 chars, strips instruction-like injection patterns,
  and labels the headline as untrusted.
"""

from __future__ import annotations

import html
import re
from abc import ABC, abstractmethod

# Regex for stripping HTML tags
_HTML_TAG_RE = re.compile(r"<[^>]+>", re.IGNORECASE)

# Patterns that look like prompt injection or execution instructions
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"system\s*:", re.IGNORECASE),
    re.compile(r"assistant\s*:", re.IGNORECASE),
    re.compile(r"user\s*:", re.IGNORECASE),
    re.compile(r"you\s+must\s+(now\s+)?(ignore|override|execute|buy|sell)", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all\s+previous)", re.IGNORECASE),
    re.compile(r"new\s+instruction\s*:", re.IGNORECASE),
    re.compile(r"override\s+rules?", re.IGNORECASE),
    re.compile(r"execute\s+command", re.IGNORECASE),
    re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL),
]


def sanitize_headline(text: str) -> str:
    """Sanitize a raw news headline.

    1. Unescape HTML entities and strip HTML tags.
    2. Strip out prompt-injection patterns or instruction-like text.
    3. Collapse whitespace and truncate to at most 200 characters.
    4. Prefix with `[UNTRUSTED]` label.
    """
    if not text:
        return "[UNTRUSTED] "

    # 1. Strip HTML
    clean = html.unescape(text)
    clean = _HTML_TAG_RE.sub(" ", clean)

    # 2. Filter out injection / instruction patterns
    for pattern in _INJECTION_PATTERNS:
        clean = pattern.sub("[REDACTED_INSTRUCTION]", clean)

    # 3. Collapse whitespace
    clean = " ".join(clean.split())

    # 4. Truncate to 200 chars
    max_len = 200
    if len(clean) > max_len:
        clean = clean[:max_len].rstrip()

    # 5. Label as untrusted
    return f"[UNTRUSTED] {clean}"


def sanitize_headlines(headlines: list[str]) -> list[str]:
    """Sanitize a list of raw news headlines."""
    return [sanitize_headline(h) for h in headlines]


class NewsProvider(ABC):
    """Abstract interface for external market news providers."""

    @abstractmethod
    async def get_headlines(self, symbol: str) -> list[str]:
        """Fetch and return sanitized, untrusted headlines for *symbol*."""

    async def get_latest_headlines(
        self, symbols: list[str] | None = None, limit: int = 10
    ) -> list[str]:
        """Fetch sanitized headlines across multiple symbols."""
        results: list[str] = []
        if symbols:
            for s in symbols:
                h = await self.get_headlines(s)
                results.extend(h)
        return results[:limit]


class NullNewsProvider(NewsProvider):
    """Default no-op news provider used when news_enabled is false."""

    async def get_headlines(self, symbol: str) -> list[str]:
        """Return empty list without any network I/O."""
        return []
