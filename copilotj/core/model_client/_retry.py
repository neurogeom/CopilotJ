# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Retry primitives for the LLM model-client layer.

Pure functions + a hardcoded :data:`DEFAULT_RETRY_CONFIG`. The retry *loop*
itself lives in :class:`copilotj.core.agent.ChatAgent` (the UI boundary, so it
can emit ``update:retry`` events); this module only supplies the timing/decision
helpers so they stay IO-free and unit-testable.

Backoff policy:
    - If the provider returned a ``Retry-After`` header (delta-seconds or an
      HTTP-date), honour it (clamped to ``max_backoff``).
    - Otherwise exponential backoff ``initial * 2**(attempt-1)`` with full
      jitter, clamped to ``max_backoff``.
"""

from __future__ import annotations

import email.utils
import random
from dataclasses import dataclass

__all__ = ["RetryConfig", "DEFAULT_RETRY_CONFIG", "parse_retry_after", "compute_backoff"]


@dataclass(frozen=True)
class RetryConfig:
    """Hardcoded retry tuning for 429 auto-retry (#96).

    Not exposed via env vars by design — see the plan's locked decisions. Bump
    the defaults here if they ever need to move.
    """

    max_attempts: int = 5
    initial_backoff: float = 1.0
    max_backoff: float = 60.0


# Single source of truth consumed by ``ChatAgent._create``.
DEFAULT_RETRY_CONFIG = RetryConfig()


def parse_retry_after(header: str | None) -> float | None:
    """Parse an HTTP ``Retry-After`` header into seconds.

    Accepts delta-seconds (``"30"``) or an HTTP-date
    (``"Wed, 21 Oct 2015 07:28:00 GMT"``). Returns ``None`` when the header is
    missing or unparseable, so callers fall back to exponential backoff.
    """
    if header is None:
        return None
    header = header.strip()
    if not header:
        return None

    # delta-seconds — the common case for rate limiting.
    try:
        seconds = float(header)
    except ValueError:
        pass
    else:
        return seconds if seconds >= 0 else None

    # HTTP-date form.
    try:
        target = email.utils.parsedate_to_datetime(header)
    except (TypeError, ValueError):
        return None
    if target is None:
        return None

    now = email.utils.parsedate_to_datetime(email.utils.formatdate(usegmt=True))
    if now is None:
        return None
    return max((target - now).total_seconds(), 0.0)


def compute_backoff(retry_after: float | None, attempt: int, cfg: RetryConfig) -> float:
    """Seconds to wait before the next retry attempt.

    ``retry_after`` (from the provider's ``Retry-After`` header) wins when
    present. Otherwise exponential backoff with full jitter, clamped to
    ``cfg.max_backoff``. ``attempt`` is 1-based: attempt 1 is the first failure,
    so the first exponential backoff is ``initial_backoff``.
    """
    if retry_after is not None:
        return min(retry_after, cfg.max_backoff)

    base = min(cfg.initial_backoff * (2 ** (attempt - 1)), cfg.max_backoff)
    # Full jitter in [base/2, base]: avoids synchronized retry storms across
    # concurrent agents without ever exceeding the cap.
    return random.uniform(base / 2, base)
