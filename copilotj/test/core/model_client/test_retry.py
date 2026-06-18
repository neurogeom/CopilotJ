# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure retry primitives in copilotj.core.model_client._retry."""

import email.utils

from copilotj.core.model_client._retry import (
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    compute_backoff,
    parse_retry_after,
)
from copilotj.core.model_client._types import ModelProviderError


def test_default_retry_config_sane():
    assert DEFAULT_RETRY_CONFIG.max_attempts >= 1
    assert DEFAULT_RETRY_CONFIG.initial_backoff > 0
    assert DEFAULT_RETRY_CONFIG.max_backoff >= DEFAULT_RETRY_CONFIG.initial_backoff


def test_is_retryable():
    assert ModelProviderError("x", "openai", status_code=429).is_retryable()
    assert not ModelProviderError("x", "openai", status_code=400).is_retryable()
    assert not ModelProviderError("x", "openai", status_code=401).is_retryable()
    # Missing status code (provider forgot to tag) → never retry blindly.
    assert not ModelProviderError("x", "openai").is_retryable()
    # Custom status set (e.g. if 5xx is ever enabled).
    assert ModelProviderError("x", "openai", status_code=503).is_retryable(retry_statuses=(429, 503))


def test_parse_retry_after_seconds():
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("   ") is None
    assert parse_retry_after("30") == 30.0
    assert parse_retry_after("0") == 0.0
    assert parse_retry_after("-5") is None  # negative makes no sense


def test_parse_retry_after_http_date():
    # A date ~60s in the future.
    import datetime as _dt

    future = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=60)
    header = email.utils.format_datetime(future, usegmt=True)
    seconds = parse_retry_after(header)
    assert seconds is not None
    assert 50 <= seconds <= 70


def test_parse_retry_after_garbage():
    assert parse_retry_after("not-a-date-or-number") is None


def test_compute_backoff_honours_retry_after():
    cfg = RetryConfig(max_attempts=5, initial_backoff=1.0, max_backoff=60.0)
    assert compute_backoff(5.0, 1, cfg) == 5.0
    # Clamped to max_backoff.
    assert compute_backoff(999.0, 1, cfg) == 60.0
    assert compute_backoff(0.0, 1, cfg) == 0.0


def test_compute_backoff_exponential_within_jitter_range_and_capped():
    cfg = RetryConfig(max_attempts=10, initial_backoff=1.0, max_backoff=60.0)
    # attempt 1 → base 1.0 → jitter in [0.5, 1.0]
    v = compute_backoff(None, 1, cfg)
    assert 0.5 <= v <= 1.0
    # attempt 3 → base 4.0 → jitter in [2.0, 4.0]
    v = compute_backoff(None, 3, cfg)
    assert 2.0 <= v <= 4.0
    # attempt 10 → base would be 512, capped at 60 → jitter in [30, 60]
    v = compute_backoff(None, 10, cfg)
    assert 30.0 <= v <= 60.0
