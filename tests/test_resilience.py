"""Comprehensive tests for the resilience module.

Test categories:
1. RateLimiter - basic behavior, lock-not-held-during-sleep, thread safety
2. get_random_browser_profile - returns valid profiles
3. RetryConfig - defaults
4. create_retry_decorator - retries on matching exceptions, no retry on non-matching
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from perplexity_web_mcp.resilience import (
    BROWSER_PROFILES,
    RetryConfig,
    RateLimiter,
    create_retry_decorator,
    get_random_browser_profile,
)


# ============================================================================
# 1. RateLimiter - Basic Behavior
# ============================================================================


class TestRateLimiterBasic:
    """Test RateLimiter basic acquire() behavior."""

    def test_first_acquire_no_wait(self) -> None:
        """First acquire() should not block."""
        limiter = RateLimiter(requests_per_second=10.0)
        t0 = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.05, "First acquire should return immediately"

    def test_rapid_acquires_respect_rate(self) -> None:
        """Two acquires in quick succession should enforce min interval."""
        limiter = RateLimiter(requests_per_second=10.0)  # 0.1s between requests
        limiter.acquire()

        t0 = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - t0

        # Should have waited ~0.1s (allow tolerance for scheduler)
        assert elapsed >= 0.08, f"Expected ~0.1s wait, got {elapsed}s"
        assert elapsed < 0.25, f"Should not wait excessively: {elapsed}s"

    def test_three_acquires_sequential(self) -> None:
        """Three acquires should take at least 2 * min_interval."""
        limiter = RateLimiter(requests_per_second=20.0)  # 0.05s between
        limiter.acquire()

        t0 = time.monotonic()
        limiter.acquire()
        limiter.acquire()
        elapsed = time.monotonic() - t0

        # 2 waits of ~0.05s each
        assert elapsed >= 0.08, f"Expected ~0.1s total wait, got {elapsed}s"


# ============================================================================
# 2. RateLimiter - Lock NOT Held During Sleep (Optimization)
# ============================================================================


class TestRateLimiterLockNotHeldDuringSleep:
    """Verify the optimization: lock is released before sleep so threads can overlap."""

    def test_concurrent_acquires_overlap_sleeps(self) -> None:
        """Multiple threads should sleep in parallel, not serialized on the lock.

        If the lock were held during sleep, N threads would take sum(wait_times).
        With lock released before sleep, they overlap and take ~max(wait_times).
        """
        # 100 req/s = 0.01s between requests
        limiter = RateLimiter(requests_per_second=100.0)
        num_threads = 5

        # If serialized: wait times would be 0, 0.01, 0.02, 0.03, 0.04 = 0.1s total
        # If parallel: max wait is 0.04s, so total wall time ~0.04-0.05s
        done = threading.Event()
        start_barrier = threading.Barrier(num_threads)

        def worker() -> None:
            start_barrier.wait()  # All threads start together
            limiter.acquire()
            done.set()

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = [ex.submit(worker) for _ in range(num_threads)]
            for f in as_completed(futures):
                f.result()
        elapsed = time.monotonic() - t0

        # Serialized would take ~0.1s. Parallel should be ~0.05s or less.
        assert elapsed < 0.08, (
            f"Elapsed {elapsed:.3f}s suggests lock held during sleep (expected <0.08s if parallel)"
        )


# ============================================================================
# 3. RateLimiter - Thread Safety
# ============================================================================


class TestRateLimiterThreadSafety:
    """Verify concurrent calls don't corrupt state."""

    def test_many_concurrent_acquires_no_crashes(self) -> None:
        """Many threads calling acquire() should not crash or corrupt state."""
        limiter = RateLimiter(requests_per_second=50.0)  # Fast for quick test
        num_threads = 10
        calls_per_thread = 5

        errors: list[Exception] = []
        completed = 0
        complete_lock = threading.Lock()

        def worker() -> None:
            nonlocal completed
            try:
                for _ in range(calls_per_thread):
                    limiter.acquire()
                with complete_lock:
                    completed += 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Thread errors: {errors}"
        assert completed == num_threads, f"Expected {num_threads} completions, got {completed}"

    def test_concurrent_acquires_respect_rate_overall(self) -> None:
        """Under load, effective rate should still be bounded."""
        limiter = RateLimiter(requests_per_second=20.0)  # 0.05s between
        num_calls = 10

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda _: limiter.acquire(), range(num_calls)))
        elapsed = time.monotonic() - t0

        # 10 calls at 20/s means 9 gaps of 0.05s = 0.45s min
        # With parallelism, first few may overlap; we expect at least ~0.3s
        assert elapsed >= 0.2, f"Too fast - rate may be violated: {elapsed}s"


# ============================================================================
# 4. get_random_browser_profile
# ============================================================================


class TestGetRandomBrowserProfile:
    """Test get_random_browser_profile() returns valid profiles."""

    def test_returns_valid_profile(self) -> None:
        """Single call returns a member of BROWSER_PROFILES."""
        profile = get_random_browser_profile()
        assert profile in BROWSER_PROFILES

    def test_returns_non_empty_string(self) -> None:
        """Profile is a non-empty string."""
        profile = get_random_browser_profile()
        assert isinstance(profile, str)
        assert len(profile) > 0

    def test_covers_all_profiles_over_many_calls(self) -> None:
        """Over many calls, all profiles should eventually appear (stochastic)."""
        seen: set[str] = set()
        for _ in range(200):  # High enough to likely hit all
            seen.add(get_random_browser_profile())
        assert seen == set(BROWSER_PROFILES), f"Missing profiles: {set(BROWSER_PROFILES) - seen}"


# ============================================================================
# 5. RetryConfig Defaults
# ============================================================================


class TestRetryConfig:
    """Test RetryConfig default values."""

    def test_defaults(self) -> None:
        """Default values match expected configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter == 0.5

    def test_custom_values(self) -> None:
        """Custom values are stored correctly."""
        config = RetryConfig(max_retries=5, base_delay=0.5, max_delay=30.0, jitter=0.3)
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.jitter == 0.3


# ============================================================================
# 6. create_retry_decorator
# ============================================================================


class TestCreateRetryDecorator:
    """Test create_retry_decorator retries on matching exceptions only."""

    def test_retries_on_matching_exception(self) -> None:
        """Decorated function retries when raising a retryable exception."""
        config = RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.05)
        retry_decorator = create_retry_decorator(
            config,
            retryable_exceptions=(ValueError,),
        )

        call_count = 0

        @retry_decorator
        def flaky_success() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "ok"

        result = flaky_success()
        assert result == "ok"
        assert call_count == 3  # Failed twice, succeeded on third

    def test_does_not_retry_on_non_matching_exception(self) -> None:
        """Decorated function does NOT retry when raising a non-retryable exception."""
        config = RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.05)
        retry_decorator = create_retry_decorator(
            config,
            retryable_exceptions=(ValueError,),
        )

        call_count = 0

        @retry_decorator
        def raises_type_error() -> str:
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError, match="not retryable"):
            raises_type_error()

        assert call_count == 1, "Should not retry on TypeError"

    def test_succeeds_immediately_no_retry(self) -> None:
        """When function succeeds on first call, no retries occur."""
        config = RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.05)
        retry_decorator = create_retry_decorator(
            config,
            retryable_exceptions=(ValueError,),
        )

        call_count = 0

        @retry_decorator
        def success_first_try() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = success_first_try()
        assert result == "ok"
        assert call_count == 1

    def test_exhausts_retries_then_reraise(self) -> None:
        """After max retries exhausted, exception is reraised."""
        config = RetryConfig(max_retries=1, base_delay=0.01, max_delay=0.05)
        retry_decorator = create_retry_decorator(
            config,
            retryable_exceptions=(ValueError,),
        )

        call_count = 0

        @retry_decorator
        def always_fails() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            always_fails()

        # max_retries=1 means 2 total attempts
        assert call_count == 2

    def test_on_retry_callback_invoked(self) -> None:
        """Optional on_retry callback is invoked before each retry sleep."""
        config = RetryConfig(max_retries=2, base_delay=0.01, max_delay=0.05)
        retry_count = 0

        def on_retry(state: object) -> None:
            nonlocal retry_count
            retry_count += 1

        retry_decorator = create_retry_decorator(
            config,
            retryable_exceptions=(ValueError,),
            on_retry=on_retry,
        )

        call_count = 0

        @retry_decorator
        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("retry")
            return "ok"

        result = flaky()
        assert result == "ok"
        assert retry_count == 2  # Called before 2nd and 3rd attempt
