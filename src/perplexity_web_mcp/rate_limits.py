"""Rate limit checking and usage tracking via Perplexity internal REST API.

Uses undocumented endpoints:
- /rest/rate-limit/all: Current remaining quotas for all features
- /rest/user/settings: User settings, subscription info, file limits

These are the same endpoints the Perplexity web UI uses to display
usage counters and enforce limits client-side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Any

from curl_cffi.requests import Session

from .constants import API_BASE_URL, ENDPOINT_RATE_LIMITS, ENDPOINT_USER_SETTINGS, SESSION_COOKIE_NAME
from .logging import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SourceLimit:
    """Rate limit for a specific source (web, scholar, Google Drive, etc.)."""

    source_id: str
    monthly_limit: int | None = None
    remaining: int | None = None

    @property
    def is_unlimited(self) -> bool:
        return self.monthly_limit is None

    @property
    def is_exhausted(self) -> bool:
        return self.remaining is not None and self.remaining <= 0


@dataclass(frozen=True, slots=True)
class RateLimits:
    """Current rate limit status from /rest/rate-limit/all.

    Attributes:
        remaining_pro: Remaining Pro Search queries (weekly rolling).
        remaining_research: Remaining Deep Research queries (monthly).
        remaining_labs: Remaining Create Files & Apps queries (monthly).
        remaining_agentic_research: Remaining Browser Agent / Comet queries (monthly).
        model_specific_limits: Per-model limits (may be empty).
        source_limits: Per-source monthly limits.
    """

    remaining_pro: int = 0
    remaining_research: int = 0
    remaining_labs: int = 0
    remaining_agentic_research: int = 0
    model_specific_limits: dict[str, Any] = field(default_factory=dict)
    source_limits: list[SourceLimit] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> RateLimits:
        """Parse the /rest/rate-limit/all JSON response."""
        source_limits: list[SourceLimit] = []
        sources_data = data.get("sources", {}).get("source_to_limit", {})
        for source_id, limit_data in sources_data.items():
            source_limits.append(
                SourceLimit(
                    source_id=source_id,
                    monthly_limit=limit_data.get("monthly_limit"),
                    remaining=limit_data.get("remaining"),
                )
            )

        return cls(
            remaining_pro=data.get("remaining_pro", 0),
            remaining_research=data.get("remaining_research", 0),
            remaining_labs=data.get("remaining_labs", 0),
            remaining_agentic_research=data.get("remaining_agentic_research", 0),
            model_specific_limits=data.get("model_specific_limits", {}),
            source_limits=source_limits,
        )

    @property
    def has_pro_queries(self) -> bool:
        return self.remaining_pro > 0

    @property
    def has_research_queries(self) -> bool:
        return self.remaining_research > 0

    def format_summary(self) -> str:
        """Human-readable summary of current limits."""
        lines = [
            f"Pro Search: {self.remaining_pro} remaining",
            f"Deep Research: {self.remaining_research} remaining",
            f"Create Files & Apps: {self.remaining_labs} remaining",
            f"Browser Agent: {self.remaining_agentic_research} remaining",
        ]

        if self.model_specific_limits:
            lines.append(f"Model-specific limits: {self.model_specific_limits}")

        # Show limited sources (those with monthly caps)
        limited_sources = [s for s in self.source_limits if not s.is_unlimited]
        if limited_sources:
            lines.append("\nSource Limits:")
            for src in limited_sources:
                lines.append(f"  {src.source_id}: {src.remaining}/{src.monthly_limit}")

        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ConnectorLimits:
    """File and connector limits from user settings."""

    max_file_size_mb: int = 50
    daily_attachment_limit: int = 500
    weekly_attachment_limit: int | None = None
    global_file_count: int = 500


@dataclass(frozen=True, slots=True)
class UserSettings:
    """Relevant user settings from /rest/user/settings.

    Only exposes non-sensitive fields useful for limit enforcement.
    Excludes: connector credentials, OAuth tokens, referral codes.
    """

    pages_limit: int = 0
    upload_limit: int = 0
    create_limit: int = 0
    max_files_per_user: int = 0
    max_files_per_repository: int = 0
    subscription_status: str = "none"
    subscription_source: str = "none"
    subscription_tier: str = "none"
    query_count: int = 0
    query_count_copilot: int = 0
    default_model: str = "turbo"
    connector_limits: ConnectorLimits = field(default_factory=ConnectorLimits)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> UserSettings:
        """Parse the /rest/user/settings JSON response (safe fields only)."""
        cl_data = data.get("connector_limits", {})
        connector_limits = ConnectorLimits(
            max_file_size_mb=cl_data.get("max_file_size_mb", 50),
            daily_attachment_limit=cl_data.get("daily_attachment_limit", 500),
            weekly_attachment_limit=cl_data.get("weekly_attachment_limit"),
            global_file_count=cl_data.get("global_file_count", 500),
        )

        return cls(
            pages_limit=data.get("pages_limit", 0),
            upload_limit=data.get("upload_limit", 0),
            create_limit=data.get("create_limit", 0),
            max_files_per_user=data.get("max_files_per_user", 0),
            max_files_per_repository=data.get("max_files_per_repository", 0),
            subscription_status=data.get("subscription_status", "none"),
            subscription_source=data.get("subscription_source", "none"),
            subscription_tier=data.get("subscription_tier", "none"),
            query_count=data.get("query_count", 0),
            query_count_copilot=data.get("query_count_copilot", 0),
            default_model=data.get("default_model", "turbo"),
            connector_limits=connector_limits,
        )

    def format_summary(self) -> str:
        """Human-readable summary of user settings."""
        lines = [
            f"Subscription: {self.subscription_tier} ({self.subscription_status})",
            f"Total queries: {self.query_count:,}",
            f"Pro queries: {self.query_count_copilot:,}",
            f"Upload limit: {self.upload_limit} files",
            f"Create limit: {self.create_limit}",
            f"Pages limit: {self.pages_limit}",
            f"Max files/user: {self.max_files_per_user:,}",
            f"Max file size: {self.connector_limits.max_file_size_mb} MB",
            f"Daily attachments: {self.connector_limits.daily_attachment_limit}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fetching (low-level, no caching)
# ---------------------------------------------------------------------------

def _create_session(token: str) -> Session:
    """Create a minimal session for REST API calls."""
    return Session(
        impersonate="chrome",
        headers={"Referer": API_BASE_URL, "Origin": API_BASE_URL},
        cookies={SESSION_COOKIE_NAME: token},
    )


def fetch_rate_limits(token: str) -> RateLimits | None:
    """Fetch current rate limits from Perplexity.

    Returns None on any error (network, auth, parsing).
    """
    try:
        session = _create_session(token)
        response = session.get(f"{API_BASE_URL}{ENDPOINT_RATE_LIMITS}")
        if response.status_code == 200:
            return RateLimits.from_api(response.json())
        logger.warning(f"Rate limits fetch failed: HTTP {response.status_code}")
    except Exception as exc:
        logger.warning(f"Rate limits fetch error: {exc}")
    return None


def fetch_user_settings(token: str) -> UserSettings | None:
    """Fetch user settings from Perplexity.

    Returns None on any error (network, auth, parsing).
    """
    try:
        session = _create_session(token)
        response = session.get(f"{API_BASE_URL}{ENDPOINT_USER_SETTINGS}")
        if response.status_code == 200:
            return UserSettings.from_api(response.json())
        logger.warning(f"User settings fetch failed: HTTP {response.status_code}")
    except Exception as exc:
        logger.warning(f"User settings fetch error: {exc}")
    return None


# ---------------------------------------------------------------------------
# Cached layer (thread-safe, time-based)
# ---------------------------------------------------------------------------

class RateLimitCache:
    """Thread-safe, time-based cache for rate limit and settings data.

    - Rate limits: cached for ``rate_limit_ttl`` seconds (default 30s).
      Automatically invalidated after a query is made.
    - User settings: cached for ``settings_ttl`` seconds (default 300s / 5min).
    """

    __slots__ = (
        "_lock",
        "_rate_limits",
        "_rate_limits_ts",
        "_rate_limit_ttl",
        "_settings",
        "_settings_ts",
        "_settings_ttl",
        "_token",
    )

    def __init__(
        self,
        token: str,
        rate_limit_ttl: float = 30.0,
        settings_ttl: float = 300.0,
    ) -> None:
        self._token = token
        self._rate_limit_ttl = rate_limit_ttl
        self._settings_ttl = settings_ttl
        self._lock = Lock()
        self._rate_limits: RateLimits | None = None
        self._rate_limits_ts: float = 0.0
        self._settings: UserSettings | None = None
        self._settings_ts: float = 0.0

    def update_token(self, token: str) -> None:
        """Update the token (e.g. after re-authentication) and clear cache."""
        with self._lock:
            self._token = token
            self._rate_limits = None
            self._rate_limits_ts = 0.0
            self._settings = None
            self._settings_ts = 0.0

    def get_rate_limits(self, force_refresh: bool = False) -> RateLimits | None:
        """Get rate limits, fetching if cache is stale or empty."""
        now = monotonic()
        with self._lock:
            if (
                not force_refresh
                and self._rate_limits is not None
                and (now - self._rate_limits_ts) < self._rate_limit_ttl
            ):
                return self._rate_limits

        # Fetch outside lock to avoid blocking other threads
        limits = fetch_rate_limits(self._token)
        if limits is not None:
            with self._lock:
                self._rate_limits = limits
                self._rate_limits_ts = monotonic()
        return limits

    def get_user_settings(self, force_refresh: bool = False) -> UserSettings | None:
        """Get user settings, fetching if cache is stale or empty."""
        now = monotonic()
        with self._lock:
            if (
                not force_refresh
                and self._settings is not None
                and (now - self._settings_ts) < self._settings_ttl
            ):
                return self._settings

        settings = fetch_user_settings(self._token)
        if settings is not None:
            with self._lock:
                self._settings = settings
                self._settings_ts = monotonic()
        return settings

    def invalidate_rate_limits(self) -> None:
        """Invalidate rate limit cache (call after making a query)."""
        with self._lock:
            self._rate_limits = None
            self._rate_limits_ts = 0.0
