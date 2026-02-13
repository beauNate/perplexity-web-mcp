"""MCP server implementation using FastMCP."""

from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP

from perplexity_web_mcp.config import ClientConfig, ConversationConfig
from perplexity_web_mcp.core import Perplexity
from perplexity_web_mcp.enums import CitationMode, SearchFocus, SourceFocus
from perplexity_web_mcp.models import Model, Models
from perplexity_web_mcp.rate_limits import RateLimitCache, RateLimits
from perplexity_web_mcp.token_store import get_token_or_raise, load_token, save_token


mcp = FastMCP(
    "perplexity-web-mcp",
    instructions=(
        "Search the web with Perplexity AI using premium models. "
        "Use pplx_query for flexible model selection with thinking toggle. "
        "Or use model-specific tools like pplx_gpt52, pplx_claude_sonnet, etc. "
        "All tools support source_focus: web, academic, social, finance, all. "
        "\n\n"
        "USAGE LIMITS: Call pplx_usage before heavy use to check remaining quotas. "
        "The server checks limits automatically and will warn you before queries "
        "that would exceed your plan's allowance.\n\n"
        "AUTHENTICATION: If you get a 403 error or 'token expired' message, use these tools to re-authenticate:\n"
        "1. pplx_auth_status - Check current authentication status\n"
        "2. pplx_auth_request_code - Send verification code to email (requires email address)\n"
        "3. pplx_auth_complete - Complete auth with the 6-digit code from email\n"
        "Session tokens last ~30 days. After re-authenticating, all pplx_* tools will work again."
    ),
)

SOURCE_FOCUS_MAP = {
    "web": [SourceFocus.WEB],
    "academic": [SourceFocus.ACADEMIC],
    "social": [SourceFocus.SOCIAL],
    "finance": [SourceFocus.FINANCE],
    "all": [SourceFocus.WEB, SourceFocus.ACADEMIC, SourceFocus.SOCIAL],
}

# Model name to Model mapping (supports thinking toggle)
MODEL_MAP: dict[str, tuple[Model, Model | None]] = {
    # (base_model, thinking_model) - None if no thinking variant
    "auto": (Models.BEST, None),
    "sonar": (Models.SONAR, None),
    "deep_research": (Models.DEEP_RESEARCH, None),
    "gpt52": (Models.GPT_52, Models.GPT_52_THINKING),
    "claude_sonnet": (Models.CLAUDE_45_SONNET, Models.CLAUDE_45_SONNET_THINKING),
    "claude_opus": (Models.CLAUDE_46_OPUS, Models.CLAUDE_46_OPUS_THINKING),
    "gemini_flash": (Models.GEMINI_3_FLASH, Models.GEMINI_3_FLASH_THINKING),
    "gemini_pro": (Models.GEMINI_3_PRO_THINKING, Models.GEMINI_3_PRO_THINKING),  # Only thinking variant
    "grok": (Models.GROK_41, Models.GROK_41_THINKING),
    "kimi": (Models.KIMI_K25_THINKING, Models.KIMI_K25_THINKING),  # Only thinking variant
}

SourceFocusName = Literal["web", "academic", "social", "finance", "all"]
ModelName = Literal["auto", "sonar", "deep_research", "gpt52", "claude_sonnet", "claude_opus", "gemini_flash", "gemini_pro", "grok", "kimi"]

def _get_client() -> Perplexity:
    """Create a fresh Perplexity client for each request.
    
    We don't cache the client because:
    1. Token can change after re-authentication
    2. curl_cffi Sessions can have stale state
    3. MCP server processes may restart between calls
    """
    token = get_token_or_raise()
    # Use minimal config to avoid session issues
    config = ClientConfig(
        rotate_fingerprint=False,  # Don't rotate to avoid state issues
        requests_per_second=0,     # Disable rate limiting (MCP handles this)
    )
    return Perplexity(token, config=config)


# ---------------------------------------------------------------------------
# Rate limit cache (persistent across requests, refreshes on token change)
# ---------------------------------------------------------------------------
_limit_cache: RateLimitCache | None = None
_limit_cache_token: str | None = None


def _get_limit_cache() -> RateLimitCache | None:
    """Get or create the rate limit cache for the current token."""
    global _limit_cache, _limit_cache_token

    token = load_token()
    if not token:
        return None

    if _limit_cache is None or _limit_cache_token != token:
        _limit_cache = RateLimitCache(token)
        _limit_cache_token = token

    return _limit_cache


def _is_research_model(model: Model) -> bool:
    """Check if the model is Deep Research (uses research quota)."""
    return model is Models.DEEP_RESEARCH


def _check_limits_before_query(model: Model) -> str | None:
    """Check rate limits before executing a query.
    
    Returns an error message string if limits are exceeded, None if OK.
    Does not block the query on cache miss / fetch failure (fail-open).
    """
    cache = _get_limit_cache()
    if cache is None:
        return None  # No token, will fail at auth stage

    limits = cache.get_rate_limits()
    if limits is None:
        return None  # Fetch failed, fail-open

    if _is_research_model(model):
        if not limits.has_research_queries:
            return (
                f"LIMIT REACHED: Deep Research queries exhausted "
                f"(0 remaining).\n\n"
                f"Current usage:\n{limits.format_summary()}\n\n"
                f"Deep Research limits reset monthly. "
                f"Use pplx_ask or another model for standard Pro Search instead."
            )
    else:
        if not limits.has_pro_queries:
            return (
                f"LIMIT REACHED: Pro Search queries exhausted "
                f"(0 remaining).\n\n"
                f"Current usage:\n{limits.format_summary()}\n\n"
                f"Pro Search limits reset weekly. "
                f"Consider waiting or upgrading your plan."
            )

    return None


def _get_limit_context_for_error() -> str:
    """Get rate limit context to include in error messages."""
    cache = _get_limit_cache()
    if cache is None:
        return ""

    limits = cache.get_rate_limits()
    if limits is None:
        return ""

    return f"\nCurrent usage:\n{limits.format_summary()}\n"


def _ask(query: str, model: Model, source_focus: SourceFocusName = "web") -> str:
    """Execute a query with a specific model."""

    # Pre-flight limit check
    limit_error = _check_limits_before_query(model)
    if limit_error:
        return limit_error

    client = _get_client()
    sources = SOURCE_FOCUS_MAP.get(source_focus, [SourceFocus.WEB])

    try:
        conversation = client.create_conversation(
            ConversationConfig(
                model=model,
                citation_mode=CitationMode.DEFAULT,
                search_focus=SearchFocus.WEB,
                source_focus=sources,
            )
        )

        conversation.ask(query)

        # Invalidate rate limit cache after successful query
        cache = _get_limit_cache()
        if cache:
            cache.invalidate_rate_limits()

        answer = conversation.answer or "No answer received"

        response_parts = [answer]

        if conversation.search_results:
            response_parts.append("\n\nCitations:")

            for i, result in enumerate(conversation.search_results, 1):
                url = result.url or ""
                response_parts.append(f"\n[{i}]: {url}")

        return "".join(response_parts)

    except Exception as error:
        error_str = str(error)
        error_type = type(error).__name__

        # Invalidate cache on error too (state may have changed)
        cache = _get_limit_cache()
        if cache:
            cache.invalidate_rate_limits()

        # Check if token actually exists and is valid
        from perplexity_web_mcp.cli.auth import get_user_info
        token = load_token()
        token_status = "No token found"
        if token:
            user_info = get_user_info(token)
            if user_info:
                token_status = f"Token valid for {user_info.email}"
            else:
                token_status = "Token exists but invalid"

        limit_context = _get_limit_context_for_error()

        if "429" in error_str or "rate limit" in error_str.lower():
            return (
                f"Error: Rate limit exceeded (429).\n\n"
                f"Token status: {token_status}\n"
                f"{limit_context}\n"
                f"Wait a few minutes before retrying. "
                f"Call pplx_usage to check your current limits."
            )

        if "403" in error_str or "forbidden" in error_str.lower():
            return (
                f"Error: Access forbidden (403).\n\n"
                f"Token status: {token_status}\n"
                f"Error type: {error_type}\n"
                f"Error details: {error_str}\n"
                f"{limit_context}\n"
                f"This may be a Perplexity API issue. If token shows as valid above, "
                f"try waiting a few seconds and retry. If persistent, re-authenticate:\n"
                f"1. Call pplx_auth_request_code(email='YOUR_EMAIL')\n"
                f"2. Check email for 6-digit code\n"
                f"3. Call pplx_auth_complete(email='YOUR_EMAIL', code='XXXXXX')"
            )
        return f"Error ({error_type}): {error_str}"


@mcp.tool
def pplx_query(
    query: str,
    model: ModelName = "auto",
    thinking: bool = False,
    source_focus: SourceFocusName = "web",
) -> str:
    """Query Perplexity AI with model selection and thinking toggle.
    
    Args:
        query: The question to ask
        model: Model to use - auto, sonar, deep_research, gpt52, claude_sonnet, 
               claude_opus, gemini_flash, gemini_pro, grok, kimi
        thinking: Enable extended thinking mode (available for gpt52, claude_sonnet, 
                  claude_opus, gemini_flash, grok)
        source_focus: Source type - web, academic, social, finance, all
    """
    model_tuple = MODEL_MAP.get(model, (Models.BEST, None))
    base_model, thinking_model = model_tuple
    
    # Use thinking model if requested and available
    selected_model = thinking_model if thinking and thinking_model else base_model
    
    return _ask(query, selected_model, source_focus)


@mcp.tool
def pplx_ask(query: str, source_focus: SourceFocusName = "web") -> str:
    """Ask a question with real-time data from the internet (auto-selects best model)."""

    return _ask(query, Models.BEST, source_focus)


@mcp.tool
def pplx_deep_research(query: str, source_focus: SourceFocusName = "web") -> str:
    """Deep Research - In-depth reports with more sources, charts, and advanced reasoning."""

    return _ask(query, Models.DEEP_RESEARCH, source_focus)


@mcp.tool
def pplx_sonar(query: str, source_focus: SourceFocusName = "web") -> str:
    """Sonar - Perplexity's latest model."""

    return _ask(query, Models.SONAR, source_focus)


@mcp.tool
def pplx_gpt52(query: str, source_focus: SourceFocusName = "web") -> str:
    """GPT-5.2 - OpenAI's latest model."""

    return _ask(query, Models.GPT_52, source_focus)


@mcp.tool
def pplx_gpt52_thinking(query: str, source_focus: SourceFocusName = "web") -> str:
    """GPT-5.2 Thinking - OpenAI's latest model with extended thinking."""

    return _ask(query, Models.GPT_52_THINKING, source_focus)


@mcp.tool
def pplx_claude_sonnet(query: str, source_focus: SourceFocusName = "web") -> str:
    """Claude Sonnet 4.5 - Anthropic's fast model."""

    return _ask(query, Models.CLAUDE_45_SONNET, source_focus)


@mcp.tool
def pplx_claude_sonnet_think(query: str, source_focus: SourceFocusName = "web") -> str:
    """Claude Sonnet 4.5 Thinking - Anthropic's fast model with extended thinking."""

    return _ask(query, Models.CLAUDE_45_SONNET_THINKING, source_focus)


@mcp.tool
def pplx_gemini_flash(query: str, source_focus: SourceFocusName = "web") -> str:
    """Gemini 3 Flash - Google's fast model."""

    return _ask(query, Models.GEMINI_3_FLASH, source_focus)


@mcp.tool
def pplx_gemini_flash_think(query: str, source_focus: SourceFocusName = "web") -> str:
    """Gemini 3 Flash Thinking - Google's fast model with extended thinking."""

    return _ask(query, Models.GEMINI_3_FLASH_THINKING, source_focus)


@mcp.tool
def pplx_gemini_pro_think(query: str, source_focus: SourceFocusName = "web") -> str:
    """Gemini 3 Pro Thinking - Google's most advanced model with extended thinking."""

    return _ask(query, Models.GEMINI_3_PRO_THINKING, source_focus)


@mcp.tool
def pplx_grok(query: str, source_focus: SourceFocusName = "web") -> str:
    """Grok 4.1 - xAI's latest model."""

    return _ask(query, Models.GROK_41, source_focus)


@mcp.tool
def pplx_grok_thinking(query: str, source_focus: SourceFocusName = "web") -> str:
    """Grok 4.1 Thinking - xAI's latest model with extended thinking."""

    return _ask(query, Models.GROK_41_THINKING, source_focus)


@mcp.tool
def pplx_kimi_thinking(query: str, source_focus: SourceFocusName = "web") -> str:
    """Kimi K2.5 Thinking - Moonshot AI's latest model."""

    return _ask(query, Models.KIMI_K25_THINKING, source_focus)


# =============================================================================
# Usage & Limits Tools
# =============================================================================


@mcp.tool
def pplx_usage(refresh: bool = False) -> str:
    """Check current Perplexity usage limits and remaining quotas.

    Shows remaining Pro Search, Deep Research, Create Files & Apps, and Browser
    Agent queries. Also shows subscription info and file/upload limits.
    
    Call this before heavy use to plan queries, or after getting rate limit
    errors to understand what's left.

    Args:
        refresh: Force refresh from Perplexity (ignores cache). Default False.
    """
    token = load_token()
    if not token:
        return (
            "NOT AUTHENTICATED\n\n"
            "No session token found. Authenticate first with pplx_auth_request_code."
        )

    cache = _get_limit_cache()
    if cache is None:
        return "ERROR: Could not initialize limit cache."

    parts: list[str] = []

    # Rate limits (the main value)
    limits = cache.get_rate_limits(force_refresh=refresh)
    if limits:
        parts.append("RATE LIMITS (remaining queries)")
        parts.append("=" * 40)
        parts.append(limits.format_summary())
    else:
        parts.append("WARNING: Could not fetch rate limits (network error or token issue).")

    # User settings (supplementary context)
    settings = cache.get_user_settings(force_refresh=refresh)
    if settings:
        parts.append("")
        parts.append("ACCOUNT INFO")
        parts.append("=" * 40)
        parts.append(settings.format_summary())

    return "\n".join(parts)


# =============================================================================
# Authentication Tools
# =============================================================================

# Session state for auth flow
_auth_session: dict = {}


@mcp.tool
def pplx_auth_status() -> str:
    """Check if Perplexity is authenticated.
    
    Returns the current authentication status and subscription tier if authenticated.
    Use this to check if re-authentication is needed before making queries.
    """
    from perplexity_web_mcp.cli.auth import get_user_info
    
    token = load_token()
    if not token:
        return (
            "NOT AUTHENTICATED\n\n"
            "No session token found. To authenticate:\n"
            "1. Call pplx_auth_request_code with your Perplexity email\n"
            "2. Check email for 6-digit verification code\n"
            "3. Call pplx_auth_complete with email and code"
        )
    
    # Verify token is valid
    user_info = get_user_info(token)
    if user_info:
        parts = [
            f"AUTHENTICATED\n",
            f"Email: {user_info.email}",
            f"Username: {user_info.username}",
            f"Subscription: {user_info.tier_display}",
        ]

        # Include rate limit snapshot
        cache = _get_limit_cache()
        if cache:
            limits = cache.get_rate_limits()
            if limits:
                parts.append(f"\nRemaining: {limits.remaining_pro} Pro | "
                             f"{limits.remaining_research} Research | "
                             f"{limits.remaining_labs} Labs | "
                             f"{limits.remaining_agentic_research} Agent")

        return "\n".join(parts)
    else:
        return (
            "TOKEN EXPIRED\n\n"
            "Session token exists but is invalid or expired. To re-authenticate:\n"
            "1. Call pplx_auth_request_code with your Perplexity email\n"
            "2. Check email for 6-digit verification code\n"
            "3. Call pplx_auth_complete with email and code"
        )


@mcp.tool
def pplx_auth_request_code(email: str) -> str:
    """Request a verification code for Perplexity authentication.
    
    Sends a 6-digit verification code to the provided email address.
    After calling this, check the email inbox and use pplx_auth_complete
    with the code to finish authentication.
    
    Args:
        email: Your Perplexity account email address
        
    Returns:
        Status message indicating if the code was sent successfully
    """
    from curl_cffi.requests import Session
    from orjson import loads
    
    global _auth_session
    
    BASE_URL = "https://www.perplexity.ai"
    
    try:
        # Initialize session
        session = Session(impersonate="chrome", headers={"Referer": BASE_URL, "Origin": BASE_URL})
        session.get(BASE_URL)
        csrf_data = loads(session.get(f"{BASE_URL}/api/auth/csrf").content)
        csrf = csrf_data.get("csrfToken")
        
        if not csrf:
            return "ERROR: Failed to obtain CSRF token. Please try again."
        
        # Send verification code
        response = session.post(
            f"{BASE_URL}/api/auth/signin/email?version=2.18&source=default",
            json={
                "email": email,
                "csrfToken": csrf,
                "useNumericOtp": "true",
                "json": "true",
                "callbackUrl": f"{BASE_URL}/?login-source=floatingSignup",
            },
        )
        
        if response.status_code != 200:
            return f"ERROR: Failed to send verification code. Status: {response.status_code}"
        
        # Store session for completion
        _auth_session = {"session": session, "email": email}
        
        return (
            f"SUCCESS: Verification code sent to {email}\n\n"
            f"Next steps:\n"
            f"1. Check your email inbox for a 6-digit code from Perplexity\n"
            f"2. Call pplx_auth_complete with email='{email}' and the code"
        )
        
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool  
def pplx_auth_complete(email: str, code: str) -> str:
    """Complete Perplexity authentication with the verification code.
    
    Use the 6-digit code received via email after calling pplx_auth_request_code.
    On success, the session token is saved and all pplx_* tools will work.
    
    Args:
        email: Your Perplexity account email (same as used in pplx_auth_request_code)
        code: The 6-digit verification code from your email
        
    Returns:
        Status message with authentication result and subscription tier
    """
    from curl_cffi.requests import Session
    from orjson import loads
    from perplexity_web_mcp.cli.auth import get_user_info
    
    global _auth_session
    
    BASE_URL = "https://www.perplexity.ai"
    SESSION_COOKIE_NAME = "__Secure-next-auth.session-token"
    
    try:
        # Use existing session or create new one
        if _auth_session and _auth_session.get("email") == email:
            session = _auth_session["session"]
        else:
            # Create fresh session if none exists
            session = Session(impersonate="chrome", headers={"Referer": BASE_URL, "Origin": BASE_URL})
            session.get(BASE_URL)
        
        # Validate code and get redirect URL
        response = session.post(
            f"{BASE_URL}/api/auth/otp-redirect-link",
            json={
                "email": email,
                "otp": code,
                "redirectUrl": f"{BASE_URL}/?login-source=floatingSignup",
                "emailLoginMethod": "web-otp",
            },
        )
        
        if response.status_code != 200:
            return f"ERROR: Invalid verification code. Please check and try again."
        
        redirect_path = loads(response.content).get("redirect")
        if not redirect_path:
            return "ERROR: No redirect URL received. Please try again."
        
        redirect_url = f"{BASE_URL}{redirect_path}" if redirect_path.startswith("/") else redirect_path
        
        # Get session token
        session.get(redirect_url)
        token = session.cookies.get(SESSION_COOKIE_NAME)
        
        if not token:
            return "ERROR: Authentication succeeded but token not found."
        
        # Save token
        if save_token(token):
            _auth_session = {}
            
            # Get user info
            user_info = get_user_info(token)
            if user_info:
                return (
                    f"SUCCESS: Authentication complete!\n\n"
                    f"Email: {user_info.email}\n"
                    f"Subscription: {user_info.tier_display}\n\n"
                    f"All pplx_* tools are now ready to use."
                )
            else:
                return "SUCCESS: Token saved. You can now use pplx_* tools."
        else:
            return "ERROR: Failed to save token. Check file permissions."
            
    except Exception as e:
        return f"ERROR: {e}"


def main() -> None:
    """Run the MCP server."""

    mcp.run()


if __name__ == "__main__":
    main()
