# Perplexity Web MCP

MCP server and Anthropic API-compatible interface for Perplexity AI's web interface.

## Quick Start

```bash
# Install
uv venv && uv pip install -e ".[mcp]"

# Authenticate
pwm-auth

# Run MCP server
PERPLEXITY_SESSION_TOKEN="your_token" pwm-mcp
```

## Project Structure

```
src/perplexity_web_mcp/
├── __init__.py          # Package exports
├── core.py              # Perplexity client, Conversation class
├── models.py            # Model definitions (GPT, Claude, Gemini, Grok, etc.)
├── config.py            # ClientConfig, ConversationConfig
├── enums.py             # CitationMode, SearchFocus, SourceFocus
├── http.py              # HTTP client with retry/rate limiting
├── rate_limits.py       # Rate limit checking via /rest/rate-limit/all & /rest/user/settings
├── cli/
│   └── auth.py          # Authentication CLI (pwm-auth)
├── mcp/
│   └── server.py        # MCP server implementation (pwm-mcp)
└── api/
    └── __init__.py      # Anthropic API compatibility (TODO)
```

## Key APIs

### Rate Limit Checking
```python
from perplexity_web_mcp.rate_limits import fetch_rate_limits, fetch_user_settings, RateLimitCache

# One-shot fetch
limits = fetch_rate_limits(token)
print(f"Pro: {limits.remaining_pro}, Research: {limits.remaining_research}")

# Cached (thread-safe, 30s TTL for limits, 5min for settings)
cache = RateLimitCache(token)
limits = cache.get_rate_limits()  # Fetches or returns cached
settings = cache.get_user_settings()
cache.invalidate_rate_limits()    # Force refresh on next call
```

MCP server uses pre-flight limit checking before every query.
The `pplx_usage` tool exposes limits to calling LLMs.

### Subscription Detection
```python
from perplexity_web_mcp.cli.auth import get_user_info, SubscriptionTier

user = get_user_info(token)
if user.subscription_tier == SubscriptionTier.PRO:
    # Pro features available
```

### Models Available
- `Models.BEST` - Auto-select best model
- `Models.DEEP_RESEARCH` - In-depth reports
- `Models.SONAR` - Perplexity's model
- `Models.GPT_52` / `Models.GPT_52_THINKING`
- `Models.CLAUDE_45_SONNET` / `Models.CLAUDE_45_SONNET_THINKING`
- `Models.CLAUDE_46_OPUS` / `Models.CLAUDE_46_OPUS_THINKING`
- `Models.GEMINI_3_FLASH` / `Models.GEMINI_3_FLASH_THINKING`
- `Models.GEMINI_3_PRO_THINKING`
- `Models.GROK_41` / `Models.GROK_41_THINKING`
- `Models.KIMI_K25_THINKING`

## Environment Variables

- `PERPLEXITY_SESSION_TOKEN` - Session token from authentication

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[mcp,api]"

# Run tests (includes unit + integration tests for rate limits)
uv run --group tests --extra mcp pytest tests/ -v

# Run just unit tests (no network calls)
uv run --group tests --extra mcp pytest tests/ -v -k "not Integration"
```

## Credits

Based on [perplexity-webui-scraper](https://github.com/henrique-coder/perplexity-webui-scraper) by henrique-coder.
