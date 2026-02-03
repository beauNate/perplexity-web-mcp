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
├── cli/
│   └── auth.py          # Authentication CLI (pwm-auth)
├── mcp/
│   └── server.py        # MCP server implementation (pwm-mcp)
└── api/
    └── __init__.py      # Anthropic API compatibility (TODO)
```

## Key APIs

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
- `Models.CLAUDE_45_OPUS` / `Models.CLAUDE_45_OPUS_THINKING`
- `Models.GEMINI_3_FLASH` / `Models.GEMINI_3_FLASH_THINKING`
- `Models.GROK_41` / `Models.GROK_41_THINKING`

## Environment Variables

- `PERPLEXITY_SESSION_TOKEN` - Session token from authentication

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[mcp,api]"

# Run tests
pytest
```

## Credits

Based on [perplexity-webui-scraper](https://github.com/henrique-coder/perplexity-webui-scraper) by henrique-coder.
