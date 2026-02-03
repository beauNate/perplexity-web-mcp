# Perplexity Web MCP

MCP server and Anthropic API-compatible interface for Perplexity AI's web interface.

## Features

- **MCP Server**: Use Perplexity models (GPT-5.2, Claude 4.5, Gemini 3, Grok 4.1, etc.) as MCP tools
- **Subscription Detection**: Automatically detects Free/Pro/Max subscription tier
- **Multiple Models**: Access to all Perplexity-supported models including thinking variants
- **Deep Research**: Full support for Perplexity's Deep Research mode
- **Anthropic API** (Coming Soon): Use Perplexity as a drop-in replacement for Anthropic's API

## Installation

```bash
# Clone the repository
git clone https://github.com/jbendavi/perplexity-web-mcp.git
cd perplexity-web-mcp

# Create virtual environment and install
uv venv
uv pip install -e ".[mcp]"
```

## Authentication

```bash
# Run the authentication CLI
pwm-auth
```

This will:
1. Prompt for your Perplexity email
2. Send a verification code
3. Save your session token to `.env`
4. Display your subscription tier (Free/Pro/Max)

## Usage

### MCP Server

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "perplexity": {
      "command": "pwm-mcp",
      "env": {
        "PERPLEXITY_SESSION_TOKEN": "your_token_here"
      }
    }
  }
}
```

### Available MCP Tools

| Tool | Model | Description |
|------|-------|-------------|
| `pplx_ask` | Best | Auto-selects best model based on query |
| `pplx_deep_research` | Deep Research | In-depth reports with sources |
| `pplx_sonar` | Sonar | Perplexity's latest model |
| `pplx_gpt52` | GPT-5.2 | OpenAI's latest model |
| `pplx_gpt52_thinking` | GPT-5.2 | With extended thinking |
| `pplx_claude_sonnet` | Claude 4.5 Sonnet | Anthropic's fast model |
| `pplx_claude_sonnet_think` | Claude 4.5 Sonnet | With extended thinking |
| `pplx_gemini_flash` | Gemini 3 Flash | Google's fast model |
| `pplx_gemini_flash_think` | Gemini 3 Flash | With extended thinking |
| `pplx_gemini_pro_think` | Gemini 3 Pro | Google's most advanced |
| `pplx_grok` | Grok 4.1 | xAI's latest model |
| `pplx_grok_thinking` | Grok 4.1 | With extended thinking |
| `pplx_kimi_thinking` | Kimi K2.5 | Moonshot AI's model |

All tools support `source_focus`: `web`, `academic`, `social`, `finance`, `all`

### Python API

```python
from perplexity_web_mcp import Perplexity, ConversationConfig, Models

client = Perplexity(session_token="your_token")
conversation = client.create_conversation(
    ConversationConfig(model=Models.CLAUDE_45_SONNET)
)

conversation.ask("What is quantum computing?")
print(conversation.answer)

# Follow-up (context preserved)
conversation.ask("Explain it simpler")
print(conversation.answer)
```

## Subscription Tiers

| Tier | Model Access |
|------|--------------|
| Free | All models (quota limited) |
| Pro ($20/mo) | Extended quotas |
| Max ($200/mo) | Unlimited |

## Credits

Based on [perplexity-webui-scraper](https://github.com/henrique-coder/perplexity-webui-scraper) by henrique-coder.

## License

MIT
