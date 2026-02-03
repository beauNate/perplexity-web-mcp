"""Anthropic API-compatible server for Perplexity Web MCP.

This server provides an Anthropic Messages API compatible interface,
allowing Claude Code and other Anthropic SDK clients to use Perplexity
models as a backend.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from perplexity_web_mcp import Perplexity, ConversationConfig, Models
from perplexity_web_mcp.enums import CitationMode
from perplexity_web_mcp.models import Model


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ServerConfig:
    """Server configuration from environment variables."""
    
    session_token: str
    api_key: str | None = None  # Optional auth
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    default_model: str = "auto"
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load from environment."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        session_token = os.getenv("PERPLEXITY_SESSION_TOKEN")
        if not session_token:
            raise ValueError(
                "PERPLEXITY_SESSION_TOKEN required. "
                "Run 'pwm-auth' to authenticate."
            )
        
        return cls(
            session_token=session_token,
            api_key=os.getenv("ANTHROPIC_API_KEY"),  # For auth validation
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8080")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            default_model=os.getenv("DEFAULT_MODEL", "auto"),
        )


# =============================================================================
# Model Mapping
# =============================================================================

# Map Anthropic model names to Perplexity models
ANTHROPIC_MODEL_MAP: dict[str, tuple[Model, Model | None]] = {
    # Anthropic models -> route through Perplexity's Claude
    "claude-sonnet-4-5": (Models.CLAUDE_45_SONNET, Models.CLAUDE_45_SONNET_THINKING),
    "claude-4-5-sonnet": (Models.CLAUDE_45_SONNET, Models.CLAUDE_45_SONNET_THINKING),
    "claude-sonnet-4-5-20250929": (Models.CLAUDE_45_SONNET, Models.CLAUDE_45_SONNET_THINKING),
    "claude-opus-4-5": (Models.CLAUDE_45_OPUS, Models.CLAUDE_45_OPUS_THINKING),
    "claude-4-5-opus": (Models.CLAUDE_45_OPUS, Models.CLAUDE_45_OPUS_THINKING),
    
    # Perplexity-specific models
    "perplexity-auto": (Models.BEST, None),
    "perplexity-sonar": (Models.SONAR, None),
    "perplexity-research": (Models.DEEP_RESEARCH, None),
    
    # Other models via Perplexity
    "gpt-5-2": (Models.GPT_52, Models.GPT_52_THINKING),
    "gpt-52": (Models.GPT_52, Models.GPT_52_THINKING),
    "gemini-3-flash": (Models.GEMINI_3_FLASH, Models.GEMINI_3_FLASH_THINKING),
    "gemini-3-pro": (Models.GEMINI_3_PRO_THINKING, Models.GEMINI_3_PRO_THINKING),
    "grok-4-1": (Models.GROK_41, Models.GROK_41_THINKING),
    "grok-41": (Models.GROK_41, Models.GROK_41_THINKING),
}

# Models we expose via /v1/models
AVAILABLE_MODELS = [
    {"id": "claude-sonnet-4-5", "name": "Claude 4.5 Sonnet via Perplexity"},
    {"id": "claude-opus-4-5", "name": "Claude 4.5 Opus via Perplexity"},
    {"id": "perplexity-auto", "name": "Perplexity Auto (best model)"},
    {"id": "perplexity-sonar", "name": "Perplexity Sonar"},
    {"id": "perplexity-research", "name": "Perplexity Deep Research"},
    {"id": "gpt-5-2", "name": "GPT-5.2 via Perplexity"},
    {"id": "gemini-3-flash", "name": "Gemini 3 Flash via Perplexity"},
    {"id": "gemini-3-pro", "name": "Gemini 3 Pro via Perplexity"},
    {"id": "grok-4-1", "name": "Grok 4.1 via Perplexity"},
]


def get_model(name: str, thinking: bool = False) -> Model:
    """Get Perplexity model from name."""
    key = name.lower().strip()
    if key in ANTHROPIC_MODEL_MAP:
        base, thinking_model = ANTHROPIC_MODEL_MAP[key]
        if thinking and thinking_model:
            return thinking_model
        return base
    # Default to auto
    logging.warning(f"Unknown model '{name}', using perplexity-auto")
    return Models.BEST


# =============================================================================
# Pydantic Models (Anthropic API format)
# =============================================================================

class TextContent(BaseModel):
    """Text content block."""
    type: str = "text"
    text: str


class MessageContent(BaseModel):
    """Message content - can be string or array of content blocks."""
    model_config = ConfigDict(extra="allow")


class MessageParam(BaseModel):
    """Input message parameter."""
    role: str  # "user" or "assistant"
    content: str | list[dict[str, Any]]
    
    model_config = ConfigDict(extra="allow")
    
    def get_text(self) -> str:
        """Extract text content."""
        if isinstance(self.content, str):
            return self.content
        # Handle content blocks array
        texts = []
        for block in self.content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)


class MessagesRequest(BaseModel):
    """Anthropic Messages API request."""
    model: str
    max_tokens: int
    messages: list[MessageParam]
    
    # Optional parameters
    system: str | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None
    
    # Extended thinking (maps to Perplexity thinking models)
    thinking: dict[str, Any] | None = None
    
    model_config = ConfigDict(extra="allow")


class Usage(BaseModel):
    """Token usage."""
    input_tokens: int
    output_tokens: int


class TextBlock(BaseModel):
    """Response text block."""
    type: str = "text"
    text: str


class MessagesResponse(BaseModel):
    """Anthropic Messages API response."""
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[TextBlock]
    model: str
    stop_reason: str | None = "end_turn"
    stop_sequence: str | None = None
    usage: Usage


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    name: str
    created: int
    type: str = "model"


class ModelsListResponse(BaseModel):
    """Models list response."""
    data: list[ModelInfo]


class ErrorDetail(BaseModel):
    """Error detail."""
    type: str
    message: str


class ErrorResponse(BaseModel):
    """Error response."""
    type: str = "error"
    error: ErrorDetail


# =============================================================================
# Global State
# =============================================================================

config: ServerConfig
client: Perplexity
start_time: datetime


# =============================================================================
# Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    global config, client, start_time
    
    start_time = datetime.now()
    config = ServerConfig.from_env()
    client = Perplexity(session_token=config.session_token)
    
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    logging.info(f"Starting Anthropic API server on http://{config.host}:{config.port}")
    logging.info(f"Auth required: {'Yes' if config.api_key else 'No'}")
    
    yield
    
    client.close()


app = FastAPI(
    title="Perplexity Web MCP - Anthropic API",
    description="Anthropic Messages API compatible interface for Perplexity AI",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helpers
# =============================================================================

def verify_auth(request: Request) -> None:
    """Verify API key if configured."""
    if not config.api_key:
        return
    
    auth = request.headers.get("x-api-key", "")
    if not auth:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            auth = auth[7:]
    
    if auth != config.api_key:
        raise HTTPException(
            status_code=401,
            detail={"type": "authentication_error", "message": "Invalid API key"}
        )


def messages_to_query(messages: list[MessageParam], system: str | None = None) -> str:
    """Convert messages to Perplexity query."""
    parts = []
    
    if system:
        parts.append(f"[System Instructions]\n{system}")
    
    # For single user message, just return it
    user_msgs = [m for m in messages if m.role == "user"]
    if len(user_msgs) == 1 and not system and len(messages) == 1:
        return user_msgs[0].get_text()
    
    # Multi-turn: format as conversation
    for msg in messages:
        text = msg.get_text()
        if msg.role == "user":
            parts.append(f"User: {text}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {text}")
    
    return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token estimate."""
    return len(text) // 4


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "uptime_seconds": (datetime.now() - start_time).total_seconds(),
    }


@app.get("/v1/models")
async def list_models(request: Request):
    """List available models."""
    verify_auth(request)
    
    now = int(time.time())
    return ModelsListResponse(
        data=[
            ModelInfo(id=m["id"], name=m["name"], created=now)
            for m in AVAILABLE_MODELS
        ]
    )


@app.post("/v1/messages")
async def create_message(request: Request, body: MessagesRequest):
    """Create a message (Anthropic Messages API)."""
    verify_auth(request)
    
    if not body.messages:
        raise HTTPException(
            status_code=400,
            detail={"type": "invalid_request_error", "message": "messages is required"}
        )
    
    # Determine if thinking mode is requested
    thinking_enabled = body.thinking is not None and body.thinking.get("type") == "enabled"
    
    # Get the appropriate model
    model = get_model(body.model, thinking=thinking_enabled)
    
    # Convert messages to query
    query = messages_to_query(body.messages, body.system)
    
    # Generate response ID
    response_id = f"msg_{uuid.uuid4().hex[:24]}"
    
    if body.stream:
        return StreamingResponse(
            stream_response(response_id, body.model, model, query),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    
    # Non-streaming response
    try:
        conversation = client.create_conversation(
            ConversationConfig(
                model=model,
                citation_mode=CitationMode.CLEAN,
            )
        )
        
        # Run in thread to not block
        await asyncio.to_thread(conversation.ask, query)
        answer = conversation.answer or ""
        
        return MessagesResponse(
            id=response_id,
            content=[TextBlock(text=answer)],
            model=body.model,
            stop_reason="end_turn",
            usage=Usage(
                input_tokens=estimate_tokens(query),
                output_tokens=estimate_tokens(answer),
            ),
        )
    
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"type": "api_error", "message": str(e)}
        )


async def stream_response(
    response_id: str,
    model_name: str,
    model: Model,
    query: str,
) -> AsyncGenerator[str, None]:
    """Stream Anthropic-format SSE response."""
    import json
    import threading
    
    # message_start event
    message_start = {
        "type": "message_start",
        "message": {
            "id": response_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model_name,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": estimate_tokens(query), "output_tokens": 0},
        }
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
    
    # content_block_start event
    content_block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"
    
    # Stream content deltas
    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    def producer():
        last = ""
        try:
            conversation = client.create_conversation(
                ConversationConfig(model=model, citation_mode=CitationMode.CLEAN)
            )
            for resp in conversation.ask(query, stream=True):
                current = resp.answer or ""
                if len(current) > len(last):
                    delta = current[len(last):]
                    last = current
                    loop.call_soon_threadsafe(queue.put_nowait, ("delta", delta))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", last))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
    
    threading.Thread(target=producer, daemon=True).start()
    
    total_output = ""
    while True:
        kind, payload = await queue.get()
        if kind == "delta":
            total_output += payload
            delta_event = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": payload},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
        elif kind == "error":
            logging.error(f"Stream error: {payload}")
            break
        else:  # done
            total_output = payload
            break
    
    # content_block_stop event
    content_block_stop = {"type": "content_block_stop", "index": 0}
    yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n"
    
    # message_delta event (final usage)
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": estimate_tokens(total_output)},
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"
    
    # message_stop event
    message_stop = {"type": "message_stop"}
    yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"


# =============================================================================
# Main
# =============================================================================

def run_server():
    """Run the API server."""
    cfg = ServerConfig.from_env()
    uvicorn.run(
        "perplexity_web_mcp.api.server:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level.lower(),
    )


if __name__ == "__main__":
    run_server()
