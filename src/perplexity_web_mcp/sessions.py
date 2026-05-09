"""Session management for multi-turn conversations.

Provides an in-memory store for maintaining conversation state
across separate queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .models import Model


@dataclass(slots=True)
class ConversationSession:
    """Minimal state needed to restore a Perplexity conversation."""
    
    backend_uuid: str
    read_write_token: str | None
    model: Model
    created_at: float


class SessionStore:
    """Thread-safe, TTL-based in-memory store for conversation sessions."""
    
    __slots__ = ("_lock", "_sessions", "_ttl_seconds")
    
    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._lock = Lock()
        self._sessions: dict[str, ConversationSession] = {}
        self._ttl_seconds = ttl_seconds

    def save(self, conversation_id: str, backend_uuid: str, read_write_token: str | None, model: Model) -> None:
        """Save or update a conversation session. Also triggers cleanup of expired sessions."""
        with self._lock:
            self._evict_expired_locked()
            self._sessions[conversation_id] = ConversationSession(
                backend_uuid=backend_uuid,
                read_write_token=read_write_token,
                model=model,
                created_at=monotonic(),
            )

    def get(self, conversation_id: str) -> ConversationSession | None:
        """Retrieve a session if it exists and hasn't expired."""
        with self._lock:
            session = self._sessions.get(conversation_id)
            if session is None:
                return None
                
            if monotonic() - session.created_at > self._ttl_seconds:
                del self._sessions[conversation_id]
                return None
                
            # Update TTL on access
            session.created_at = monotonic()
            return session

    def _evict_expired_locked(self) -> None:
        """Evict expired sessions. Must be called with lock held."""
        now = monotonic()
        expired = [
            conv_id for conv_id, session in self._sessions.items()
            if now - session.created_at > self._ttl_seconds
        ]
        for conv_id in expired:
            del self._sessions[conv_id]
