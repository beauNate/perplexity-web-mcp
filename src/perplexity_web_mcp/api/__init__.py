"""Anthropic API-compatible server for Perplexity Web MCP."""

from __future__ import annotations


__all__: list[str] = ["run_server"]


def run_server() -> None:
    """Run the Anthropic API-compatible server."""
    from .server import run_server as _run

    _run()
