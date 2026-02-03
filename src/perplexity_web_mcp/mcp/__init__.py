"""MCP server for Perplexity Web MCP."""

from __future__ import annotations


__all__: list[str] = ["run_server"]


def run_server() -> None:
    """Run the MCP server."""

    from .server import main  # noqa: PLC0415

    main()
