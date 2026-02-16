"""Tests for the setup command (cli/setup.py).

All filesystem operations use tmp_path to avoid touching real configs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from perplexity_web_mcp.cli.setup import (
    AITool,
    MCP_SERVER_NAME,
    _add_file,
    _is_configured_file,
    _remove_file,
    cmd_setup,
)


# ============================================================================
# 1. _is_configured_file
# ============================================================================


class TestIsConfiguredFile:
    """Test file-based MCP config detection."""

    def test_returns_false_when_no_file(self, tmp_path: Path) -> None:
        tool = AITool(name="test", description="", config_path=tmp_path / "nope.json", config_hint="")
        assert _is_configured_file(tool) is False

    def test_returns_false_when_no_server(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text('{"mcpServers": {}}')
        tool = AITool(name="test", description="", config_path=cfg, config_hint="")
        assert _is_configured_file(tool) is False

    def test_returns_true_when_configured(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"mcpServers": {MCP_SERVER_NAME: {"command": "pwm-mcp"}}}))
        tool = AITool(name="test", description="", config_path=cfg, config_hint="")
        assert _is_configured_file(tool) is True

    def test_returns_false_for_invalid_json(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text("not json")
        tool = AITool(name="test", description="", config_path=cfg, config_hint="")
        assert _is_configured_file(tool) is False


# ============================================================================
# 2. _add_file / _remove_file
# ============================================================================


class TestAddRemoveFile:
    """Test adding/removing MCP config from JSON files."""

    def test_add_creates_new_file(self, tmp_path: Path) -> None:
        cfg = tmp_path / "sub" / "config.json"
        tool = AITool(name="test", description="", config_path=cfg, config_hint="")

        result = _add_file(tool)

        assert result is True
        assert cfg.exists()
        data = json.loads(cfg.read_text())
        assert MCP_SERVER_NAME in data["mcpServers"]

    def test_add_preserves_existing_servers(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"mcpServers": {"other-server": {"command": "other"}}}))
        tool = AITool(name="test", description="", config_path=cfg, config_hint="")

        _add_file(tool)

        data = json.loads(cfg.read_text())
        assert "other-server" in data["mcpServers"]
        assert MCP_SERVER_NAME in data["mcpServers"]

    def test_remove_deletes_server(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"mcpServers": {MCP_SERVER_NAME: {"command": "pwm-mcp"}, "other": {}}}))
        tool = AITool(name="test", description="", config_path=cfg, config_hint="")

        result = _remove_file(tool)

        assert result is True
        data = json.loads(cfg.read_text())
        assert MCP_SERVER_NAME not in data["mcpServers"]
        assert "other" in data["mcpServers"]

    def test_remove_returns_false_when_not_configured(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"mcpServers": {}}))
        tool = AITool(name="test", description="", config_path=cfg, config_hint="")

        assert _remove_file(tool) is False

    def test_remove_returns_false_when_no_file(self, tmp_path: Path) -> None:
        tool = AITool(name="test", description="", config_path=tmp_path / "nope.json", config_hint="")
        assert _remove_file(tool) is False

    def test_add_returns_false_for_none_path(self) -> None:
        tool = AITool(name="test", description="", config_path=None, config_hint="", uses_cli=True)
        assert _add_file(tool) is False


# ============================================================================
# 3. cmd_setup routing
# ============================================================================


class TestCmdSetup:
    """Test the cmd_setup CLI handler."""

    def test_help_returns_0(self, capsys: pytest.CaptureFixture) -> None:
        assert cmd_setup(["--help"]) == 0
        assert "Configure MCP server" in capsys.readouterr().out

    def test_no_args_returns_0(self, capsys: pytest.CaptureFixture) -> None:
        assert cmd_setup([]) == 0

    @patch("perplexity_web_mcp.cli.setup._get_tools")
    @patch("perplexity_web_mcp.cli.setup._is_configured", return_value=False)
    def test_list_shows_tools(
        self, mock_is_conf: MagicMock, mock_tools: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        mock_tools.return_value = [
            AITool(name="test-tool", description="A test tool", config_path=None, config_hint="/path"),
        ]
        cmd_setup(["list"])
        out = capsys.readouterr().out
        assert "test-tool" in out
        assert "A test tool" in out

    def test_add_unknown_client_returns_1(self, capsys: pytest.CaptureFixture) -> None:
        assert cmd_setup(["add", "nonexistent"]) == 1
        assert "Unknown client" in capsys.readouterr().err

    def test_add_missing_client_returns_1(self, capsys: pytest.CaptureFixture) -> None:
        assert cmd_setup(["add"]) == 1
        assert "requires a client" in capsys.readouterr().err

    def test_unknown_action_returns_1(self, capsys: pytest.CaptureFixture) -> None:
        assert cmd_setup(["bogus"]) == 1
