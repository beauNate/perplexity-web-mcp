#!/usr/bin/env python3
"""Detect Perplexity model changes by comparing live API config to stored reference.

Usage:
    python scripts/detect_model_changes.py                    # Show diff
    python scripts/detect_model_changes.py --save             # Update reference snapshot
    python scripts/detect_model_changes.py --json             # Output raw JSON
    python scripts/detect_model_changes.py --from-file X.json # Use a local JSON file
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

API_URL = "https://www.perplexity.ai/rest/models/config?config_schema=v1"
REFERENCE_PATH = Path(__file__).parent / "reference_model_config.json"
MODELS_PY = Path(__file__).parent.parent / "src" / "perplexity_web_mcp" / "models.py"


BROWSER_FETCH_HINT = """
💡 Cloudflare may block direct API requests. Workarounds:
   1. Open Chrome with debug port:
      /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome \\
          --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug-profile \\
          https://www.perplexity.ai
      Then run: python scripts/detect_model_changes.py --from-browser

   2. Or fetch manually in any browser console:
      fetch('/rest/models/config?config_schema=v1').then(r=>r.json()).then(d=>console.log(JSON.stringify(d)))
      Copy the JSON output to a file, then run:
      python scripts/detect_model_changes.py --from-file config.json
"""


def fetch_live_config() -> dict:
    """Fetch the current model config from Perplexity's public API via curl."""
    result = subprocess.run(
        ["curl", "-s", "-f", "-H", "User-Agent: Mozilla/5.0", API_URL],
        capture_output=True, text=True, timeout=20,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"curl failed (exit {result.returncode}): {result.stderr.strip()}\n"
            f"{BROWSER_FETCH_HINT}"
        )
    return json.loads(result.stdout)


def fetch_via_browser(port: int = 9222) -> dict:
    """Fetch model config via Chrome DevTools on the given debug port."""
    import urllib.request

    # Get the first page's WebSocket URL
    pages = json.loads(
        urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=5).read()
    )
    if not pages:
        raise RuntimeError("No Chrome pages found. Is Chrome open with --remote-debugging-port?")

    # Navigate to perplexity and fetch via the page
    ws_url = pages[0]["webSocketDebuggerUrl"]
    print(f"   Connected to: {pages[0].get('title', 'unknown')}")

    # Use subprocess to run a quick node/python fetch through the page
    # Simpler: just use the Chrome DevTools HTTP endpoint to evaluate JS
    import websocket  # type: ignore[import-untyped]

    ws = websocket.create_connection(ws_url, timeout=15)
    ws.send(json.dumps({
        "id": 1,
        "method": "Runtime.evaluate",
        "params": {
            "expression": "fetch('/rest/models/config?config_schema=v1').then(r=>r.json()).then(d=>JSON.stringify(d))",
            "awaitPromise": True,
            "returnByValue": True,
        },
    }))
    resp = json.loads(ws.recv())
    ws.close()

    value = resp.get("result", {}).get("result", {}).get("value", "")
    if not value:
        raise RuntimeError(f"Failed to evaluate script in Chrome: {resp}")
    return json.loads(value)


def load_reference() -> dict | None:
    """Load the stored reference config, or None if missing."""
    if REFERENCE_PATH.exists():
        return json.loads(REFERENCE_PATH.read_text())
    return None


def save_reference(config: dict) -> None:
    """Save a new reference snapshot."""
    REFERENCE_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"✅ Reference config saved to {REFERENCE_PATH}")


def get_codebase_models() -> set[str]:
    """Extract model identifiers currently defined in models.py."""
    if not MODELS_PY.exists():
        return set()
    identifiers = set()
    for line in MODELS_PY.read_text().splitlines():
        if 'identifier="' in line:
            ident = line.split('identifier="')[1].split('"')[0]
            identifiers.add(ident)
    return identifiers


def diff_configs(old: dict, new: dict) -> None:
    """Print a human-readable diff between two model configs."""
    old_models = set(old.get("models", {}).keys())
    new_models = set(new.get("models", {}).keys())

    added = new_models - old_models
    removed = old_models - new_models
    common = old_models & new_models

    # --- Model catalog changes ---
    print("\n" + "=" * 60)
    print("MODEL CATALOG CHANGES (models dict)")
    print("=" * 60)

    if added:
        print(f"\n🆕 Added ({len(added)}):")
        for m in sorted(added):
            info = new["models"][m]
            print(f"   + {m}: {info['label']} ({info['provider']})")
            print(f"     Description: {info['description']}")
    else:
        print("\n   No new models added.")

    if removed:
        print(f"\n🗑️  Removed ({len(removed)}):")
        for m in sorted(removed):
            info = old["models"][m]
            print(f"   - {m}: {info['label']} ({info['provider']})")
    else:
        print("\n   No models removed.")

    # Check for label/description/provider changes
    changed = []
    for m in sorted(common):
        old_info, new_info = old["models"][m], new["models"][m]
        diffs = {}
        for key in ("label", "description", "provider", "mode"):
            if old_info.get(key) != new_info.get(key):
                diffs[key] = (old_info.get(key), new_info.get(key))
        if diffs:
            changed.append((m, diffs))

    if changed:
        print(f"\n📝 Changed ({len(changed)}):")
        for m, diffs in changed:
            print(f"   ~ {m}:")
            for key, (old_val, new_val) in diffs.items():
                print(f"     {key}: {old_val!r} → {new_val!r}")
    else:
        print("\n   No model metadata changes.")

    # --- Active selector config changes ---
    print("\n" + "=" * 60)
    print("ACTIVE MODEL SELECTOR CHANGES (config array)")
    print("=" * 60)

    def config_key(c: dict) -> str:
        return c.get("reasoning_model") or c.get("non_reasoning_model") or c.get("label", "?")

    def config_modes_match(c: dict) -> str:
        """Return a string summarizing the config's mode filter."""
        # Use the model identifier from the models dict to get the mode
        ident = c.get("reasoning_model") or c.get("non_reasoning_model")
        if ident and ident in new.get("models", {}):
            return new["models"][ident].get("mode", "search")
        return "search"

    old_config = {config_key(c): c for c in old.get("config", [])}
    new_config = {config_key(c): c for c in new.get("config", [])}

    config_added = set(new_config.keys()) - set(old_config.keys())
    config_removed = set(old_config.keys()) - set(new_config.keys())

    # Filter to only search mode for the selector comparison
    if config_added:
        print(f"\n🆕 Added to selector ({len(config_added)}):")
        for k in sorted(config_added):
            c = new_config[k]
            thinking = "Always" if not c.get("non_reasoning_model") else (
                "Toggle" if c.get("reasoning_model") else "No"
            )
            print(f"   + {c['label']} (id: {k}, tier: {c.get('subscription_tier')}, thinking: {thinking})")
    else:
        print("\n   No models added to selector.")

    if config_removed:
        print(f"\n🗑️  Removed from selector ({len(config_removed)}):")
        for k in sorted(config_removed):
            c = old_config[k]
            print(f"   - {c['label']} (id: {k})")
    else:
        print("\n   No models removed from selector.")

    # --- Codebase comparison ---
    print("\n" + "=" * 60)
    print("CODEBASE STATUS")
    print("=" * 60)

    codebase_models = get_codebase_models()
    # Get the identifiers from the active config
    active_identifiers = set()
    for c in new.get("config", []):
        if c.get("non_reasoning_model"):
            active_identifiers.add(c["non_reasoning_model"])
        if c.get("reasoning_model"):
            active_identifiers.add(c["reasoning_model"])

    # Filter to search-mode models only
    search_active = set()
    for ident in active_identifiers:
        model_info = new.get("models", {}).get(ident, {})
        if model_info.get("mode") == "search":
            search_active.add(ident)

    missing_from_codebase = search_active - codebase_models
    extra_in_codebase = codebase_models - search_active

    if missing_from_codebase:
        print(f"\n⚠️  In Perplexity but NOT in codebase ({len(missing_from_codebase)}):")
        for m in sorted(missing_from_codebase):
            info = new["models"].get(m, {})
            print(f"   + {m}: {info.get('label', '?')} ({info.get('provider', '?')})")
    else:
        print("\n   ✅ All active search models are in the codebase.")

    if extra_in_codebase:
        print(f"\n⚠️  In codebase but NOT in Perplexity active selector ({len(extra_in_codebase)}):")
        for m in sorted(extra_in_codebase):
            status = "still in API catalog" if m in new_models else "GONE from API"
            print(f"   - {m} ({status})")
    else:
        print("\n   ✅ No stale models in the codebase.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect Perplexity model changes")
    parser.add_argument("--save", action="store_true", help="Save current config as reference")
    parser.add_argument("--json", action="store_true", help="Output raw JSON config")
    parser.add_argument("--from-file", type=Path, metavar="FILE",
                        help="Use a local JSON file instead of fetching from API")
    parser.add_argument("--from-browser", action="store_true",
                        help="Fetch via Chrome DevTools (requires --remote-debugging-port=9222)")
    args = parser.parse_args()

    if args.from_file:
        print(f"📂 Loading config from {args.from_file}...")
        live = json.loads(args.from_file.read_text())
    elif args.from_browser:
        print("🌐 Fetching config via Chrome DevTools...")
        try:
            live = fetch_via_browser()
        except Exception as e:
            print(f"❌ Failed to fetch via browser: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("🔍 Fetching live model config from Perplexity API...")
        try:
            live = fetch_live_config()
        except Exception as e:
            print(f"❌ Failed to fetch config: {e}", file=sys.stderr)
            sys.exit(1)

    if args.json:
        print(json.dumps(live, indent=2))
        return

    if args.save:
        save_reference(live)
        return

    ref = load_reference()
    if ref is None:
        print("⚠️  No reference config found. Saving current config as baseline.")
        save_reference(live)
        print("\nRun again to detect future changes.")
        return

    diff_configs(ref, live)

    # Summary
    print("\n" + "=" * 60)
    print("FILES TO UPDATE (if changes found)")
    print("=" * 60)
    print("""
1. src/perplexity_web_mcp/models.py         — Model enum
2. src/perplexity_web_mcp/shared.py         — MODEL_MAP + ALL_SHORTCUTS
3. src/perplexity_web_mcp/mcp/server.py     — pplx_* tool functions
4. src/perplexity_web_mcp/api/server.py     — MODEL_ALIASES + AVAILABLE_MODELS
5. src/perplexity_web_mcp/cli/ai_doc.py     — CLI help text
6. README.md                                — Model tables
7. CLAUDE.md                                — Model list
8. src/.../data/SKILL.md                    — Skill docs
9. src/.../data/references/models.md        — Model reference
10. src/.../data/references/mcp-tools.md    — MCP tools reference
11. src/.../data/references/api-endpoints.md — API docs
12. skills/perplexity-web-mcp/              — Mirror of data/
13. tests/test_shared.py                    — Model shortcut tests
14. tests/test_rate_limits.py               — Rate limit tests
""")

    print("💡 Run with --save after applying changes to update the reference.")


if __name__ == "__main__":
    main()
