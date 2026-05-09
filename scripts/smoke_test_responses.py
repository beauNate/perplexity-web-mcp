#!/usr/bin/env python3
"""
Integration smoke test for the Perplexity Web MCP /v1/responses API.
Validates live HTTP REST (streaming and non-streaming) and WebSockets against a running server.

Assumes the server is running on http://127.0.0.1:8181
Usage: uv run python scripts/smoke_test_responses.py
"""

import asyncio
import json
import sys

import httpx
import websockets


BASE_URL = "http://127.0.0.1:8181/v1/responses"
WS_URL = "ws://127.0.0.1:8181/v1/responses"

PAYLOAD = {
    "input": "Write a 1-sentence haiku about coding.",
    "instructions": "Be creative.",
    "model": "sonar",
}


async def test_rest_non_streaming():
    print("\n--- Testing REST (Non-Streaming) ---")
    async with httpx.AsyncClient() as client:
        try:
            payload = PAYLOAD.copy()
            payload["stream"] = False
            response = await client.post(BASE_URL, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            print("Response:", data)
            assert isinstance(data["output"], list)
            assert len(data["output"]) > 0
            message = data["output"][0]
            assert "content" in message
            assert isinstance(message["content"], list)
            assert len(message["content"]) > 0
            assert message["content"][0].get("text")
            print("✅ REST Non-Streaming SUCCESS")
        except Exception as e:
            print(f"❌ REST Non-Streaming FAILED: {e}")
            raise


async def test_rest_streaming():
    print("\n--- Testing REST (Streaming) ---")
    async with httpx.AsyncClient() as client:
        try:
            payload = PAYLOAD.copy()
            payload["stream"] = True
            chunks = []
            async with client.stream("POST", BASE_URL, json=payload, timeout=30.0) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        data = json.loads(data_str)
                        if data.get("type") == "response.output_text.delta":
                            chunks.append(data["delta"])

            print("Response chunk count:", len(chunks))
            full_text = "".join(chunks)
            print("Assembled Text:", full_text)
            assert len(chunks) > 0
            print("✅ REST Streaming SUCCESS")
        except Exception as e:
            print(f"❌ REST Streaming FAILED: {e}")
            raise


async def test_websockets():
    print("\n--- Testing WebSockets ---")
    try:
        async with websockets.connect(WS_URL) as websocket:
            ws_payload = {"type": "response.create", "response": PAYLOAD}
            await websocket.send(json.dumps(ws_payload))

            chunks = []
            async for message in websocket:
                data = json.loads(message)
                if data.get("type") == "response.completed":
                    break
                if data.get("type") == "response.output_text.delta":
                    chunks.append(data["delta"])

            print("Response chunk count:", len(chunks))
            full_text = "".join(chunks)
            print("Assembled Text:", full_text)
            assert len(chunks) > 0
            print("✅ WebSockets SUCCESS")
    except Exception as e:
        print(f"❌ WebSockets FAILED: {e}")
        raise


async def main():
    try:
        await test_rest_non_streaming()
        await test_rest_streaming()
        await test_websockets()
        print("\n🎉 ALL SMOKE TESTS PASSED 🎉")
    except Exception as e:
        print(f"\n💥 SMOKE TESTS FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
