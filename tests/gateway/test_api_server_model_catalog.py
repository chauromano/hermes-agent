"""Tests for /api/model/catalog in APIServerAdapter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.fixture
def adapter():
    a = APIServerAdapter(PlatformConfig(enabled=True))
    a._api_key = "test-key"
    return a


def _make_request(adapter, path="/api/model/catalog", query=None, headers=None):
    from unittest.mock import MagicMock
    req = MagicMock(spec=web.Request)
    req.path = path
    req.query = query or {}
    req.headers = headers or {"Authorization": "Bearer test-key"}
    return req


@pytest.mark.asyncio
async def test_handle_model_catalog_success(adapter):
    mock_payload = {
        "provider": "openai-codex",
        "model": "gpt-6.1-sol",
        "providers": [
            {
                "slug": "openai-codex",
                "name": "ChatGPT or Codex Subscription",
                "authenticated": True,
                "is_current": True,
                "models": ["gpt-6.1-sol", "gpt-6-astra"],
                "capabilities": {
                    "gpt-6.1-sol": {"fast": True, "reasoning": True},
                    "gpt-6-astra": {"fast": True, "reasoning": True},
                },
            },
            {
                "slug": "vertex",
                "name": "Google Vertex AI",
                "authenticated": True,
                "is_current": False,
                "models": ["google/gemini-3.8-flash"],
                "capabilities": {
                    "google/gemini-3.8-flash": {"fast": False, "reasoning": True},
                },
            },
        ],
    }

    with patch("hermes_cli.inventory.build_model_options_payload", return_value=mock_payload), \
         patch("hermes_cli.inventory.load_picker_context", return_value=MagicMock()):
        req = _make_request(adapter)
        resp = await adapter._handle_model_catalog(req)
        assert resp.status == 200
        data = json.loads(resp.text)

        assert data["ok"] is True
        assert data["current_provider"] == "openai-codex"
        assert data["current_model"] == "gpt-6.1-sol"
        assert len(data["providers"]) == 2

        # Check provider details
        codex_p = next(p for p in data["providers"] if p["id"] == "openai-codex")
        assert codex_p["label"] == "ChatGPT or Codex Subscription"
        assert codex_p["authenticated"] is True
        assert codex_p["is_current"] is True
        assert len(codex_p["models"]) == 2

        # Check model entries
        sol = next(m for m in data["models"] if m["id"] == "gpt-6.1-sol")
        assert sol["provider"] == "openai-codex"
        assert sol["provider_label"] == "ChatGPT or Codex Subscription"
        assert sol["is_current"] is True
        assert sol["capabilities"]["fast"] is True

        gemini = next(m for m in data["models"] if m["id"] == "google/gemini-3.8-flash")
        assert gemini["provider"] == "vertex"
        assert gemini["is_current"] is False

        # Verify no credentials leaked
        text_dump = resp.text.lower()
        assert "api_key" not in text_dump
        assert "secret" not in text_dump
        assert "bearer" not in text_dump


@pytest.mark.asyncio
async def test_handle_model_catalog_unauthorized(adapter):
    req = _make_request(adapter, headers={"Authorization": "Bearer wrong-key"})
    resp = await adapter._handle_model_catalog(req)
    assert resp.status == 401
