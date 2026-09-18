"""Tests for profile-scoped session DB resolution in API server."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from gateway.platforms.api_server_runs import _resolve_live_session_id


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


def test_resolve_profile_home_explicit_profile(adapter, tmp_path):
    prof_dir = tmp_path / "profiles" / "architect"
    prof_dir.mkdir(parents=True)
    with patch("hermes_cli.profiles.get_profile_dir", return_value=prof_dir):
        home = adapter._resolve_profile_home(profile="architect")
        assert home == prof_dir


def test_resolve_profile_home_contextvar_profile(adapter, tmp_path):
    prof_dir = tmp_path / "profiles" / "researcher"
    prof_dir.mkdir(parents=True)
    token = _api_request_profile.set("researcher")
    try:
        with patch("hermes_cli.profiles.get_profile_dir", return_value=prof_dir):
            home = adapter._resolve_profile_home()
            assert home == prof_dir
    finally:
        _api_request_profile.reset(token)


def test_resolve_profile_home_default_fallback(adapter, tmp_path):
    default_home = tmp_path / "default_home"
    with patch("hermes_constants.get_hermes_home", return_value=default_home):
        assert adapter._resolve_profile_home(profile=None) == default_home
        assert adapter._resolve_profile_home(profile="default") == default_home
        assert adapter._resolve_profile_home(profile="") == default_home


@pytest.mark.asyncio
async def test_ensure_session_db_async_passes_profile_home(adapter, tmp_path):
    prof_dir = tmp_path / "profiles" / "architect"
    mock_db = MagicMock()
    with patch.object(adapter, "_resolve_profile_home", return_value=prof_dir) as mock_resolve, \
         patch.object(adapter, "_open_and_cache_session_db", return_value=mock_db) as mock_open:
        db = await adapter._ensure_session_db_async(profile="architect")
        mock_resolve.assert_called_once_with("architect")
        assert db is mock_db


@pytest.mark.asyncio
async def test_conversation_history_for_session_passes_profile(adapter):
    mock_db = MagicMock()
    mock_db.get_messages_as_conversation.return_value = [{"role": "user", "content": "prior turn"}]
    with patch.object(adapter, "_ensure_session_db_async", new=AsyncMock(return_value=mock_db)) as mock_ensure:
        history = await adapter._conversation_history_for_session("sess_123", profile="architect")
        mock_ensure.assert_awaited_once_with(profile="architect")
        assert history == [{"role": "user", "content": "prior turn"}]


@pytest.mark.asyncio
async def test_resolve_live_session_id_passes_profile(adapter):
    mock_db = MagicMock()
    mock_db.resolve_resume_session_id.return_value = "sess_resolved"
    with patch.object(adapter, "_ensure_session_db_async", new=AsyncMock(return_value=mock_db)) as mock_ensure:
        resolved = await _resolve_live_session_id(adapter, "sess_original", profile="architect")
        mock_ensure.assert_awaited_once_with(profile="architect")
        assert resolved == "sess_resolved"
