"""Local video inputs are encoded internally; unsafe inputs never reach providers."""
import base64
import json
from types import SimpleNamespace

import pytest

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGNscFBgYGBgYmBgYGBgAAALCgDkaOWWTwAAAABJRU5ErkJggg==")

@pytest.fixture
def dispatch(monkeypatch):
    from tools import video_generation_tool as tool
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    calls = []
    provider = SimpleNamespace(name="recording", default_model=lambda: "test",
        generate=lambda **kwargs: calls.append(kwargs) or {"success": True})
    monkeypatch.setattr(tool, "_resolve_active_provider", lambda: provider)
    monkeypatch.setattr(tool, "_read_configured_video_provider", lambda: "recording")
    monkeypatch.setattr(tool, "_read_configured_video_model", lambda: "test")
    token = set_terminal_scope({"TERMINAL_ENV": "local"})
    yield lambda **kwargs: json.loads(tool._handle_video_generate({"prompt": "animate", **kwargs})), calls
    reset_terminal_scope(token)

def test_local_frame_and_references_encoded_before_dispatch(tmp_path, dispatch):
    run, calls = dispatch
    image = tmp_path / "frame with spaces.png"
    image.write_bytes(PNG)
    result = run(image_url=str(image), reference_image_urls=[str(image), "https://example.com/ref.png"])
    assert result["success"], result
    sent = calls[0]
    assert sent["image_url"].startswith("data:image/png;base64,")
    assert base64.b64decode(sent["image_url"].split(",", 1)[1]) == PNG
    assert sent["reference_image_urls"] == [sent["image_url"], "https://example.com/ref.png"]
    assert "data:image" not in json.dumps(result)

@pytest.mark.parametrize("kind", ["missing", "not_image", "credential"])
def test_invalid_local_input_stops_before_provider(tmp_path, dispatch, kind):
    run, calls = dispatch
    image = tmp_path / (".env" if kind == "credential" else "bad.png")
    if kind != "missing":
        image.write_bytes(PNG if kind == "credential" else b"not an image")
    result = run(image_url=str(image))
    assert not result["success"]
    assert not calls

def test_remote_and_inline_sources_unchanged(dispatch):
    run, calls = dispatch
    inline = "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert run(image_url="https://example.com/frame.png", reference_image_urls=[inline])["success"]
    assert calls[0]["image_url"] == "https://example.com/frame.png"
    assert calls[0]["reference_image_urls"] == [inline]

def test_sandbox_cannot_fall_back_to_host_file(tmp_path, dispatch, monkeypatch):
    from tools import image_source
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    run, calls = dispatch
    image = tmp_path / "host-only.png"
    image.write_bytes(PNG)
    monkeypatch.setattr(image_source, "_media_cache_roots", lambda: [])
    token = set_terminal_scope({"TERMINAL_ENV": "docker"})
    try:
        result = run(image_url=str(image))
    finally:
        reset_terminal_scope(token)
    assert not result["success"]
    assert not calls

def test_oversized_local_source_rejected(tmp_path, dispatch, monkeypatch):
    from tools import image_source
    run, calls = dispatch
    image = tmp_path / "oversized.png"
    image.write_bytes(PNG)
    monkeypatch.setattr(image_source, "_MAX_INGEST_BYTES", len(PNG) - 1)
    assert not run(image_url=str(image))["success"]
    assert not calls


