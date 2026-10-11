from __future__ import annotations

import json

import pytest

from agent import image_gen_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


class TestPluginDispatch:


    def test_handler_forwards_only_the_creative_controls_the_plugin_declares(self, monkeypatch, tmp_path):
        """Declared controls reach generate(); an undeclared one the model sent anyway is dropped, so a
        plugin whose generate() lacks **kwargs never sees it."""
        from agent.image_gen_provider import ImageGenProvider
        from hermes_cli import plugins as plugins_module
        from tools import image_generation_tool

        seen = {}

        class _Recorder(ImageGenProvider):
            @property
            def name(self):
                return "recorder"

            def capabilities(self):
                return {"modalities": ["text"], "creative_controls": ["intensity"]}

            def generate(self, prompt, aspect_ratio="landscape", **kwargs):
                seen.update(kwargs)
                return {"success": True, "image": "/tmp/recorder.png", "model": "m", "prompt": prompt,
                        "aspect_ratio": aspect_ratio, "provider": "recorder"}

            def list_models(self):
                return []

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        image_gen_registry.register_provider(_Recorder())
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "recorder")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda **kwargs: None)

        result = json.loads(image_generation_tool._handle_image_generate(
            {"prompt": "draw cat", "aspect_ratio": "square", "intensity": 80, "creativity": "raw"}))

        assert result["success"] is True
        assert seen["intensity"] == 80
        assert "creativity" not in seen

    def test_deepinfra_key_alone_does_not_select_image_backend(self, monkeypatch):
        """DeepInfra chat credentials do not imply consent to image billing."""
        from tools import image_generation_tool

        monkeypatch.setenv("DEEPINFRA_API_KEY", "«redacted:sk-…»")
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: None)
        assert image_generation_tool._dispatch_to_plugin_provider("a cat", "square") is None

    def test_requirements_ignore_unselected_paid_plugin(self, monkeypatch):
        from tools import image_generation_tool

        monkeypatch.setattr(image_generation_tool, "check_fal_api_key", lambda: False)
        monkeypatch.setattr(
            image_generation_tool, "_read_configured_image_provider", lambda: None
        )
        assert image_generation_tool.check_image_generation_requirements() is False

    def test_fallback_provider_called_when_primary_fails(self, monkeypatch):
        """When primary provider returns success=False, fallback provider is automatically invoked."""
        from agent.image_gen_provider import ImageGenProvider
        from tools import image_generation_tool

        class _FailingPrimary(ImageGenProvider):
            @property
            def name(self):
                return "failing_primary"

            def generate(self, prompt, aspect_ratio="landscape", **kwargs):
                return {"success": False, "error": "rate limit reached", "error_type": "rate_limit"}

        class _WorkingFallback(ImageGenProvider):
            @property
            def name(self):
                return "working_fallback"

            def generate(self, prompt, aspect_ratio="landscape", **kwargs):
                return {"success": True, "image": "/tmp/fallback.png", "prompt": prompt, "aspect_ratio": aspect_ratio}

        image_gen_registry.register_provider(_FailingPrimary())
        image_gen_registry.register_provider(_WorkingFallback())

        monkeypatch.setattr(image_generation_tool, "_plugin_provider_name", lambda: "failing_primary")
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_model", lambda: "model_a")
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_fallback_provider", lambda: "working_fallback")
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_fallback_model", lambda: None)

        res_json = image_generation_tool._dispatch_to_plugin_provider("draw mountain", "landscape")
        res = json.loads(res_json)
        assert res["success"] is True
        assert res["image"] == "/tmp/fallback.png"
        assert res.get("fallback_used") is True
        assert res.get("primary_provider") == "failing_primary"

    def test_fallback_provider_called_when_primary_raises(self, monkeypatch):
        """When primary provider raises an exception, fallback provider is invoked."""
        from agent.image_gen_provider import ImageGenProvider
        from tools import image_generation_tool

        class _CrashingPrimary(ImageGenProvider):
            @property
            def name(self):
                return "crashing_primary"

            def generate(self, prompt, aspect_ratio="landscape", **kwargs):
                raise RuntimeError("Service unavailable 503")

        class _WorkingFallback(ImageGenProvider):
            @property
            def name(self):
                return "working_fallback"

            def generate(self, prompt, aspect_ratio="landscape", **kwargs):
                return {"success": True, "image": "/tmp/recovered.png", "prompt": prompt}

        image_gen_registry.register_provider(_CrashingPrimary())
        image_gen_registry.register_provider(_WorkingFallback())

        monkeypatch.setattr(image_generation_tool, "_plugin_provider_name", lambda: "crashing_primary")
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_model", lambda: None)
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_fallback_provider", lambda: "working_fallback")
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_fallback_model", lambda: None)

        res_json = image_generation_tool._dispatch_to_plugin_provider("draw sky", "square")
        res = json.loads(res_json)
        assert res["success"] is True
        assert res["image"] == "/tmp/recovered.png"
        assert res.get("fallback_used") is True

    def test_primary_success_does_not_call_fallback(self, monkeypatch):
        """When primary provider succeeds, fallback provider is not invoked."""
        from agent.image_gen_provider import ImageGenProvider
        from tools import image_generation_tool

        fallback_called = []

        class _GoodPrimary(ImageGenProvider):
            @property
            def name(self):
                return "good_primary"

            def generate(self, prompt, aspect_ratio="landscape", **kwargs):
                return {"success": True, "image": "/tmp/primary.png"}

        class _UnusedFallback(ImageGenProvider):
            @property
            def name(self):
                return "unused_fallback"

            def generate(self, prompt, aspect_ratio="landscape", **kwargs):
                fallback_called.append(True)
                return {"success": True, "image": "/tmp/fallback.png"}

        image_gen_registry.register_provider(_GoodPrimary())
        image_gen_registry.register_provider(_UnusedFallback())

        monkeypatch.setattr(image_generation_tool, "_plugin_provider_name", lambda: "good_primary")
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_model", lambda: None)
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_fallback_provider", lambda: "unused_fallback")

        res_json = image_generation_tool._dispatch_to_plugin_provider("draw river", "landscape")
        res = json.loads(res_json)
        assert res["success"] is True
        assert res["image"] == "/tmp/primary.png"
        assert not fallback_called

