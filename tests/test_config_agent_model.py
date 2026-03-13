"""Tests for config.resolve_agent_model()."""

import pytest

import config


class TestResolveAgentModel:
    def test_default_orchestrator_uses_smart_model(self):
        provider, model, base_url = config.resolve_agent_model("orchestrator")
        assert provider == config.LLM_PROVIDER
        assert model == config.SMART_MODEL
        assert base_url is None

    def test_default_sub_agent_uses_sub_agent_model(self):
        for agent_type in ("viz_plotly", "viz_mpl", "viz_jsx", "data_ops", "data_io", "envoy", "memory", "eureka"):
            provider, model, base_url = config.resolve_agent_model(agent_type)
            assert provider == config.LLM_PROVIDER
            assert model == config.SUB_AGENT_MODEL, f"Failed for {agent_type}"
            assert base_url is None

    def test_override_from_agent_models(self, monkeypatch):
        monkeypatch.setattr(config, "_user_config", {
            "agent_models": {"viz_plotly": "anthropic/claude-sonnet-4-20250514"}
        })
        provider, model, base_url = config.resolve_agent_model("viz_plotly")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-20250514"
        assert base_url is None

    def test_invalid_format_raises(self, monkeypatch):
        monkeypatch.setattr(config, "_user_config", {
            "agent_models": {"viz_plotly": "no-slash-here"}
        })
        with pytest.raises(ValueError, match="provider/model"):
            config.resolve_agent_model("viz_plotly")

    def test_workbench_agent_override(self, monkeypatch):
        monkeypatch.setattr(config, "_user_config", {
            "workbench": {
                "agents": {
                    "viz_plotly": {
                        "provider": "openai",
                        "model": "gpt-4o",
                        "base_url": "https://custom.api.com/v1",
                    }
                }
            }
        })
        provider, model, base_url = config.resolve_agent_model("viz_plotly")
        assert provider == "openai"
        assert model == "gpt-4o"
        assert base_url == "https://custom.api.com/v1"

    def test_workbench_preset_override(self, monkeypatch):
        monkeypatch.setattr(config, "_user_config", {
            "workbench": {"preset": "fast"},
            "presets": {
                "fast": {
                    "agents": {
                        "data_ops": {
                            "provider": "gemini",
                            "model": "gemini-2.5-flash-lite",
                        }
                    }
                }
            },
        })
        provider, model, base_url = config.resolve_agent_model("data_ops")
        assert provider == "gemini"
        assert model == "gemini-2.5-flash-lite"
        assert base_url is None

    def test_workbench_agent_takes_priority_over_preset(self, monkeypatch):
        monkeypatch.setattr(config, "_user_config", {
            "workbench": {
                "preset": "fast",
                "agents": {
                    "viz_plotly": {
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-20250514",
                    }
                },
            },
            "presets": {
                "fast": {
                    "agents": {
                        "viz_plotly": {
                            "provider": "openai",
                            "model": "gpt-4o",
                        }
                    }
                }
            },
        })
        provider, model, base_url = config.resolve_agent_model("viz_plotly")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-20250514"

    def test_workbench_takes_priority_over_legacy(self, monkeypatch):
        monkeypatch.setattr(config, "_user_config", {
            "workbench": {
                "agents": {
                    "viz_plotly": {
                        "provider": "openai",
                        "model": "gpt-4o",
                    }
                }
            },
            "agent_models": {"viz_plotly": "anthropic/claude-sonnet-4-20250514"},
        })
        provider, model, base_url = config.resolve_agent_model("viz_plotly")
        assert provider == "openai"
        assert model == "gpt-4o"
