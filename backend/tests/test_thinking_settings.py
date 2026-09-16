import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from config.models import unavailable_thinking_config
from llm_clients import _discover_thinking_configs, _thinking_config_from_recommendation
from schemas import Settings, settings_defaults


class ThinkingSettingsTests(unittest.TestCase):
    def test_defaults_publish_capabilities_and_textgrad_safeguards(self):
        defaults = settings_defaults(discovered_thinking={
            "qwen3.5:cloud": {
                "configurable": True, "default": "default",
                "modes": ["default", "off", "on"], "note": "discovered",
            },
            "gpt-oss:120b-cloud": {
                "configurable": True, "default": "default",
                "modes": ["default", "low", "medium", "high"],
                "note": "discovered",
            },
        })

        self.assertIn("off", defaults["modelThinking"]["qwen3.5:cloud"]["modes"])
        self.assertNotIn("off", defaults["modelThinking"]["gpt-oss:120b"]["modes"])
        self.assertIn("NON-NEGOTIABLE", defaults["immutableGenerationContract"])
        self.assertIn("general and reusable", defaults["optimizationBoundary"])

    def test_model_specific_thinking_mode_is_validated(self):
        Settings(model="qwen3.5:cloud", thinkingMode="off")
        with self.assertRaises(ValueError):
            Settings(model="gpt-oss:120b", thinkingMode="extreme")

    def test_unknown_model_exposes_only_provider_default(self):
        self.assertEqual(unavailable_thinking_config()["modes"], ["default"])

    def test_recommendation_values_are_translated_without_model_names(self):
        config = _thinking_config_from_recommendation({
            "values": [False, True, "max"], "default": True,
        })

        self.assertEqual(config["modes"], ["default", "off", "on", "max"])
        self.assertEqual(config["providerDefault"], "on")
        self.assertEqual(config["source"], "ollama_recommendations")

    def test_cloud_suffix_aliases_share_discovered_metadata(self):
        discovered = {
            "gpt-oss:120b-cloud": {
                "configurable": True, "default": "default",
                "modes": ["default", "low", "medium", "high"],
            }
        }
        defaults = settings_defaults(discovered_thinking=discovered)

        self.assertEqual(
            defaults["modelThinking"]["gpt-oss:120b"]["modes"],
            ["default", "low", "medium", "high"],
        )

    def test_new_cloud_recommendations_are_added_to_available_models(self):
        discovered = {
            "deepseek-v4.1-flash:cloud": {
                "configurable": True, "default": "default",
                "modes": ["default", "off", "low", "high", "max"],
            },
            "gemma4:26b": {
                "configurable": True, "default": "default",
                "modes": ["default", "off", "on"],
            },
        }

        defaults = settings_defaults(discovered_thinking=discovered)

        self.assertIn("deepseek-v4.1-flash:cloud", defaults["models"])
        self.assertNotIn("gemma4:26b", defaults["models"])

    def test_discovery_combines_exact_recommendations_and_show_capabilities(self):
        response = Mock(status_code=200)
        response.json.return_value = {"recommendations": [{
            "model": "recommended:cloud",
            "thinking": {"values": [False, True], "default": True},
        }]}
        client = Mock()
        client.show.return_value = SimpleNamespace(capabilities=["completion", "thinking"])

        with patch("llm_clients.httpx.get", return_value=response):
            discovered = _discover_thinking_configs(client, ["local:latest"])

        self.assertEqual(
            discovered["recommended:cloud"]["modes"],
            ["default", "off", "on"],
        )
        self.assertEqual(discovered["local:latest"]["modes"], ["default"])
        self.assertFalse(discovered["local:latest"]["configurable"])


if __name__ == "__main__":
    unittest.main()
