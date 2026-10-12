"""Explicit shared model contract, independent from any host environment."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent_core.contracts.errors import ModelConfigurationError
from agent_core.models import build_chat_model, model_connection_snapshot, model_profile_from_mapping


class ModelConnectionContractTest(unittest.TestCase):
    def setUp(self):
        self.values = {"model": "fixture-model", "provider": "openai", "base_url": "https://model.example/v1",
                       "api_key": "fixture-only-key", "temperature": "0.2", "timeout_seconds": "30", "max_tokens": "256"}

    def test_host_must_explicitly_supply_all_connection_values(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "legacy", "AGENT_LLM_MODEL": "ambient"}, clear=True):
            for field in ("model", "base_url", "api_key"):
                with self.subTest(field=field), self.assertRaises(ModelConfigurationError):
                    model_profile_from_mapping({**self.values, field: " "})

    def test_transport_normalization_is_explicit_and_snapshot_has_no_credential(self):
        profile = model_profile_from_mapping({**self.values, "provider": " OpenAI-Compatible ", "base_url": self.values["base_url"] + "/"})
        snapshot = model_connection_snapshot(profile)
        self.assertEqual(profile.provider, "openai")
        self.assertEqual(profile.base_url, self.values["base_url"])
        self.assertEqual(set(snapshot), {"model", "provider", "base_url", "temperature", "timeout_seconds", "max_tokens"})
        self.assertNotIn(self.values["api_key"], repr(profile))
        self.assertNotIn(self.values["api_key"], repr(snapshot))

    def test_unsupported_provider_never_silently_uses_openai(self):
        with self.assertRaises(ModelConfigurationError):
            model_profile_from_mapping({**self.values, "provider": "anthropic"})

    def test_invalid_urls_fail_with_safe_errors(self):
        urls = ("relative/v1", "ftp://model.example", "https://user:password@model.example/v1",
                "https://model.example/v1?token=fixture", "https://model.example/v1#secret",
                "https://[broken/v1", "https://model.example:bad/v1", "https://model.example:99999/v1")
        for url in urls:
            with self.subTest(url=url), self.assertRaises(ModelConfigurationError) as caught:
                model_profile_from_mapping({**self.values, "base_url": url})
            self.assertNotIn(url, str(caught.exception))

    def test_nonfinite_or_out_of_range_numbers_are_rejected(self):
        for field, values in {"temperature": ("nan", "inf", "-inf", "-1", "2.1"),
                              "timeout_seconds": ("nan", "inf", "-inf", "0"),
                              "max_tokens": ("1.2", "0", "-1", "nan")}.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ModelConfigurationError):
                    model_profile_from_mapping({**self.values, field: value})

    def test_factory_rejects_endpoint_or_key_overrides_before_constructing_clients(self):
        profile = model_profile_from_mapping(self.values)
        for name in ("api_key", "base_url", "model", "http_client", "http_async_client", "model_kwargs"):
            with self.subTest(name=name), self.assertRaises(ModelConfigurationError):
                build_chat_model(profile, options={name: "override"})

    def test_remediation_options_cannot_introduce_invalid_temperature(self):
        profile = model_profile_from_mapping(self.values)
        for temperature in ("nan", "inf", -1, 3):
            with self.subTest(temperature=temperature), self.assertRaises(ModelConfigurationError):
                build_chat_model(profile, options={"temperature": temperature})


if __name__ == "__main__":
    unittest.main()
