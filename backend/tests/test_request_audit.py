import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from ollama import Client

import request_audit
from llm_clients import OllamaCloudLLM


class RequestAuditTests(unittest.TestCase):
    def test_cloud_client_captures_serialized_system_prompt_without_credentials(self):
        received = []

        def respond(request):
            received.append(json.loads(request.content))
            return httpx.Response(200, json={
                "model": "kimi-k2.6:cloud", "done": True,
                "message": {"role": "assistant", "content": "answer"},
            })

        def make_client(**kwargs):
            return Client(**kwargs, transport=httpx.MockTransport(respond))

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audit.jsonl"
            with patch.object(request_audit, "AUDIT_PATH", path), patch(
                "llm_clients.Client", side_effect=make_client,
            ):
                llm = OllamaCloudLLM(
                    "kimi-k2.6:cloud", api_key="secret-test-key",
                    mode="full", max_output_tokens=8192,
                )
                self.assertEqual(llm("problem", system_prompt="Only Python."), "answer")
            raw = path.read_text(encoding="utf-8")
            entry = json.loads(raw)
            self.assertEqual(entry["payload"], received[0])
            self.assertEqual(entry["payload"]["messages"][0], {
                "role": "system", "content": "Only Python.",
            })
            self.assertIs(entry["payload"]["think"], False)
            self.assertEqual(entry["mode"], "full")
            self.assertNotIn("secret-test-key", raw)
            self.assertNotIn("Authorization", raw)


if __name__ == "__main__":
    unittest.main()
