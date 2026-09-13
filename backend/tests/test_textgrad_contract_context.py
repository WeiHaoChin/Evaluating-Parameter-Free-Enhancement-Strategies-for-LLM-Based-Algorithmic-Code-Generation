import contextlib
import io
import unittest
from unittest.mock import patch

from TextGrad import IMMUTABLE_GENERATION_CONTRACT, run_textgrad_sync


class RecordingEngine:
    def __init__(self):
        self.call_type = None
        self.calls = []

    def set_generation_context(self, call_type, mode=None):
        self.call_type = call_type

    def __call__(self, prompt, system_prompt=None):
        self.calls.append((self.call_type, str(prompt), system_prompt or ""))
        if self.call_type == "prompt_optimization":
            return "<IMPROVED_VARIABLE>Prefer efficient algorithms.</IMPROVED_VARIABLE>"
        if self.call_type in ("initial_generation", "final_generation"):
            return "```python\nprint(1)\n```"
        return "Preserve the fixed output requirements."


class ContractContextTests(unittest.TestCase):
    def test_contract_reaches_real_textgrad_feedback_and_final_generation(self):
        main, feedback = RecordingEngine(), RecordingEngine()
        with patch("TextGrad.create_llm_client", side_effect=[main, feedback]), contextlib.redirect_stdout(io.StringIO()):
            result = run_textgrad_sync(
                prompt_text="Print one.", system_prompt="Solve the problem.",
                textGradModel="gpt-oss:120b", model="kimi-k2.6:cloud",
                loss_prompt="Evaluate correctness.", return_details=True,
            )
        for stage in ("critique_evaluation", "prompt_optimization"):
            calls = [call for call in feedback.calls if call[0] == stage]
            self.assertEqual(len(calls), 1)
            self.assertIn(IMMUTABLE_GENERATION_CONTRACT, calls[0][1] + calls[0][2])
        self.assertIn("These rules apply to the solution, not to your feedback", feedback.calls[0][1])
        self.assertEqual(main.calls[-1][2].count(IMMUTABLE_GENERATION_CONTRACT), 1)
        self.assertIn("Prefer efficient algorithms.", main.calls[-1][2])
        self.assertEqual(result[0], "```python\nprint(1)\n```")


if __name__ == "__main__":
    unittest.main()
