import json
import tempfile
import unittest
from pathlib import Path

from benchmark.metrics import compute_metrics
from benchmark.logger import append_test_case, _compact_results
from llm_clients import OllamaLLM
import rag_handler
from solver import summarize_test_outcomes


class FakeOllamaClient:
    def __init__(self, response):
        self.response = response
        self.last_options = None

    def chat(self, *, model, messages, options):
        self.last_options = options
        return self.response


class GenerationMetadataTests(unittest.TestCase):
    def test_rag_similarity_cutoff_keeps_boundary_and_higher_results(self):
        class FakeCollection:
            def query(self, **_kwargs):
                return {
                    "documents": [["below", "boundary", "above"]],
                    "metadatas": [[{}, {}, {}]],
                    "distances": [[0.201, 0.2, 0.1]],
                }

        previous = rag_handler._chroma_collection
        rag_handler._chroma_collection = FakeCollection()
        try:
            results = rag_handler.query_rag("problem")
        finally:
            rag_handler._chroma_collection = previous

        self.assertEqual(
            [result["text"] for result in results], ["boundary", "above"]
        )

    def test_test_case_companion_appends_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_testcases.json"
            append_test_case(path, {
                "id": "one", "title": "One", "test_cases": [1],
                "private_tests": [2], "evaluation_sample": {"sample": 1},
            }, {"seed": 42})
            append_test_case(path, {
                "id": "two", "title": "Two", "test_cases": [3],
                "private_tests": [4], "evaluation_sample": {"sample": 2},
            }, {"seed": 42})

            saved = json.loads(path.read_text())
            self.assertEqual(saved["benchmark"]["seed"], 42)
            self.assertEqual(
                [problem["id"] for problem in saved["problems"]],
                ["one", "two"],
            )
            self.assertEqual(saved["problems"][1]["private_tests"], [4])

    def test_result_compaction_removes_only_evaluation_payloads(self):
        results = [{
            "problem": {
                "id": "one", "title": "Problem", "statement": "Keep me",
                "test_cases": ["large"], "private_tests": ["very large"],
                "evaluation_sample": {"input_output": "very large"},
            },
            "modes": {"baseline": {
                "passed": True, "graded_list": [[True]],
                "results": {"duplicate": True},
            }},
        }]

        compacted = _compact_results(results)
        self.assertEqual(compacted[0]["problem"]["statement"], "Keep me")
        self.assertNotIn("private_tests", compacted[0]["problem"])
        self.assertNotIn("test_cases", compacted[0]["problem"])
        self.assertNotIn("evaluation_sample", compacted[0]["problem"])
        self.assertNotIn("results", compacted[0]["modes"]["baseline"])
        self.assertEqual(
            compacted[0]["modes"]["baseline"]["graded_list"], [[True]]
        )

    def test_ollama_limit_and_native_metadata_are_recorded(self):
        records = []
        llm = OllamaLLM(
            "local-model", max_output_tokens=123, metadata_sink=records,
            mode="rag_only", call_type="final_generation",
        )
        llm.client = FakeOllamaClient({
            "model": "local-model",
            "message": {"content": "solution"},
            "prompt_eval_count": 42,
            "eval_count": 17,
            "done_reason": "stop",
            "total_duration": 1000,
            "prompt_eval_duration": 200,
            "eval_duration": 700,
        })

        self.assertEqual(llm("full augmented prompt"), "solution")
        self.assertEqual(llm.client.last_options["num_predict"], 123)
        self.assertEqual(records[0]["prompt_tokens"], 42)
        self.assertEqual(records[0]["output_tokens"], 17)
        self.assertGreaterEqual(records[0]["client_duration_ns"], 0)
        self.assertFalse(records[0]["truncated"])

    def test_natural_stop_at_limit_is_not_marked_truncated(self):
        records = []
        llm = OllamaLLM("model", max_output_tokens=10, metadata_sink=records)
        llm.client = FakeOllamaClient({
            "message": {"content": "done"}, "eval_count": 10,
            "done_reason": "stop",
        })
        llm("prompt")
        self.assertFalse(records[0]["truncated"])

    def test_length_reason_is_marked_truncated(self):
        records = []
        llm = OllamaLLM("model", max_output_tokens=10, metadata_sink=records)
        llm.client = FakeOllamaClient({
            "message": {"content": "cut"}, "eval_count": 8,
            "done_reason": "length",
        })
        llm("prompt")
        self.assertTrue(records[0]["truncated"])

    def test_generation_aggregates_are_added_to_existing_summary(self):
        record = {
            "model": "model", "prompt_tokens": 100, "output_tokens": 50,
            "truncated": True, "eval_duration": 500,
            "total_duration": 800, "client_duration_ns": 2_000_000,
            "call_type": "final_generation",
        }
        results = [{
            "problem": {"difficulty": "easy", "platform": "test"},
            "modes": {"baseline": {
                "passed": True, "latency_ms": 10,
                "generation_records": [record],
            }},
        }]
        summary = compute_metrics(results)
        overall = summary["overall"]["baseline"]
        self.assertEqual(overall["total_input_tokens"], 100)
        self.assertEqual(overall["total_output_tokens"], 50)
        self.assertEqual(overall["truncated_percentage"], 100.0)
        self.assertEqual(overall["average_model_wall_time_ms"], 2.0)
        self.assertEqual(
            summary["generation_by_model_and_mode"]["model"]["baseline"]
            ["average_generation_duration"],
            500,
        )
        self.assertEqual(
            summary["generation_by_model_mode_and_call_type"]
            ["model"]["baseline"]["final_generation"]
            ["average_server_total_duration"],
            800,
        )

    def test_negative_evaluator_codes_are_counted_as_distinct_failures(self):
        passed, counts, primary_error = summarize_test_outcomes(
            [True, True, -2, -3, -4]
        )

        self.assertEqual(passed, 2)
        self.assertEqual(counts["PASSED"], 2)
        self.assertEqual(counts["WRONG_ANSWER"], 1)
        self.assertEqual(counts["TIME_LIMIT_EXCEEDED"], 1)
        self.assertEqual(counts["RUNTIME_ERROR"], 1)
        self.assertEqual(primary_error, "RUNTIME_ERROR")

    def test_benchmark_retains_mixed_test_outcomes(self):
        results = [{
            "problem": {"difficulty": "easy", "platform": "test"},
            "modes": {"baseline": {
                "passed": False,
                "pass_rate": 1 / 3,
                "latency_ms": 10,
                "test_outcome_counts": {
                    "PASSED": 1,
                    "WRONG_ANSWER": 1,
                    "TIME_LIMIT_EXCEEDED": 1,
                    "RUNTIME_ERROR": 0,
                },
            }},
        }]

        summary = compute_metrics(results)
        self.assertEqual(summary["overall"]["baseline"]["pass_rate"], 0.0)
        self.assertEqual(summary["by_error_type"]["baseline"]["PASSED"], 1)
        self.assertEqual(
            summary["by_error_type"]["baseline"]["WRONG_ANSWER"], 1
        )
        self.assertEqual(
            summary["by_error_type"]["baseline"]["TIME_LIMIT_EXCEEDED"], 1
        )


if __name__ == "__main__":
    unittest.main()
