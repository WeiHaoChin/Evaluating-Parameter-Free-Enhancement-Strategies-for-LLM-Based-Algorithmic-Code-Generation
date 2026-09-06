import time
import unittest
from unittest.mock import patch

from backend.lcb_runner.evaluation.testing_util import (
    grade_stdio,
    numerically_equal_lines,
)


class EvaluatorOutputComparisonTests(unittest.TestCase):
    def test_numeric_comparison_preserves_livecodebench_semantics(self):
        equivalent = [
            ("1 2 3", "1 2 3"),
            ("1.0 2e0 -0", "1 2 0.0"),
            ("+001 -002", "1 -2"),
            ("Infinity -Infinity", "Infinity -Infinity"),
        ]
        different = [
            ("1 2 3", "1 2 4"),
            ("1 2", "1 2 3"),
            ("one 2", "1 2"),
        ]

        for prediction, expected in equivalent:
            self.assertTrue(numerically_equal_lines(prediction, expected))
        for prediction, expected in different:
            self.assertFalse(numerically_equal_lines(prediction, expected))

    def test_large_permutation_pattern_keeps_all_recorded_outcomes(self):
        size = 20_000
        expected = " ".join(str(value) for value in range(1, size + 1))
        wrong = "2 1 " + " ".join(str(value) for value in range(3, size + 1))
        outputs = [wrong] * 6 + [expected] * 44
        code = f"print('{' '.join(wrong.split())}')"

        started = time.perf_counter()
        with patch(
            "backend.lcb_runner.evaluation.testing_util.signal.alarm",
            create=True,
        ):
            results, metadata = grade_stdio(code, [""] * 50, outputs, timeout=6)
        elapsed = time.perf_counter() - started

        self.assertEqual(results, [True] * 6 + [-2] * 44)
        self.assertEqual(len(metadata["failure_details"]), 44)
        self.assertEqual(metadata["failure_details"][0]["test_idx"], 6)
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
