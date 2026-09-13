import unittest
from unittest.mock import patch

import rag_handler


class SolutionRagContextTests(unittest.TestCase):
    def test_queries_four_knowledge_chunks_and_one_problem(self):
        knowledge = [{"chunk_type": "theory"}]
        problem = [{"chunk_type": "problem_statement"}]

        with (
            patch.object(
                rag_handler,
                "query_rag",
                side_effect=[knowledge, problem],
            ) as query,
            patch.object(
                rag_handler,
                "format_solution_rag_context",
                return_value="formatted",
            ),
        ):
            results, context = rag_handler.query_solution_rag("task")

        self.assertEqual(results, knowledge + problem)
        self.assertEqual(context, "formatted")
        self.assertEqual(query.call_args_list[0].kwargs["n_results"], 4)
        self.assertEqual(
            query.call_args_list[0].kwargs["filters"],
            {"chunk_type": {"$in": ["editorial", "theory"]}},
        )
        self.assertEqual(query.call_args_list[1].kwargs["n_results"], 1)
        self.assertEqual(
            query.call_args_list[1].kwargs["filters"],
            {"chunk_type": {"$eq": "problem_statement"}},
        )
        self.assertEqual(
            query.call_args_list[0].kwargs["min_similarity"],
            rag_handler.DEFAULT_MIN_SIMILARITY,
        )

    def test_formatter_labels_the_analogous_problem(self):
        knowledge = [{
            "title": "Shortest paths",
            "source": "cp-algorithms",
            "chunk_type": "theory",
            "similarity": 0.9,
            "text": "Theory text",
        }]
        problem = [{
            "title": "Related task",
            "source": "atcoder",
            "chunk_type": "problem_statement",
            "similarity": 0.8,
            "text": "Problem text",
        }]

        context = rag_handler.format_solution_rag_context(knowledge, problem)

        self.assertIn("## Retrieved solution references", context)
        self.assertIn("## Analogous problem", context)
        self.assertIn("This is a different problem", context)
        self.assertLess(context.index("Theory text"), context.index("Problem text"))


if __name__ == "__main__":
    unittest.main()
