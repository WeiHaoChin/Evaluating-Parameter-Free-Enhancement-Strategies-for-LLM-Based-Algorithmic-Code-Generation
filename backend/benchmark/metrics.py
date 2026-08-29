# benchmark/metrics.py
from typing import Any
from collections import defaultdict


def compute_metrics(results: list[dict]) -> dict:
    """
    Compute benchmark metrics from results.

    Args:
        results: List of problem results from runner

    Returns:
        Dict with aggregated metrics per mode and breakdowns
    """
    mode_stats = defaultdict(
        lambda: {
            "pass_count": 0,
            "second_gen_pass_count": 0,
            "total": 0,
            "latencies": [],
            "generation_count": 0,
            "prompt_tokens": [],
            "output_tokens": [],
            "truncated_count": 0,
            "evaluator_group_rates": [],
            "evaluator_groups_passed": 0,
            "evaluator_groups_total": 0,
            "generation_durations": [],
            "client_durations": [],
            "server_total_durations": [],
            "by_difficulty": defaultdict(lambda: {"pass": 0, "total": 0}),
            "by_platform": defaultdict(lambda: {"pass": 0, "total": 0}),
            "error_types": defaultdict(int),
        }
    )
    generation_by_model_and_mode = defaultdict(
        lambda: defaultdict(lambda: {
            "generation_count": 0, "prompt_tokens": [], "output_tokens": [],
            "truncated_count": 0, "generation_durations": [],
            "client_durations": [], "server_total_durations": [],
        })
    )
    generation_by_model_mode_call_type = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {
            "generation_count": 0, "prompt_tokens": [], "output_tokens": [],
            "truncated_count": 0, "generation_durations": [],
            "client_durations": [], "server_total_durations": [],
        }))
    )

    def add_generation(stats: dict, record: dict) -> None:
        stats["generation_count"] += 1
        if record.get("prompt_tokens") is not None:
            stats["prompt_tokens"].append(record["prompt_tokens"])
        if record.get("output_tokens") is not None:
            stats["output_tokens"].append(record["output_tokens"])
        if record.get("truncated") is True:
            stats["truncated_count"] += 1
        if record.get("eval_duration") is not None:
            stats["generation_durations"].append(record["eval_duration"])
        if record.get("client_duration_ns") is not None:
            stats["client_durations"].append(record["client_duration_ns"])
        if record.get("total_duration") is not None:
            stats["server_total_durations"].append(record["total_duration"])

    def generation_summary(stats: dict) -> dict:
        count = stats["generation_count"]
        input_total = sum(stats["prompt_tokens"])
        output_total = sum(stats["output_tokens"])
        return {
            "generation_count": count,
            "average_input_tokens": input_total / len(stats["prompt_tokens"]) if stats["prompt_tokens"] else None,
            "average_output_tokens": output_total / len(stats["output_tokens"]) if stats["output_tokens"] else None,
            "total_input_tokens": input_total,
            "total_output_tokens": output_total,
            "truncated_generations": stats["truncated_count"],
            "truncated_percentage": (100 * stats["truncated_count"] / count) if count else 0.0,
            "average_generation_duration": (
                sum(stats["generation_durations"]) / len(stats["generation_durations"])
                if stats["generation_durations"] else None
            ),
            "average_model_wall_time_ms": (
                sum(stats["client_durations"]) / len(stats["client_durations"]) / 1_000_000
                if stats["client_durations"] else None
            ),
            "average_server_total_duration": (
                sum(stats["server_total_durations"]) / len(stats["server_total_durations"])
                if stats["server_total_durations"] else None
            ),
            "duration_unit": "nanoseconds",
            "model_wall_time_unit": "milliseconds",
        }

    for result in results:
        problem = result["problem"]
        difficulty = problem.get("difficulty", "unknown")
        platform = problem.get("platform", "unknown")

        for mode_label, mode_result in result["modes"].items():
            stats = mode_stats[mode_label]

            stats["total"] += 1
            stats["latencies"].append(mode_result.get("latency_ms", 0))
            for record in mode_result.get("generation_records", []):
                add_generation(stats, record)
                add_generation(
                    generation_by_model_and_mode[record.get("model") or "unknown"][mode_label],
                    record,
                )
                add_generation(
                    generation_by_model_mode_call_type
                    [record.get("model") or "unknown"]
                    [mode_label]
                    [record.get("call_type") or "generation"],
                    record,
                )

            passed = mode_result.get("passed", False)
            if passed:
                stats["pass_count"] += 1

            if mode_result.get("second_gen_passed", False):
                stats["second_gen_pass_count"] += 1

            passed_tests = mode_result.get("passed_tests")
            total_tests = mode_result.get("total_tests")
            if (
                isinstance(passed_tests, (int, float))
                and isinstance(total_tests, (int, float))
                and total_tests > 0
            ):
                stats["evaluator_group_rates"].append(passed_tests / total_tests)
                stats["evaluator_groups_passed"] += passed_tests
                stats["evaluator_groups_total"] += total_tests
            elif isinstance(mode_result.get("pass_rate"), (int, float)):
                stats["evaluator_group_rates"].append(mode_result["pass_rate"])

            stats["by_difficulty"][difficulty]["total"] += 1
            if passed:
                stats["by_difficulty"][difficulty]["pass"] += 1

            stats["by_platform"][platform]["total"] += 1
            if passed:
                stats["by_platform"][platform]["pass"] += 1

            outcome_counts = mode_result.get("test_outcome_counts")
            if isinstance(outcome_counts, dict):
                for outcome, count in outcome_counts.items():
                    count = int(count or 0)
                    if count > 0:
                        stats["error_types"][outcome] += count
            else:
                # Backward compatibility for result files created before
                # per-test evaluator outcomes were retained.
                error_type = mode_result.get("error_type") or "PASSED"
                stats["error_types"][error_type] += 1

    metrics = {
        "overall": {},
        "by_difficulty": {},
        "by_platform": {},
        "by_error_type": {},
        "generation_by_model_and_mode": {},
        "generation_by_model_mode_and_call_type": {},
    }

    baseline_rate = None
    rag_only_rate = None

    for mode_label, stats in mode_stats.items():
        total = stats["total"]
        if total == 0:
            continue

        pass_rate = stats["pass_count"] / total
        second_gen_pass_rate = stats["second_gen_pass_count"] / total
        avg_latency = (
            sum(stats["latencies"]) / len(stats["latencies"])
            if stats["latencies"]
            else 0
        )

        mode_metrics = {
            "pass_rate": pass_rate,
            "macro_evaluator_group_accuracy": (
                sum(stats["evaluator_group_rates"])
                / len(stats["evaluator_group_rates"])
                if stats["evaluator_group_rates"] else None
            ),
            "evaluator_groups_passed": stats["evaluator_groups_passed"],
            "evaluator_groups_total": stats["evaluator_groups_total"],
            "second_gen_pass_rate": second_gen_pass_rate,
            "avg_latency_ms": avg_latency,
            "total_problems": total,
            **generation_summary(stats),
        }

        if mode_label == "baseline":
            baseline_rate = pass_rate
        elif mode_label == "rag_only":
            rag_only_rate = pass_rate

        if mode_label == "textgrad_only" and baseline_rate is not None:
            mode_metrics["textgrad_delta"] = pass_rate - baseline_rate
        elif mode_label == "full" and rag_only_rate is not None:
            mode_metrics["textgrad_delta"] = pass_rate - rag_only_rate

        metrics["overall"][mode_label] = mode_metrics

        for difficulty, diff_stats in stats["by_difficulty"].items():
            if difficulty not in metrics["by_difficulty"]:
                metrics["by_difficulty"][difficulty] = {}
            metrics["by_difficulty"][difficulty][mode_label] = {
                "pass_rate": (
                    diff_stats["pass"] / diff_stats["total"]
                    if diff_stats["total"] > 0
                    else 0
                ),
                "total": diff_stats["total"],
            }

        for platform, plat_stats in stats["by_platform"].items():
            if platform not in metrics["by_platform"]:
                metrics["by_platform"][platform] = {}
            metrics["by_platform"][platform][mode_label] = {
                "pass_rate": (
                    plat_stats["pass"] / plat_stats["total"]
                    if plat_stats["total"] > 0
                    else 0
                ),
                "total": plat_stats["total"],
            }

        metrics["by_error_type"][mode_label] = dict(stats["error_types"])

    metrics["generation_by_model_and_mode"] = {
        model: {
            mode: generation_summary(stats)
            for mode, stats in modes.items()
        }
        for model, modes in generation_by_model_and_mode.items()
    }
    metrics["generation_by_model_mode_and_call_type"] = {
        model: {
            mode: {
                call_type: generation_summary(stats)
                for call_type, stats in call_types.items()
            }
            for mode, call_types in modes.items()
        }
        for model, modes in generation_by_model_mode_call_type.items()
    }

    return metrics
