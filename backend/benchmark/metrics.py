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
            "first_gen_pass_count": 0,
            "second_gen_pass_count": 0,
            "total": 0,
            "latencies": [],
            "by_difficulty": defaultdict(lambda: {"pass": 0, "total": 0}),
            "by_platform": defaultdict(lambda: {"pass": 0, "total": 0}),
            "error_types": defaultdict(int),
        }
    )

    for result in results:
        problem = result["problem"]
        difficulty = problem.get("difficulty", "unknown")
        platform = problem.get("platform", "unknown")

        for mode_label, mode_result in result["modes"].items():
            stats = mode_stats[mode_label]

            stats["total"] += 1
            stats["latencies"].append(mode_result.get("latency_ms", 0))

            passed = mode_result.get("passed", False)
            if passed:
                stats["pass_count"] += 1

            if mode_result.get("first_gen_passed", False):
                stats["first_gen_pass_count"] += 1

            if mode_result.get("second_gen_passed", False):
                stats["second_gen_pass_count"] += 1

            stats["by_difficulty"][difficulty]["total"] += 1
            if passed:
                stats["by_difficulty"][difficulty]["pass"] += 1

            stats["by_platform"][platform]["total"] += 1
            if passed:
                stats["by_platform"][platform]["pass"] += 1

            error_type = mode_result.get("error_type") or "SUCCESS"
            stats["error_types"][error_type] += 1

    metrics = {
        "overall": {},
        "by_difficulty": {},
        "by_platform": {},
        "by_error_type": {},
    }

    baseline_rate = None
    rag_only_rate = None

    for mode_label, stats in mode_stats.items():
        total = stats["total"]
        if total == 0:
            continue

        pass_rate = stats["pass_count"] / total
        first_gen_pass_rate = stats["first_gen_pass_count"] / total
        second_gen_pass_rate = stats["second_gen_pass_count"] / total
        avg_latency = (
            sum(stats["latencies"]) / len(stats["latencies"])
            if stats["latencies"]
            else 0
        )

        mode_metrics = {
            "pass_rate": pass_rate,
            "first_gen_pass_rate": first_gen_pass_rate,
            "second_gen_pass_rate": second_gen_pass_rate,
            "avg_latency_ms": avg_latency,
            "total_problems": total,
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

    return metrics
