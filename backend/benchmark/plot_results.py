"""Plot the comparable metrics available in the named benchmark results.

When a model has both an original result and a result whose filename ends in
`` 2.json``, the latter is selected.  Timestamped runs and testcase exports are
not considered.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fyp-matplotlib"))
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_PATTERN = re.compile(r"^(?P<family>.+?GPTOSS)(?: (?P<variant>2|100q))?\.json$")
VARIANT_PRIORITY = {"2": 2, None: 1, "100q": 0}
MODE_ORDER = ("baseline", "rag_only", "textgrad_only", "full")
TEXTGRAD_PAIRS = (
    ("baseline", "textgrad_only", "Baseline → TextGrad-only"),
    ("rag_only", "full", "RAG-only → Combined"),
)
MODE_LABELS = {
    "baseline": "Baseline",
    "rag_only": "RAG-only",
    "textgrad_only": "TextGrad-only",
    "full": "Combined",
}
MODEL_LABELS = {
    "DeepseekGPTOSS": "DeepSeek 4 Flash",
    "Qwen3.5GPTOSS": "Qwen 3.5",
    "GemmaGPTOSS": "Gemma 4",
    "GPTOSSGPTOSS": "GPT-OSS 120B",
    "MistralGPTOSS": "Mistral Large 3",
    "Kimi2.6GPTOSS": "Kimi K2.6",
}


def normalized_grades(
    mode_result: dict[str, Any], expected_total: int | None = None
) -> list[bool] | None:
    """Return one pass/fail value per expected testcase group.

    Pipeline exceptions have ``total_tests == 0`` because judging never began.
    When another mode supplies the problem's expected group count, those
    unexecuted groups are retained as failed outcomes instead of being omitted.
    """
    total = max(
        int(mode_result.get("total_tests") or 0),
        int(expected_total or 0),
    )
    if total <= 0:
        return None

    flattened: list[bool] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        else:
            flattened.append(value is True)

    visit(mode_result.get("graded_list") or [])
    return (flattened[:total] + [False] * total)[:total]


def select_result_files(results_dir: Path) -> dict[str, Path]:
    """Select one result per model family, preferring the ``2`` variant."""
    candidates: dict[str, list[tuple[int, Path]]] = {}
    for path in results_dir.glob("*.json"):
        match = RESULT_PATTERN.fullmatch(path.name)
        if not match:
            continue
        family = match.group("family")
        variant = match.group("variant")
        candidates.setdefault(family, []).append((VARIANT_PRIORITY[variant], path))

    if not candidates:
        raise FileNotFoundError(f"No named benchmark result files found in {results_dir}")

    return {
        family: max(files, key=lambda candidate: candidate[0])[1]
        for family, files in candidates.items()
    }


def summarize_outcome_transitions(
    results: list[dict[str, Any]],
    before_mode: str,
    after_mode: str,
    *,
    require_group_comparability_for_problems: bool = True,
) -> dict[str, Any]:
    problem_fixes = problem_regressions = 0
    testcase_fixes = testcase_regressions = 0
    paired_problems = paired_testcase_groups = 0
    accuracy_deltas = []
    for result in results:
        before = result["modes"][before_mode]
        after = result["modes"][after_mode]
        expected_total = max(
            int(mode_result.get("total_tests") or 0)
            for mode_result in result["modes"].values()
        )
        before_passed = bool(before.get("passed"))
        after_passed = bool(after.get("passed"))
        if not require_group_comparability_for_problems:
            paired_problems += 1
            problem_fixes += int(not before_passed and after_passed)
            problem_regressions += int(before_passed and not after_passed)

        before_grades = normalized_grades(before, expected_total)
        after_grades = normalized_grades(after, expected_total)
        if (
            before_grades is None
            or after_grades is None
            or len(before_grades) != len(after_grades)
        ):
            continue

        if require_group_comparability_for_problems:
            paired_problems += 1
            problem_fixes += int(not before_passed and after_passed)
            problem_regressions += int(before_passed and not after_passed)
        paired_testcase_groups += len(before_grades)
        testcase_fixes += sum(
            not old and new for old, new in zip(before_grades, after_grades)
        )
        testcase_regressions += sum(
            old and not new for old, new in zip(before_grades, after_grades)
        )
        accuracy_deltas.append(
            100
            * (
                sum(after_grades) / len(after_grades)
                - sum(before_grades) / len(before_grades)
            )
        )

    return {
        "problem_fixes": problem_fixes,
        "problem_regressions": problem_regressions,
        "testcase_fixes": testcase_fixes,
        "testcase_regressions": testcase_regressions,
        "accuracy_delta": sum(accuracy_deltas) / len(accuracy_deltas),
        "paired_problems": paired_problems,
        "paired_testcase_groups": paired_testcase_groups,
    }


def summarize_testcase_outcomes(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Count every expected testcase group, including unexecuted groups."""
    outcomes = {
        mode: {
            "PASSED": 0,
            "WRONG_ANSWER": 0,
            "RUNTIME_ERROR": 0,
            "TIME_LIMIT_EXCEEDED": 0,
            "NOT_EVALUATED": 0,
        }
        for mode in MODE_ORDER
    }
    for result in results:
        expected_total = max(
            int(mode_result.get("total_tests") or 0)
            for mode_result in result["modes"].values()
        )
        for mode in MODE_ORDER:
            mode_result = result["modes"][mode]
            counts = mode_result.get("test_outcome_counts")
            if isinstance(counts, dict):
                recorded = 0
                for outcome in (
                    "PASSED",
                    "WRONG_ANSWER",
                    "RUNTIME_ERROR",
                    "TIME_LIMIT_EXCEEDED",
                ):
                    count = int(counts.get(outcome) or 0)
                    outcomes[mode][outcome] += count
                    recorded += count
                if recorded < expected_total:
                    outcomes[mode]["NOT_EVALUATED"] += expected_total - recorded
            else:
                # Older pipeline-exception records contain one problem-level
                # EXCEPTION but no per-group outcomes. Keep all affected group
                # slots in the denominator and label them as not evaluated.
                outcomes[mode]["NOT_EVALUATED"] += expected_total
    return outcomes


def paired_initial_record(
    result: dict[str, Any], parent_mode: str
) -> dict[str, Any]:
    """Return the initial generation reused by a derived non-TextGrad mode."""
    record = next(
        (
            item
            for item in result["modes"][parent_mode].get("generation_records", [])
            if item.get("call_type") == "initial_generation"
        ),
        None,
    )
    if record is None:
        raise ValueError(f"Missing paired initial generation for {parent_mode}")
    return record


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as result_file:
        data = json.load(result_file)

    overall = data["summary"]["overall"]
    missing = [mode for mode in MODE_ORDER if mode not in overall]
    if missing:
        raise ValueError(f"{path.name} is missing benchmark modes: {', '.join(missing)}")

    records_by_mode: dict[str, list[dict[str, Any]]] = {
        "baseline": [
            paired_initial_record(result, "textgrad_only")
            for result in data["results"]
        ],
        "rag_only": [
            paired_initial_record(result, "full")
            for result in data["results"]
        ],
        "textgrad_only": [
            record
            for result in data["results"]
            for record in result["modes"]["textgrad_only"].get(
                "generation_records", []
            )
        ],
        "full": [
            record
            for result in data["results"]
            for record in result["modes"]["full"].get("generation_records", [])
        ],
    }

    workflow_samples: dict[str, list[float]] = {}
    for mode, parent_mode in (
        ("baseline", "textgrad_only"),
        ("rag_only", "full"),
    ):
        samples = []
        for result in data["results"]:
            record = paired_initial_record(result, parent_mode)
            retrieval_ms = (
                result["modes"][parent_mode].get("timings", {}).get(
                    "retrieval_ms", 0
                )
                or 0
                if mode == "rag_only"
                else 0
            )
            generation_ms = float(record.get("client_duration_ns") or 0) / 1_000_000
            samples.append((retrieval_ms + generation_ms) / 1000)
        workflow_samples[mode] = samples

    for mode in ("textgrad_only", "full"):
        workflow_samples[mode] = [
            (
                (timing.get("retrieval_ms") or 0)
                + (timing.get("generation_workflow_ms") or 0)
            )
            / 1000
            for result in data["results"]
            if isinstance(
                (timing := result["modes"][mode].get("timings")), dict
            )
        ]

    workflow_seconds = {}
    workflow_counts = {}
    for mode in MODE_ORDER:
        samples = workflow_samples[mode]
        if not samples:
            raise ValueError(f"{path.name} has no workflow timings for {mode}")
        # This is the user-facing generation workflow: RAG retrieval plus all
        # solution-generation and TextGrad calls. Code judging is excluded.
        workflow_seconds[mode] = sum(samples) / len(samples)
        workflow_counts[mode] = len(samples)

    tokens_per_problem = {
        mode: sum(
            int(record.get("prompt_tokens") or 0)
            + int(record.get("output_tokens") or 0)
            for record in records_by_mode[mode]
        )
        / len(data["results"])
        for mode in MODE_ORDER
    }
    solution_generation_call_types = {
        "initial_generation",
        "regenerated_solution",
        "final_generation",
    }
    final_generation_call_types = {"regenerated_solution", "final_generation"}
    # Compare the single output produced by each configuration. The initial
    # candidate reused by TextGrad is already represented by Baseline/RAG-only
    # and must not be counted again in the TextGrad-only/Combined bars.
    configuration_output_records = {
        "baseline": records_by_mode["baseline"],
        "rag_only": records_by_mode["rag_only"],
        "textgrad_only": [
            record
            for record in records_by_mode["textgrad_only"]
            if record.get("call_type") in final_generation_call_types
        ],
        "full": [
            record
            for record in records_by_mode["full"]
            if record.get("call_type") in final_generation_call_types
        ],
    }
    truncation_counts = {
        mode: sum(record.get("truncated") is True for record in records)
        for mode, records in configuration_output_records.items()
    }
    generation_counts = {
        mode: len(records) for mode, records in configuration_output_records.items()
    }
    truncation_rate = {
        mode: 100 * truncation_counts[mode] / generation_counts[mode]
        for mode in MODE_ORDER
    }
    textgrad_internal_records = {
        mode: [
            record
            for record in records_by_mode[mode]
            if record.get("call_type") not in solution_generation_call_types
        ]
        for mode in ("textgrad_only", "full")
    }
    textgrad_internal_truncation_counts = {
        mode: sum(record.get("truncated") is True for record in records)
        for mode, records in textgrad_internal_records.items()
    }
    textgrad_internal_call_counts = {
        mode: len(records) for mode, records in textgrad_internal_records.items()
    }
    textgrad_internal_truncation_rate = {
        mode: (
            100
            * textgrad_internal_truncation_counts[mode]
            / textgrad_internal_call_counts[mode]
        )
        for mode in ("textgrad_only", "full")
    }

    textgrad_transitions = {
        pair_label: summarize_outcome_transitions(
            data["results"], before_mode, after_mode
        )
        for before_mode, after_mode, pair_label in TEXTGRAD_PAIRS
    }
    rag_transitions = summarize_outcome_transitions(
        data["results"],
        "baseline",
        "rag_only",
        require_group_comparability_for_problems=False,
    )
    testcase_outcomes = summarize_testcase_outcomes(data["results"])

    return {
        "pass_rate": {mode: overall[mode]["pass_rate"] * 100 for mode in MODE_ORDER},
        "group_accuracy": {
            mode: overall[mode]["macro_evaluator_group_accuracy"] * 100
            for mode in MODE_ORDER
        },
        "workflow_seconds": workflow_seconds,
        "workflow_counts": workflow_counts,
        "workflow_samples": workflow_samples,
        "tokens_per_problem": tokens_per_problem,
        "truncation_rate": truncation_rate,
        "truncation_labels": {
            mode: (
                f"{truncation_counts[mode]}/{generation_counts[mode]}\n"
                f"{truncation_rate[mode]:.1f}%"
            )
            for mode in MODE_ORDER
        },
        "textgrad_internal_truncation_rate": textgrad_internal_truncation_rate,
        "textgrad_internal_truncation_labels": {
            mode: (
                f"{textgrad_internal_truncation_counts[mode]}/"
                f"{textgrad_internal_call_counts[mode]}\n"
                f"{textgrad_internal_truncation_rate[mode]:.1f}%"
            )
            for mode in ("textgrad_only", "full")
        },
        "testcase_outcomes": testcase_outcomes,
        "platforms": data["summary"]["by_platform"],
        "textgrad_transitions": textgrad_transitions,
        "rag_transitions": rag_transitions,
    }


def plot_grouped_bars(
    families: list[str],
    values_by_family: dict[str, dict[str, float]],
    output: Path,
    *,
    title: str,
    ylabel: str,
    value_format: str,
    tick_suffix: str,
    footer: str,
    bar_labels_by_family: dict[str, dict[str, str]] | None = None,
    label_fontsize: float = 9,
) -> None:
    x = np.arange(len(families))
    width = 0.19
    colors = ("#64748B", "#0EA5E9", "#F59E0B", "#8B5CF6")

    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.15, top=0.84)
    maximum = max(
        values_by_family[family][mode]
        for family in families
        for mode in MODE_ORDER
    )
    upper_limit = max(10, np.ceil((maximum * 1.15) / 10) * 10)

    for index, (mode, color) in enumerate(zip(MODE_ORDER, colors)):
        values = [values_by_family[family][mode] for family in families]
        offset = (index - (len(MODE_ORDER) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=MODE_LABELS[mode],
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.bar_label(
            bars,
            labels=(
                [bar_labels_by_family[family][mode] for family in families]
                if bar_labels_by_family is not None
                else [f"{format(value, value_format)}{tick_suffix}" for value in values]
            ),
            padding=3,
            fontsize=label_fontsize,
        )

    ax.set_title(title, fontsize=18, weight="bold", pad=18)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x, [MODEL_LABELS.get(family, family) for family in families])
    ax.set_ylim(0, upper_limit)
    ax.yaxis.set_major_formatter(
        lambda value, _position: f"{value:.0f}{tick_suffix}"
    )
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.text(0.5, 0.035, footer, ha="center", color="#475569", fontsize=10)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def plot_textgrad_internal_truncation(
    families: list[str], metrics: dict[str, dict[str, Any]], output: Path
) -> None:
    """Plot truncation only for TextGrad critic and optimisation calls."""
    modes = ("textgrad_only", "full")
    colors = ("#F59E0B", "#8B5CF6")
    x = np.arange(len(families))
    width = 0.34

    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.84)
    maximum = max(
        metrics[family]["textgrad_internal_truncation_rate"][mode]
        for family in families
        for mode in modes
    )
    upper_limit = max(0.2, np.ceil(maximum * 14) / 10)

    for index, (mode, color) in enumerate(zip(modes, colors)):
        values = [
            metrics[family]["textgrad_internal_truncation_rate"][mode]
            for family in families
        ]
        offset = (index - 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=MODE_LABELS[mode],
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.bar_label(
            bars,
            labels=[
                metrics[family]["textgrad_internal_truncation_labels"][mode]
                for family in families
            ],
            padding=3,
            fontsize=8,
        )

    ax.set_title(
        "TextGrad Internal API Call Truncation Rate by Model",
        fontsize=18,
        weight="bold",
        pad=18,
    )
    ax.set_ylabel("Truncated internal API calls (%)", fontsize=12)
    ax.set_xticks(x, [MODEL_LABELS.get(family, family) for family in families])
    ax.set_ylim(0, upper_limit)
    ax.yaxis.set_major_formatter(lambda value, _position: f"{value:.1f}%")
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.text(
        0.5,
        0.035,
        "Critic evaluation, critique-backward and prompt-optimisation calls only; solution generations excluded",
        ha="center",
        color="#475569",
        fontsize=10,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def plot_mean_correctness_comparison(
    families: list[str], metrics: dict[str, dict[str, Any]], output: Path
) -> None:
    pass_means = np.array([
        np.mean([metrics[family]["pass_rate"][mode] for family in families])
        for mode in MODE_ORDER
    ])
    group_means = np.array([
        np.mean([metrics[family]["group_accuracy"][mode] for family in families])
        for mode in MODE_ORDER
    ])

    x = np.arange(len(MODE_ORDER))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.17, top=0.82)

    pass_bars = ax.bar(
        x - width / 2,
        pass_means,
        width,
        label="Pass@1",
        color="#64748B",
        edgecolor="white",
        linewidth=0.9,
    )
    group_bars = ax.bar(
        x + width / 2,
        group_means,
        width,
        label="Macro testcase-group accuracy",
        color="#0EA5E9",
        edgecolor="#0369A1",
        linewidth=0.9,
        hatch="//",
    )
    ax.bar_label(
        pass_bars,
        labels=[f"{value:.2f}%" for value in pass_means],
        padding=4,
        fontsize=9,
    )
    ax.bar_label(
        group_bars,
        labels=[f"{value:.2f}%" for value in group_means],
        padding=4,
        fontsize=9,
    )

    ax.set_title("Mean Correctness Across Models", fontsize=18, weight="bold", pad=45)
    ax.set_ylabel("Mean accuracy (%)", fontsize=12)
    ax.set_xticks(x, [MODE_LABELS[mode] for mode in MODE_ORDER])
    ax.set_ylim(0, 70)
    ax.yaxis.set_major_formatter(lambda value, _position: f"{value:.0f}%")
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        frameon=False,
    )
    fig.text(
        0.5,
        0.045,
        "Equal-weight average across the six models; testcase-group accuracy is first macro-averaged by problem",
        ha="center",
        color="#475569",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def plot_testcase_outcomes(
    families: list[str], metrics: dict[str, dict[str, Any]], output: Path
) -> None:
    categories = (
        "WRONG_ANSWER",
        "RUNTIME_ERROR",
        "TIME_LIMIT_EXCEEDED",
        "NOT_EVALUATED",
    )
    category_labels = (
        "Wrong Answer",
        "Runtime Error",
        "Time Limit",
        "Not Evaluated",
    )
    mode_colors = ("#64748B", "#0EA5E9", "#F59E0B", "#8B5CF6")
    markers = ("o", "s", "^", "D")
    offsets = np.linspace(-0.21, 0.21, len(MODE_ORDER))

    aggregate_counts = {
        mode: {
            category: sum(
                metrics[family]["testcase_outcomes"][mode].get(category, 0)
                for family in families
            )
            for category in categories
        }
        for mode in MODE_ORDER
    }
    totals = {
        mode: sum(
            sum(metrics[family]["testcase_outcomes"][mode].values())
            for family in families
        )
        for mode in MODE_ORDER
    }

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.16, top=0.79)
    y = np.arange(len(categories))

    maximum = 0.0
    for mode, color, marker, offset in zip(MODE_ORDER, mode_colors, markers, offsets):
        values = np.array([
            100 * aggregate_counts[mode][category] / totals[mode]
            if totals[mode]
            else 0
            for category in categories
        ])
        maximum = max(maximum, float(values.max()))
        positions = y + offset
        ax.scatter(
            values,
            positions,
            s=75,
            color=color,
            marker=marker,
            label=MODE_LABELS[mode],
            zorder=3,
        )
        for value, position in zip(values, positions):
            ax.annotate(
                f"{value:.1f}%",
                (value, position),
                xytext=(7, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                color="#334155",
            )

    ax.set_yticks(y, category_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, maximum + 5)
    ax.xaxis.set_major_formatter(lambda value, _position: f"{value:.0f}%")
    ax.set_xlabel("Share of all evaluator testcase groups")
    ax.grid(axis="x", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(
        "Testcase-Group Failure Outcomes by Pipeline Mode",
        fontsize=19,
        weight="bold",
        pad=62,
    )
    ax.legend(
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
    )
    fig.text(
        0.5,
        0.045,
        (
            "Aggregated across all six models; passed groups are excluded from the figure, "
            "but remain in the percentage denominator"
        ),
        ha="center",
        color="#475569",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def plot_platform_pass_rate(
    families: list[str],
    metrics: dict[str, dict[str, Any]],
    output: Path,
    platform: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.15, top=0.80)
    x = np.arange(len(families))
    width = 0.19
    colors = ("#64748B", "#0EA5E9", "#F59E0B", "#8B5CF6")

    maximum = 0.0
    for index, (mode, color) in enumerate(zip(MODE_ORDER, colors)):
        values = [
            metrics[family]["platforms"][platform][mode]["pass_rate"] * 100
            for family in families
        ]
        maximum = max(maximum, *values)
        offset = (index - (len(MODE_ORDER) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=MODE_LABELS[mode], color=color)
        ax.bar_label(bars, labels=[f"{value:.0f}%" for value in values], padding=3, fontsize=9)

    sample_size = metrics[families[0]]["platforms"][platform]["baseline"]["total"]
    fig.suptitle(
        f"{platform.title()} Benchmark Pass Rate (n={sample_size} problems)",
        fontsize=18,
        weight="bold",
        y=0.965,
    )
    ax.set_ylim(0, min(110, max(10, np.ceil(((maximum + 8) / 10)) * 10)))
    ax.yaxis.set_major_formatter(lambda value, _position: f"{value:.0f}%")
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_ylabel("Problems Passed (%)")
    ax.set_xticks(x, [MODEL_LABELS.get(family, family) for family in families])
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        frameon=False,
    )
    fig.text(
        0.5,
        0.035,
        (
            "Single-problem result; percentages are either 0% or 100% and should not be "
            "generalized"
            if sample_size == 1
            else "Platform-specific subset of the 100-problem benchmark"
        ),
        ha="center",
        color="#475569",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def plot_textgrad_transitions(
    families: list[str],
    metrics: dict[str, dict[str, Any]],
    output: Path,
    *,
    level: str,
) -> None:
    fixes_key = f"{level}_fixes"
    regressions_key = f"{level}_regressions"
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.82, wspace=0.08)
    x = np.arange(len(families))

    for ax, (_, _, pair_label) in zip(axes, TEXTGRAD_PAIRS):
        fixes = [metrics[family]["textgrad_transitions"][pair_label][fixes_key] for family in families]
        regressions = [
            -metrics[family]["textgrad_transitions"][pair_label][regressions_key]
            for family in families
        ]
        fixed_bars = ax.bar(x, fixes, color="#22C55E", width=0.66, label="Fixed")
        regressed_bars = ax.bar(
            x, regressions, color="#EF4444", width=0.66, label="Regressed"
        )
        ax.bar_label(fixed_bars, labels=[f"+{value}" for value in fixes], padding=3)
        ax.bar_label(
            regressed_bars,
            labels=[str(value) for value in regressions],
            padding=3,
        )
        ax.axhline(0, color="#334155", linewidth=1)
        ax.set_title(pair_label, fontsize=13, weight="bold")
        ax.set_xticks(
            x,
            [MODEL_LABELS.get(family, family) for family in families],
            rotation=18,
            ha="right",
        )
        ax.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)

    unit_title = "Whole-Problem" if level == "problem" else "Testcase-Group"
    axes[0].set_ylabel(f"{unit_title} transitions (regressions shown below zero)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        f"TextGrad {unit_title} Fixes and Regressions",
        fontsize=19,
        weight="bold",
        y=0.96,
    )
    fig.legend(
        handles,
        labels,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        frameon=False,
    )
    fig.text(
        0.5,
        0.035,
        (
            "Paired initial → TextGrad-final comparison; unexecuted testcase groups are "
            "counted as failed"
        ),
        ha="center",
        color="#475569",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def plot_rag_transitions(
    families: list[str],
    metrics: dict[str, dict[str, Any]],
    output: Path,
    *,
    level: str,
) -> None:
    fixes_key = f"{level}_fixes"
    regressions_key = f"{level}_regressions"
    fixes = [metrics[family]["rag_transitions"][fixes_key] for family in families]
    regressions = [
        -metrics[family]["rag_transitions"][regressions_key]
        for family in families
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.82)
    x = np.arange(len(families))
    fixed_bars = ax.bar(
        x,
        fixes,
        color="#22C55E",
        width=0.66,
        label="RAG-only success (Baseline failed)",
    )
    regressed_bars = ax.bar(
        x,
        regressions,
        color="#EF4444",
        width=0.66,
        label="Baseline-only success (RAG-only failed)",
    )
    ax.bar_label(fixed_bars, labels=[f"+{value}" for value in fixes], padding=3)
    ax.bar_label(
        regressed_bars,
        labels=[str(value) for value in regressions],
        padding=3,
    )
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_xticks(
        x,
        [MODEL_LABELS.get(family, family) for family in families],
        rotation=12,
        ha="right",
    )
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)

    unit_title = "Whole-Problem" if level == "problem" else "Testcase-Group"
    ax.set_ylabel(f"{unit_title} outcome changes (Baseline-only below zero)")
    ax.set_title(
        f"Baseline vs RAG-Only: {unit_title} Outcome Changes",
        fontsize=19,
        weight="bold",
        pad=48,
    )
    ax.legend(
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        frameon=False,
    )
    fig.text(
        0.5,
        0.04,
        (
            "Baseline vs RAG-only on the same problems; responses "
            "were generated independently"
        ),
        ha="center",
        color="#475569",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def plot_textgrad_accuracy_delta(
    families: list[str], metrics: dict[str, dict[str, Any]], output: Path
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.15, top=0.84)
    x = np.arange(len(families))
    width = 0.36
    colors = ("#F59E0B", "#8B5CF6")
    all_values = []

    for index, ((_, _, pair_label), color) in enumerate(zip(TEXTGRAD_PAIRS, colors)):
        values = [
            metrics[family]["textgrad_transitions"][pair_label]["accuracy_delta"]
            for family in families
        ]
        all_values.extend(values)
        bars = ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            color=color,
            label=pair_label,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:+.1f} pp" for value in values],
            padding=3,
            fontsize=9,
        )

    limit = max(5, np.ceil(max(abs(value) for value in all_values) / 5) * 5 + 5)
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_title(
        "TextGrad Change in Testcase-Group Accuracy",
        fontsize=18,
        weight="bold",
        pad=18,
    )
    ax.set_ylabel("Mean paired change (percentage points)")
    ax.set_xticks(x, [MODEL_LABELS.get(family, family) for family in families])
    ax.yaxis.set_major_formatter(lambda value, _position: f"{value:+.0f} pp")
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.text(
        0.5,
        0.035,
        "Macro average of within-problem accuracy changes on comparable paired results",
        ha="center",
        color="#475569",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def plot_workflow_percentiles(
    families: list[str], metrics: dict[str, dict[str, Any]], output: Path
) -> None:
    colors = ("#64748B", "#0EA5E9", "#F59E0B", "#8B5CF6")
    percentiles = np.arange(0, 101)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.86, hspace=0.30)

    for family, ax in zip(families, axes.flat):
        for mode, color in zip(MODE_ORDER, colors):
            values = np.percentile(
                metrics[family]["workflow_samples"][mode], percentiles
            )
            ax.plot(
                percentiles,
                values,
                color=color,
                linewidth=2,
                label=MODE_LABELS[mode],
            )
        for reference in (50, 90, 95):
            ax.axvline(reference, color="#94A3B8", linewidth=0.8, linestyle="--")
        ax.set_yscale("log")
        ax.set_title(MODEL_LABELS.get(family, family), fontsize=12, weight="bold")
        ax.set_xticks((0, 25, 50, 75, 90, 100))
        ax.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)

    axes[0, 0].set_ylabel("Workflow time (seconds, log scale)")
    axes[1, 0].set_ylabel("Workflow time (seconds, log scale)")
    for ax in axes[1, :]:
        ax.set_xlabel("Percentile")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Generation Workflow Time Percentile Curves",
        fontsize=19,
        weight="bold",
        y=0.97,
    )
    fig.legend(
        handles,
        labels,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        frameon=False,
    )
    fig.text(
        0.5,
        0.035,
        "Dashed guides mark P50, P90, and P95; lower curves indicate faster and more consistent workflows",
        ha="center",
        color="#475569",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def plot_workflow_latency_summary(
    families: list[str], metrics: dict[str, dict[str, Any]], output: Path
) -> None:
    """Show typical and slow-case latency without a logarithmic axis."""
    colors = ("#64748B", "#0EA5E9", "#F59E0B", "#8B5CF6")
    summaries = {
        family: {
            mode: np.percentile(metrics[family]["workflow_samples"][mode], (50, 95))
            for mode in MODE_ORDER
        }
        for family in families
    }
    largest_p95 = max(
        values[mode][1]
        for values in summaries.values()
        for mode in MODE_ORDER
    )
    x_limit = np.ceil(largest_p95 / 50) * 50

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.08, top=0.86, hspace=0.30)
    y_positions = np.arange(len(MODE_ORDER))[::-1]

    for family, ax in zip(families, axes.flat):
        for y, mode, color in zip(y_positions, MODE_ORDER, colors):
            p50, p95 = summaries[family][mode]
            ax.plot((p50, p95), (y, y), color=color, linewidth=4, solid_capstyle="round")
            ax.scatter(p50, y, s=55, color=color, marker="o", zorder=3)
            ax.scatter(p95, y, s=65, color=color, marker="D", zorder=3)
            ax.annotate(
                f"{p50:.0f}s → {p95:.0f}s",
                ((p50 + p95) / 2, y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#334155",
            )
        ax.set_title(MODEL_LABELS.get(family, family), fontsize=12, weight="bold")
        ax.set_xlim(0, x_limit)
        ax.set_yticks(y_positions, [MODE_LABELS[mode] for mode in MODE_ORDER])
        ax.grid(axis="x", color="#CBD5E1", linewidth=0.7, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)

    for ax in axes[1, :]:
        ax.set_xlabel("Workflow time (seconds, linear scale)")
    fig.suptitle(
        "Typical vs Slow Generation Workflow Time",
        fontsize=19,
        weight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.905,
        "● P50 (typical case)     ◆ P95 (slow case)     Shorter and further left is better",
        ha="center",
        color="#334155",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def plot_workflow_boxplots(
    families: list[str], metrics: dict[str, dict[str, Any]], output: Path
) -> None:
    """Plot comparable workflow distributions, leaving Kimi to its own scale."""
    colors = ("#64748B", "#0EA5E9", "#F59E0B", "#8B5CF6")
    display_families = [family for family in families if family != "Kimi2.6GPTOSS"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.08, top=0.86, hspace=0.30)

    for family, ax in zip(display_families, axes.flat):
        samples = [metrics[family]["workflow_samples"][mode] for mode in MODE_ORDER]
        boxplot = ax.boxplot(
            samples,
            orientation="horizontal",
            patch_artist=True,
            tick_labels=[MODE_LABELS[mode] for mode in MODE_ORDER],
            widths=0.55,
            whis=1.5,
            medianprops={"color": "white", "linewidth": 2},
            whiskerprops={"color": "#64748B", "linewidth": 1.2},
            capprops={"color": "#64748B", "linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": "#94A3B8",
                "markeredgecolor": "none",
                "alpha": 0.55,
            },
        )
        for box, color in zip(boxplot["boxes"], colors):
            box.set_facecolor(color)
            box.set_edgecolor(color)
            box.set_alpha(0.88)
        for position, mode in enumerate(MODE_ORDER, start=1):
            median = np.median(metrics[family]["workflow_samples"][mode])
            ax.annotate(
                f"{median:.0f}s",
                (median, position),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#334155",
            )
        ax.set_title(MODEL_LABELS.get(family, family), fontsize=12, weight="bold")
        ax.grid(axis="x", color="#CBD5E1", linewidth=0.7, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)

    for ax in axes.flat[len(display_families):]:
        ax.remove()

    for ax in axes[1, :]:
        ax.set_xlabel("Workflow time (seconds, linear scale)")
    fig.suptitle(
        "Generation Workflow Time Distribution — Comparable-Scale Models",
        fontsize=19,
        weight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.905,
        "Centre line = median • Box = middle 50% • Whiskers = typical range • Dots = unusually slow cases",
        ha="center",
        color="#334155",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def plot_kimi_workflow_boxplot(metrics: dict[str, dict[str, Any]], output: Path) -> None:
    """Give Kimi's much wider distribution a readable, separate linear scale."""
    family = "Kimi2.6GPTOSS"
    if family not in metrics:
        return
    colors = ("#64748B", "#0EA5E9", "#F59E0B", "#8B5CF6")
    samples = [metrics[family]["workflow_samples"][mode] for mode in MODE_ORDER]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.12, top=0.78)
    boxplot = ax.boxplot(
        samples,
        orientation="horizontal",
        patch_artist=True,
        tick_labels=[MODE_LABELS[mode] for mode in MODE_ORDER],
        widths=0.55,
        whis=1.5,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": "#64748B", "linewidth": 1.2},
        capprops={"color": "#64748B", "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "#94A3B8",
            "markeredgecolor": "none",
            "alpha": 0.6,
        },
    )
    for box, color in zip(boxplot["boxes"], colors):
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(0.88)
    for position, mode in enumerate(MODE_ORDER, start=1):
        median = np.median(metrics[family]["workflow_samples"][mode])
        ax.annotate(
            f"Median {median:.0f}s",
            (median, position),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#334155",
        )
    ax.set_title("Kimi K2.6 Generation Workflow Time Distribution", fontsize=18, weight="bold", pad=50)
    ax.text(
        0.5,
        1.08,
        "Centre line = median • Box = middle 50% • Whiskers = typical range • Dots = unusually slow cases",
        transform=ax.transAxes,
        ha="center",
        color="#334155",
        fontsize=10,
    )
    ax.set_xlabel("Workflow time (seconds, linear scale)")
    ax.grid(axis="x", color="#CBD5E1", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def plot_results(results_dir: Path, output: Path) -> tuple[list[Path], list[Path]]:
    selected = select_result_files(results_dir)
    families = sorted(
        selected,
        key=lambda family: list(MODEL_LABELS).index(family)
        if family in MODEL_LABELS
        else len(MODEL_LABELS),
    )
    metrics = {family: load_metrics(selected[family]) for family in families}
    outputs = {
        "pass_rate": output,
        "testcase_group_accuracy": output.with_name(
            "benchmark_testcase_group_accuracy.png"
        ),
        "mean_correctness": output.with_name(
            "benchmark_mean_correctness_comparison.png"
        ),
        "rag_problem_transitions": output.with_name(
            "benchmark_rag_problem_fixes_regressions.png"
        ),
        "rag_testcase_transitions": output.with_name(
            "benchmark_rag_testcase_fixes_regressions.png"
        ),
        "textgrad_problem_transitions": output.with_name(
            "benchmark_textgrad_problem_fixes_regressions.png"
        ),
        "textgrad_testcase_transitions": output.with_name(
            "benchmark_textgrad_testcase_fixes_regressions.png"
        ),
        "token_usage": output.with_name("benchmark_token_usage.png"),
        "workflow_boxplot": output.with_name(
            "benchmark_generation_workflow_boxplot.png"
        ),
        "workflow_boxplot_kimi": output.with_name(
            "benchmark_generation_workflow_boxplot_kimi.png"
        ),
        "truncation_rate": output.with_name("benchmark_truncation_rate.png"),
    }

    plot_grouped_bars(
        families,
        {family: metrics[family]["pass_rate"] for family in families},
        outputs["pass_rate"],
        title="Benchmark Pass Rate by Model and Pipeline Mode",
        ylabel="Problems Passed (%)",
        value_format=".0f",
        tick_suffix="%",
        footer="100 benchmark problems per model",
    )
    plot_grouped_bars(
        families,
        {family: metrics[family]["group_accuracy"] for family in families},
        outputs["testcase_group_accuracy"],
        title="Testcase-Group Accuracy by Model and Pipeline Mode",
        ylabel="Testcase Groups Passed (%)",
        value_format=".1f",
        tick_suffix="%",
        footer="Macro accuracy: each benchmark problem is weighted equally",
    )
    plot_mean_correctness_comparison(families, metrics, outputs["mean_correctness"])
    plot_grouped_bars(
        families,
        {family: metrics[family]["tokens_per_problem"] for family in families},
        outputs["token_usage"],
        title="Recorded API Token Usage by Model and Pipeline Mode",
        ylabel="Mean Tokens per Problem",
        value_format=",.0f",
        tick_suffix="",
        footer="Input + output tokens across every recorded API call, divided by 100 problems",
    )
    plot_grouped_bars(
        families,
        {family: metrics[family]["truncation_rate"] for family in families},
        outputs["truncation_rate"],
        title="Configuration Output-Token Cap-Hit Rate by Model and Pipeline Mode",
        ylabel="Outputs flagged at token limit (%)",
        value_format=".1f",
        tick_suffix="%",
        footer=(
            "One output per problem: Baseline/RAG initial output; "
            "TextGrad/Combined final regenerated output"
        ),
        bar_labels_by_family={
            family: metrics[family]["truncation_labels"] for family in families
        },
        label_fontsize=7.5,
    )
    plot_textgrad_transitions(
        families, metrics, outputs["textgrad_problem_transitions"], level="problem"
    )
    plot_textgrad_transitions(
        families, metrics, outputs["textgrad_testcase_transitions"], level="testcase"
    )
    plot_workflow_boxplots(families, metrics, outputs["workflow_boxplot"])
    plot_kimi_workflow_boxplot(metrics, outputs["workflow_boxplot_kimi"])
    plot_rag_transitions(
        families, metrics, outputs["rag_problem_transitions"], level="problem"
    )
    plot_rag_transitions(
        families, metrics, outputs["rag_testcase_transitions"], level="testcase"
    )
    return [selected[family] for family in families], list(outputs.values())


def main() -> None:
    default_results = Path(__file__).resolve().parent / "results"
    default_output = (
        Path(__file__).resolve().parents[2]
        / "report"
        / "figures"
        / "benchmark_results.png"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
    )
    args = parser.parse_args()

    selected, outputs = plot_results(args.results_dir, args.output)
    print("Saved graphs:")
    for output in outputs:
        print(f"  - {output.resolve()}")
    print("Selected result files:")
    for path in selected:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
