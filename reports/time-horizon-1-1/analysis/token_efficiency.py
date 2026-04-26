"""Token efficiency vs 50% time horizon.

For each agent we compute:
  - tokens_per_task = median(tokens_count) over its runs in runs.jsonl
  - horizon_50 = the 50% time horizon (in minutes) from the same logistic
    regression used elsewhere in this report (invsqrt-task-weighted, headline config)
  - tokens_per_minute_of_human_work = tokens_per_task / horizon_50

We then plot tokens_per_minute_of_human_work (log y) vs horizon_50 (log x), and
connect models that share the same scaffold within a developer / lineage.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from horizon.compute_task_weights import add_task_weight_columns  # noqa: E402
from horizon.utils.logistic import (  # noqa: E402
    get_x_for_quantile,
    logistic_regression,
)

logger = logging.getLogger(__name__)


HEADLINE_REGULARIZATION = 0.00001
HEADLINE_WEIGHTING = "invsqrt_task_weight"


def fit_horizon_50(runs: pd.DataFrame) -> dict[str, float]:
    """Replicate the headline logistic regression and return p50 horizons in minutes."""
    horizons: dict[str, float] = {}
    for agent, agent_runs in runs.groupby("alias"):
        x = np.log2(agent_runs["human_minutes"].values).reshape(-1, 1)
        y = agent_runs["score_binarized"].values.astype(float)
        w = agent_runs[HEADLINE_WEIGHTING].values
        if np.all(y == 0) or np.all(y == 1):
            horizons[str(agent)] = float("nan")
            continue
        model = logistic_regression(
            x, y, sample_weight=w, regularization=HEADLINE_REGULARIZATION
        )
        x_p50 = get_x_for_quantile(model, 0.50)
        horizons[str(agent)] = float(np.exp2(x_p50))
    return horizons


def median_tokens_per_task(runs: pd.DataFrame) -> dict[str, float]:
    """Median tokens spent on a task by an agent, across its runs."""
    out: dict[str, float] = {}
    for agent, agent_runs in runs.groupby("alias"):
        tokens = agent_runs["tokens_count"].dropna()
        out[str(agent)] = float(np.median(tokens)) if len(tokens) else float("nan")
    return out


# Plot configuration: a "lineage" is a (developer, scaffold) pair.  The order
# of the agents within each lineage is the order they will be connected by a line.
LINEAGES: list[dict] = [
    {
        "key": "anthropic_react",
        "label": "Anthropic / react",
        "color": "#D55E00",
        "marker": "o",
        "agents": [
            "Claude 3 Opus (Inspect)",
            "Claude 3.5 Sonnet (Old) (Inspect)",
            "Claude 3.5 Sonnet (New) (Inspect)",
            "Claude 3.7 Sonnet (Inspect)",
            "Claude 4 Opus (Inspect)",
            "Claude 4.1 Opus (Inspect)",
            "Claude Opus 4.5 (Inspect)",
            "Claude Opus 4.6 (Inspect)",
        ],
    },
    {
        "key": "openai_reasoning_triframe",
        "label": "OpenAI reasoning / triframe",
        "color": "#009E73",
        "marker": "D",
        "agents": [
            "o1 (Inspect)",
            "o3 (Inspect)",
            "GPT-5 (Inspect)",
            "GPT-5.1-Codex-Max (Inspect)",
            "GPT-5.2",
        ],
    },
    {
        "key": "openai_nonreasoning_react",
        "label": "OpenAI non-reasoning / react",
        "color": "#56B870",
        "marker": "o",
        "linestyle": ":",
        "agents": [
            "GPT-4 1106 (Inspect)",
            "GPT-4 Turbo (Inspect)",
            "GPT-4o (Inspect)",
        ],
    },
    {
        "key": "google",
        "label": "Gemini 3 Pro",
        "color": "#0072B2",
        "marker": "o",
        "agents": ["Gemini 3 Pro"],
    },
]

# Short labels for the plot.
SHORT_LABELS = {
    "Claude 3 Opus (Inspect)": "Claude 3 Opus",
    "Claude 3.5 Sonnet (Old) (Inspect)": "Claude 3.5 Sonnet (Jun)",
    "Claude 3.5 Sonnet (New) (Inspect)": "Claude 3.5 Sonnet (Oct)",
    "Claude 3.7 Sonnet (Inspect)": "Claude 3.7 Sonnet",
    "Claude 4 Opus (Inspect)": "Claude 4 Opus",
    "Claude 4.1 Opus (Inspect)": "Claude 4.1 Opus",
    "Claude Opus 4.5 (Inspect)": "Claude Opus 4.5",
    "Claude Opus 4.6 (Inspect)": "Claude Opus 4.6",
    "GPT-4 1106 (Inspect)": "GPT-4 1106",
    "GPT-4 Turbo (Inspect)": "GPT-4 Turbo",
    "GPT-4o (Inspect)": "GPT-4o",
    "o1 (Inspect)": "o1",
    "o3 (Inspect)": "o3",
    "GPT-5 (Inspect)": "GPT-5",
    "GPT-5.1-Codex-Max (Inspect)": "GPT-5.1 Codex-Max",
    "GPT-5.2": "GPT-5.2",
    "Gemini 3 Pro": "Gemini 3 Pro",
}


def all_lineage_agents() -> list[str]:
    seen: list[str] = []
    for lineage in LINEAGES:
        for agent in lineage["agents"]:
            if agent not in seen:
                seen.append(agent)
    return seen


def format_minutes(m: float) -> str:
    if m < 60:
        return f"{int(round(m))}m"
    if m < 60 * 24:
        return f"{m / 60:.0f}h"
    return f"{m / (60 * 24):.0f}d"


def yearly_drop_factor(per_agent: pd.DataFrame, lineage_key: str) -> float:
    """Least-squares fit of log(tokens-per-minute-of-human-work) vs release date.

    Returns the multiplicative drop per year (e.g. 7.7 means 7.7x cheaper per year)."""
    sub = per_agent.loc[per_agent["lineage"] == lineage_key].copy()
    sub = sub.sort_values("release_date").dropna(
        subset=["horizon_50", "tokens_per_minute", "release_date"]
    )
    if len(sub) < 2:
        return float("nan")
    t = (sub["release_date"] - sub["release_date"].iloc[0]).dt.days.values / 365.0
    y = np.log(sub["tokens_per_minute"].values)
    slope, _ = np.polyfit(t, y, 1)
    return float(np.exp(-slope))


def main(
    runs_file: pathlib.Path,
    release_dates_file: pathlib.Path,
    output_plot: pathlib.Path,
    output_table: pathlib.Path,
) -> None:
    runs = pd.read_json(runs_file, lines=True, convert_dates=False)
    runs = runs[runs["alias"].isin(all_lineage_agents())].copy()
    runs = runs[runs["tokens_count"].notna()]
    runs = runs[runs["score_binarized"].notna()]
    # Recompute weights since we filtered.
    runs = runs.drop(
        columns=[c for c in ("equal_task_weight", "invsqrt_task_weight") if c in runs]
    )
    runs = add_task_weight_columns(runs)

    horizons = fit_horizon_50(runs)
    tokens = median_tokens_per_task(runs)
    release_dates = yaml.safe_load(release_dates_file.read_text())["date"]

    rows = []
    for lineage in LINEAGES:
        for agent in lineage["agents"]:
            h = horizons.get(agent)
            t = tokens.get(agent)
            if h is None or t is None or np.isnan(h) or np.isnan(t):
                logger.warning("Missing data for %s; skipping", agent)
                continue
            rows.append(
                {
                    "agent": agent,
                    "lineage": lineage["key"],
                    "horizon_50": h,
                    "median_tokens_per_task": t,
                    "tokens_per_minute": t / h,
                    "release_date": pd.Timestamp(release_dates.get(agent)),
                }
            )
    per_agent = pd.DataFrame(rows)

    output_table.parent.mkdir(parents=True, exist_ok=True)
    per_agent.sort_values(["lineage", "release_date"]).to_csv(output_table, index=False)
    logger.info("Wrote table to %s", output_table)

    for key in ("anthropic_react", "openai_reasoning_triframe"):
        sub = per_agent[per_agent["lineage"] == key].sort_values("release_date")
        if len(sub) < 2:
            continue
        years = (
            sub["release_date"].iloc[-1] - sub["release_date"].iloc[0]
        ).days / 365.0
        total_drop = (
            sub["tokens_per_minute"].iloc[0] / sub["tokens_per_minute"].iloc[-1]
        )
        logger.info(
            "%s: %.1fx total over %.2f years (~%.2fx/yr regression)",
            key,
            total_drop,
            years,
            yearly_drop_factor(per_agent, key),
        )

    _make_plot(per_agent, output_plot)


def _make_plot(per_agent: pd.DataFrame, output_plot: pathlib.Path) -> None:
    plt.style.use(str(REPO_ROOT / "reports/time-horizon-1-1/matplotlibrc"))
    fig, ax = plt.subplots(figsize=(13, 8))

    for lineage in LINEAGES:
        sub = per_agent[per_agent["lineage"] == lineage["key"]].sort_values(
            "release_date"
        )
        if sub.empty:
            continue
        ax.plot(
            sub["horizon_50"],
            sub["tokens_per_minute"],
            color=lineage["color"],
            marker=lineage["marker"],
            markersize=8,
            linewidth=2,
            linestyle=lineage.get("linestyle", "-"),
            label=lineage["label"],
        )
        for _, row in sub.iterrows():
            ax.annotate(
                SHORT_LABELS.get(row["agent"], row["agent"]),
                (row["horizon_50"], row["tokens_per_minute"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=10,
                color=lineage["color"],
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("50% time horizon (length of human task AI matches)")
    ax.set_ylabel("Median tokens per minute of human work")
    fig.suptitle(
        "Token efficiency: who's actually getting cheaper?",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=15,
    )
    fig.text(
        0.06,
        0.93,
        "Connecting models within a fixed scaffold. Lower-right = more capability per token.",
        fontsize=11,
        color="#444",
    )

    minute_ticks = [4, 15, 60, 240, 60 * 24]
    minute_labels = ["4m", "15m", "1h", "4h", "1d"]
    ax.set_xticks(minute_ticks)
    ax.set_xticklabels(minute_labels)
    ax.grid(True, which="major", alpha=0.25)

    anthropic_yr = yearly_drop_factor(per_agent, "anthropic_react")
    if anthropic_yr and not np.isnan(anthropic_yr):
        a = per_agent[per_agent["lineage"] == "anthropic_react"].sort_values(
            "release_date"
        )
        # Anchor between Claude Opus 4.5 and 4.6 (the last two points), nudged below.
        x_anchor = a["horizon_50"].iloc[-2]
        y_anchor = a["tokens_per_minute"].iloc[-2]
        ax.annotate(
            f"~{anthropic_yr:.1f}x more efficient per year",
            xy=(x_anchor, y_anchor),
            xytext=(x_anchor * 0.55, y_anchor * 0.55),
            fontsize=11,
            color="#D55E00",
        )

    openai_yr = yearly_drop_factor(per_agent, "openai_reasoning_triframe")
    if openai_yr and not np.isnan(openai_yr):
        o = per_agent[per_agent["lineage"] == "openai_reasoning_triframe"].sort_values(
            "release_date"
        )
        x = o["horizon_50"].iloc[-1]
        y = o["tokens_per_minute"].iloc[-1]
        ax.annotate(
            f"~{openai_yr:.1f}x per year",
            xy=(x, y),
            xytext=(x * 1.25, y * 1.8),
            fontsize=11,
            color="#009E73",
        )

    ax.legend(loc="lower left", frameon=False)
    fig.text(
        0.99,
        0.01,
        "Source: METR eval-analysis-public · benchmark_results_1_1.yaml + runs.jsonl",
        ha="right",
        fontsize=9,
        color="#888",
    )
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=200, bbox_inches="tight")
    logger.info("Wrote plot to %s", output_plot)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-file",
        type=pathlib.Path,
        default=REPO_ROOT
        / "reports/time-horizon-1-1/data/raw/runs.jsonl",
    )
    parser.add_argument(
        "--release-dates",
        type=pathlib.Path,
        default=REPO_ROOT / "data/external/release_dates.yaml",
    )
    parser.add_argument(
        "--output-plot",
        type=pathlib.Path,
        default=REPO_ROOT
        / "reports/time-horizon-1-1/analysis/output/token_efficiency.png",
    )
    parser.add_argument(
        "--output-table",
        type=pathlib.Path,
        default=REPO_ROOT
        / "reports/time-horizon-1-1/analysis/output/token_efficiency.csv",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
    main(args.runs_file, args.release_dates, args.output_plot, args.output_table)
