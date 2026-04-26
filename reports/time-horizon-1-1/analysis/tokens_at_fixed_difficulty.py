"""Counterpoint to the headline tokens-per-minute-of-human-work plot.

The headline metric T/H (median tokens / 50% horizon) is a ratio of two
moving things.  If "token efficiency" was really improving along a
parallel curve to capability, then the *numerator alone* should drop:
holding task difficulty constant, newer models should spend fewer tokens
on a task of equivalent human workload.

This script computes that directly: per agent, the median tokens spent
on tasks where the human baseline is 15-60 minutes.  Then it shows how
that quantity evolves over time per lineage.

Result: it doesn't drop.  Anthropic and OpenAI-reasoning frontiers are
both essentially flat at fixed difficulty -- which means the headline
metric's improvement is essentially all capability growth (the
denominator), not efficiency (the numerator).
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
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from horizon.compute_task_weights import add_task_weight_columns  # noqa: E402
from token_efficiency import LINEAGES, SHORT_LABELS, all_lineage_agents  # noqa: E402

logger = logging.getLogger(__name__)

BUCKET_LO, BUCKET_HI, BUCKET_LABEL = 15, 60, "15-60 minute"


def median_tokens_in_bucket(
    runs: pd.DataFrame, lo: float, hi: float
) -> dict[str, tuple[float, int]]:
    sub = runs[(runs["human_minutes"] >= lo) & (runs["human_minutes"] < hi)]
    return {
        str(a): (float(np.median(g["tokens_count"].dropna())), int(len(g)))
        for a, g in sub.groupby("alias")
    }


def yearly_factor(values: pd.Series, dates: pd.Series) -> float:
    mask = values.notna() & dates.notna() & (values > 0)
    v, d = values[mask], dates[mask]
    if len(v) < 2:
        return float("nan")
    t = (d - d.min()).dt.days.values / 365.0
    slope, _ = np.polyfit(t, np.log(v.values), 1)
    return float(np.exp(slope))


def build_table(
    runs: pd.DataFrame, release_dates: dict[str, str]
) -> pd.DataFrame:
    bucket = median_tokens_in_bucket(runs, BUCKET_LO, BUCKET_HI)
    overall = {
        str(a): float(np.median(g["tokens_count"].dropna()))
        for a, g in runs.groupby("alias")
    }
    rows = []
    for lineage in LINEAGES:
        for agent in lineage["agents"]:
            bt, n = bucket.get(agent, (float("nan"), 0))
            rows.append(
                {
                    "agent": agent,
                    "lineage": lineage["key"],
                    "lineage_label": lineage["label"],
                    "color": lineage["color"],
                    "marker": lineage["marker"],
                    "linestyle": lineage.get("linestyle", "-"),
                    "median_tokens_overall": overall.get(agent, float("nan")),
                    "median_tokens_at_fixed_difficulty": bt,
                    "n_runs_in_bucket": n,
                    "release_date": pd.Timestamp(release_dates.get(agent)),
                }
            )
    return pd.DataFrame(rows)


def make_plot(per_agent: pd.DataFrame, out_path: pathlib.Path) -> None:
    plt.style.use(str(REPO_ROOT / "reports/time-horizon-1-1/matplotlibrc"))
    fig, ax = plt.subplots(figsize=(13, 7.5))

    summary_lines: list[tuple[str, str]] = []

    for lineage in LINEAGES:
        sub = per_agent[per_agent["lineage"] == lineage["key"]].sort_values(
            "release_date"
        ).dropna(subset=["median_tokens_at_fixed_difficulty"])
        if sub.empty:
            continue

        # Faint connecting line + bold markers, with a separate bold trend line
        # so flatness pops out visually.
        ax.plot(
            sub["release_date"],
            sub["median_tokens_at_fixed_difficulty"],
            color=lineage["color"],
            marker=lineage["marker"],
            markersize=8,
            linewidth=1,
            linestyle=":",
            alpha=0.55,
            label=lineage["label"],
        )

        if len(sub) >= 2:
            t = (
                sub["release_date"] - sub["release_date"].min()
            ).dt.days.values / 365.0
            y = np.log(sub["median_tokens_at_fixed_difficulty"].values)
            slope, intercept = np.polyfit(t, y, 1)
            trend_y = np.exp(intercept + slope * t)
            ax.plot(
                sub["release_date"], trend_y,
                color=lineage["color"], linewidth=2.5,
                linestyle=lineage.get("linestyle", "-"),
            )
            f = float(np.exp(slope))
            if 0.95 <= f <= 1.05:
                tag = "essentially flat"
            elif f > 1:
                tag = f"{f:.2f}x / yr (rising)"
            else:
                tag = f"{f:.2f}x / yr (falling)"
            summary_lines.append((lineage["color"], f"{lineage['label']}: {tag}"))

        # Label first and last point in each lineage only, to avoid clutter.
        for idx in (0, -1) if len(sub) > 1 else (0,):
            row = sub.iloc[idx]
            ax.annotate(
                SHORT_LABELS.get(row["agent"], row["agent"]),
                (row["release_date"], row["median_tokens_at_fixed_difficulty"]),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=9,
                color=lineage["color"],
            )

    ax.set_yscale("log")
    ax.set_ylabel(f"Median tokens spent on a {BUCKET_LABEL} task")
    ax.set_xlabel("Release date")
    ax.grid(True, which="major", alpha=0.25)

    # Headline + subtitle.  Place above the axes so they don't overlap the data.
    fig.suptitle(
        "At fixed task difficulty, token use isn't really dropping",
        x=0.06, y=0.97, ha="left", fontsize=16,
    )
    fig.text(
        0.06, 0.91,
        f"Median tokens spent on tasks where the human baseline is {BUCKET_LABEL}.\n"
        f"If \"tokens / minute of human work\" was falling because of efficiency, "
        f"this should fall too.  It doesn't.",
        fontsize=10.5, color="#444",
    )

    # Per-lineage trend summary, placed in the data-free middle band of the plot.
    box_x, box_y = 0.27, 0.5
    ax.text(
        box_x, box_y, "Trend at fixed difficulty:",
        transform=ax.transAxes, fontsize=10.5, color="#222",
        ha="left", va="top", weight="bold",
    )
    for i, (color, line) in enumerate(summary_lines):
        ax.text(
            box_x, box_y - 0.05 - i * 0.045, line,
            transform=ax.transAxes, fontsize=10.5, color=color,
            ha="left", va="top",
        )

    # Eye-popping gap between Anthropic and OpenAI reasoning.
    anth = per_agent[per_agent["lineage"] == "anthropic_react"][
        "median_tokens_at_fixed_difficulty"
    ].dropna()
    oai = per_agent[per_agent["lineage"] == "openai_reasoning_triframe"][
        "median_tokens_at_fixed_difficulty"
    ].dropna()
    if len(anth) and len(oai):
        ratio = float(np.exp(np.mean(np.log(oai))) / np.exp(np.mean(np.log(anth))))
        ax.text(
            0.27, 0.65,
            f"OpenAI reasoning models burn ~{ratio:.0f}x the tokens of\n"
            f"Anthropic models on the same task (reasoning trace).",
            transform=ax.transAxes, fontsize=10.5, color="#333",
            ha="left", va="top",
        )

    ax.legend(loc="center left", frameon=False, fontsize=10.5,
              bbox_to_anchor=(0.0, 0.55))

    fig.text(
        0.99, 0.005,
        "Source: METR eval-analysis-public . runs.jsonl",
        ha="right", fontsize=9, color="#888",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.86))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info("Wrote %s", out_path)


def main(
    runs_file: pathlib.Path,
    release_dates_file: pathlib.Path,
    out_plot: pathlib.Path,
    out_table: pathlib.Path,
) -> None:
    runs = pd.read_json(runs_file, lines=True, convert_dates=False)
    runs = runs[runs["alias"].isin(all_lineage_agents())].copy()
    runs = runs[runs["tokens_count"].notna() & runs["score_binarized"].notna()]
    runs = runs.drop(
        columns=[c for c in ("equal_task_weight", "invsqrt_task_weight") if c in runs]
    )
    runs = add_task_weight_columns(runs)

    release_dates = yaml.safe_load(release_dates_file.read_text())["date"]
    per_agent = build_table(runs, release_dates)

    out_table.parent.mkdir(parents=True, exist_ok=True)
    per_agent.drop(columns=["color", "marker", "linestyle"]).sort_values(
        ["lineage", "release_date"]
    ).to_csv(out_table, index=False)
    logger.info("Wrote table to %s", out_table)

    for lineage in LINEAGES:
        sub = per_agent[per_agent["lineage"] == lineage["key"]].sort_values(
            "release_date"
        ).dropna(subset=["median_tokens_at_fixed_difficulty"])
        if len(sub) < 2:
            continue
        f = yearly_factor(
            sub["median_tokens_at_fixed_difficulty"], sub["release_date"]
        )
        logger.info("%-32s %.2f× / yr at fixed difficulty", lineage["label"], f)

    make_plot(per_agent, out_plot)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-file", type=pathlib.Path,
        default=REPO_ROOT / "reports/time-horizon-1-1/data/raw/runs.jsonl",
    )
    parser.add_argument(
        "--release-dates", type=pathlib.Path,
        default=REPO_ROOT / "data/external/release_dates.yaml",
    )
    parser.add_argument(
        "--out-plot", type=pathlib.Path,
        default=REPO_ROOT
        / "reports/time-horizon-1-1/analysis/output/tokens_at_fixed_difficulty.png",
    )
    parser.add_argument(
        "--out-table", type=pathlib.Path,
        default=REPO_ROOT
        / "reports/time-horizon-1-1/analysis/output/tokens_at_fixed_difficulty.csv",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
    main(args.runs_file, args.release_dates, args.out_plot, args.out_table)
