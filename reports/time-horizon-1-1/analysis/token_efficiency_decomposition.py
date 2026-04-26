"""Decomposition plots for the token-efficiency story.

The headline metric T/H (median tokens-per-task / 50% horizon) collapses two
different motions into one number.  This script splits them:

  Plot A (two-panel): T over time and H over time, side-by-side.
                      Their gap is the headline metric.
  Plot B: median tokens spent on tasks within a fixed difficulty bucket
          (15-60 min of human work) vs release date.  This holds task
          difficulty constant so we see "tokens per task at fixed workload"
          stripped of capability growth.
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

# Reuse the lineage / labeling config so the two scripts agree.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from token_efficiency import (  # noqa: E402
    HEADLINE_REGULARIZATION,
    HEADLINE_WEIGHTING,
    LINEAGES,
    SHORT_LABELS,
    all_lineage_agents,
)

logger = logging.getLogger(__name__)

CONTROLLED_BUCKET = (15, 60, "15-60 min")


def fit_horizons(runs: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for agent, agent_runs in runs.groupby("alias"):
        x = np.log2(agent_runs["human_minutes"].values).reshape(-1, 1)
        y = agent_runs["score_binarized"].values.astype(float)
        w = agent_runs[HEADLINE_WEIGHTING].values
        if np.all(y == 0) or np.all(y == 1):
            out[str(agent)] = float("nan")
            continue
        model = logistic_regression(
            x, y, sample_weight=w, regularization=HEADLINE_REGULARIZATION
        )
        out[str(agent)] = float(np.exp2(get_x_for_quantile(model, 0.50)))
    return out


def median_tokens(runs: pd.DataFrame) -> dict[str, float]:
    return {
        str(a): float(np.median(g["tokens_count"].dropna()))
        for a, g in runs.groupby("alias")
    }


def median_tokens_in_bucket(
    runs: pd.DataFrame, lo: float, hi: float
) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    sub = runs[(runs["human_minutes"] >= lo) & (runs["human_minutes"] < hi)]
    for agent, g in sub.groupby("alias"):
        toks = g["tokens_count"].dropna()
        out[str(agent)] = (float(np.median(toks)) if len(toks) else float("nan"), len(toks))
    return out


def yearly_factor(values: pd.Series, dates: pd.Series, *, increasing: bool) -> float:
    """Multiplicative change per year fit by least squares on log(value).

    `increasing=True` returns growth-per-year; `False` returns drop-per-year."""
    mask = values.notna() & dates.notna() & (values > 0)
    v, d = values[mask], dates[mask]
    if len(v) < 2:
        return float("nan")
    t = (d - d.min()).dt.days.values / 365.0
    slope, _ = np.polyfit(t, np.log(v.values), 1)
    return float(np.exp(slope) if increasing else np.exp(-slope))


def build_per_agent_table(
    runs: pd.DataFrame, release_dates: dict[str, str]
) -> pd.DataFrame:
    horizons = fit_horizons(runs)
    tokens = median_tokens(runs)
    bucket_tokens = median_tokens_in_bucket(runs, CONTROLLED_BUCKET[0], CONTROLLED_BUCKET[1])

    rows = []
    for lineage in LINEAGES:
        for agent in lineage["agents"]:
            h = horizons.get(agent)
            t = tokens.get(agent)
            bt, n = bucket_tokens.get(agent, (float("nan"), 0))
            if h is None or t is None or np.isnan(h):
                continue
            rows.append(
                {
                    "agent": agent,
                    "lineage": lineage["key"],
                    "lineage_label": lineage["label"],
                    "color": lineage["color"],
                    "marker": lineage["marker"],
                    "linestyle": lineage.get("linestyle", "-"),
                    "horizon_50": h,
                    "median_tokens": t,
                    "bucket_median_tokens": bt,
                    "bucket_n_runs": n,
                    "release_date": pd.Timestamp(release_dates.get(agent)),
                }
            )
    return pd.DataFrame(rows)


def plot_decomposition(per_agent: pd.DataFrame, out_path: pathlib.Path) -> None:
    plt.style.use(str(REPO_ROOT / "reports/time-horizon-1-1/matplotlibrc"))
    fig, (ax_t, ax_h) = plt.subplots(1, 2, figsize=(15, 7), sharex=True)

    for lineage in LINEAGES:
        sub = per_agent[per_agent["lineage"] == lineage["key"]].sort_values("release_date")
        if sub.empty:
            continue
        for ax, ycol in ((ax_t, "median_tokens"), (ax_h, "horizon_50")):
            ax.plot(
                sub["release_date"], sub[ycol],
                color=lineage["color"], marker=lineage["marker"], markersize=8,
                linewidth=2, linestyle=lineage.get("linestyle", "-"),
                label=lineage["label"],
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    SHORT_LABELS.get(row["agent"], row["agent"]),
                    (row["release_date"], row[ycol]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color=lineage["color"],
                )

    for ax in (ax_t, ax_h):
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.25)
        ax.tick_params(axis="x", rotation=30)

    ax_t.set_title("Tokens per task (median over runs)")
    ax_t.set_ylabel("Median tokens per task")
    ax_h.set_title("50% time horizon")
    ax_h.set_ylabel("Minutes of human work")
    minute_ticks = [4, 15, 60, 240, 60 * 24]
    minute_labels = ["4m", "15m", "1h", "4h", "1d"]
    ax_h.set_yticks(minute_ticks)
    ax_h.set_yticklabels(minute_labels)
    ax_h.set_ylim(2, 60 * 24)

    # Per-year growth annotations.
    for ax, ycol, increasing, suffix in (
        (ax_t, "median_tokens", True, "tokens/yr"),
        (ax_h, "horizon_50", True, "horizon/yr"),
    ):
        ann_lines = []
        for lineage in LINEAGES:
            sub = per_agent[per_agent["lineage"] == lineage["key"]].sort_values(
                "release_date"
            )
            if len(sub) < 2:
                continue
            f = yearly_factor(sub[ycol], sub["release_date"], increasing=increasing)
            ann_lines.append((lineage["color"], f"{lineage['label']}: {f:.1f}× / yr"))
        for i, (color, line) in enumerate(ann_lines):
            ax.text(
                0.02, 0.98 - i * 0.06, line,
                transform=ax.transAxes, fontsize=10, color=color, va="top",
            )

    ax_t.legend(loc="lower right", frameon=False, fontsize=10)
    fig.suptitle(
        "Decomposing the headline: tokens-per-task vs capability, separately",
        x=0.06, y=0.995, ha="left", fontsize=15,
    )
    fig.text(
        0.06, 0.96,
        "Headline tokens/min-of-human-work = (tokens) / (horizon). Both grow over time; "
        "their ratio is what \"unit economics\" measures.",
        fontsize=10, color="#444",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info("Wrote %s", out_path)


def plot_controlled(per_agent: pd.DataFrame, out_path: pathlib.Path) -> None:
    plt.style.use(str(REPO_ROOT / "reports/time-horizon-1-1/matplotlibrc"))
    fig, ax = plt.subplots(figsize=(12, 7))

    for lineage in LINEAGES:
        sub = per_agent[per_agent["lineage"] == lineage["key"]].sort_values(
            "release_date"
        )
        sub = sub.dropna(subset=["bucket_median_tokens"])
        if sub.empty:
            continue
        ax.plot(
            sub["release_date"], sub["bucket_median_tokens"],
            color=lineage["color"], marker=lineage["marker"], markersize=8,
            linewidth=2, linestyle=lineage.get("linestyle", "-"),
            label=lineage["label"],
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f'{SHORT_LABELS.get(row["agent"], row["agent"])} (n={row["bucket_n_runs"]})',
                (row["release_date"], row["bucket_median_tokens"]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=8, color=lineage["color"],
            )

    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.25)
    ax.set_ylabel(f"Median tokens spent on {CONTROLLED_BUCKET[2]} tasks")
    ax.set_xlabel("Release date")

    # Per-year change annotations.
    ann_lines = []
    for lineage in LINEAGES:
        sub = per_agent[per_agent["lineage"] == lineage["key"]].sort_values(
            "release_date"
        ).dropna(subset=["bucket_median_tokens"])
        if len(sub) < 2:
            continue
        f = yearly_factor(
            sub["bucket_median_tokens"], sub["release_date"], increasing=True
        )
        if 0.97 <= f <= 1.03:
            tag = "≈ flat"
        elif f > 1:
            tag = f"{f:.2f}× / yr (growth)"
        else:
            tag = f"{f:.2f}× / yr (drop)"
        ann_lines.append((lineage["color"], f"{lineage['label']}: {tag}"))
    for i, (color, line) in enumerate(ann_lines):
        ax.text(
            0.98, 0.98 - i * 0.05, line,
            transform=ax.transAxes, fontsize=10, color=color, va="top", ha="right",
        )

    fig.suptitle(
        "Controlled-difficulty: tokens spent on a fixed task-length bucket",
        x=0.06, y=0.995, ha="left", fontsize=15,
    )
    fig.text(
        0.06, 0.94,
        f"Median tokens-per-task restricted to runs where the human baseline is "
        f"{CONTROLLED_BUCKET[2]}.  Removes the capability-growth confound.",
        fontsize=10, color="#444",
    )
    ax.legend(loc="center left", frameon=False, fontsize=10, bbox_to_anchor=(0.0, 0.55))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info("Wrote %s", out_path)


def main(
    runs_file: pathlib.Path,
    release_dates_file: pathlib.Path,
    out_decomp: pathlib.Path,
    out_controlled: pathlib.Path,
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
    per_agent = build_per_agent_table(runs, release_dates)

    out_table.parent.mkdir(parents=True, exist_ok=True)
    per_agent.drop(columns=["color", "marker", "linestyle"]).sort_values(
        ["lineage", "release_date"]
    ).to_csv(out_table, index=False)
    logger.info("Wrote table to %s", out_table)

    for lineage in LINEAGES:
        sub = per_agent[per_agent["lineage"] == lineage["key"]].sort_values(
            "release_date"
        )
        if len(sub) < 2:
            continue
        t_yr = yearly_factor(sub["median_tokens"], sub["release_date"], increasing=True)
        h_yr = yearly_factor(sub["horizon_50"], sub["release_date"], increasing=True)
        b_yr = yearly_factor(
            sub["bucket_median_tokens"], sub["release_date"], increasing=True
        )
        logger.info(
            "%-32s tokens %.2f×/yr  horizon %.2f×/yr  bucket-tokens %.2f×/yr",
            lineage["label"], t_yr, h_yr, b_yr,
        )

    plot_decomposition(per_agent, out_decomp)
    plot_controlled(per_agent, out_controlled)


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
        "--out-decomposition", type=pathlib.Path,
        default=REPO_ROOT / "reports/time-horizon-1-1/analysis/output/token_efficiency_decomposition.png",
    )
    parser.add_argument(
        "--out-controlled", type=pathlib.Path,
        default=REPO_ROOT / "reports/time-horizon-1-1/analysis/output/token_efficiency_controlled_difficulty.png",
    )
    parser.add_argument(
        "--out-table", type=pathlib.Path,
        default=REPO_ROOT / "reports/time-horizon-1-1/analysis/output/token_efficiency_decomposition.csv",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
    main(
        args.runs_file, args.release_dates,
        args.out_decomposition, args.out_controlled, args.out_table,
    )
