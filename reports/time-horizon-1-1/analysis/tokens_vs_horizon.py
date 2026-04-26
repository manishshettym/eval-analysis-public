"""Tokens-per-task vs 50% time horizon, both at fixed difficulty.

This is the artifact-free version of Tom's plot.  His plot puts T/H on the
y-axis and H on the x-axis -- both contain H, so the line slopes downward
even when there's no real efficiency signal (the algebraic baseline is
slope -1 on log-log).

Here we plot T (median tokens at fixed difficulty) on y and H (50% horizon)
on x.  No shared variable.  Now the slope means something:

  slope < 0 : real efficiency -- as horizon grows, tokens-per-task drop
  slope = 0 : tokens flat -- no efficiency, just capability
  slope > 0 : newer/more-capable models actively use more tokens

Tom's framing implies slope clearly < 0.  Let's see what the data says.
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
from horizon.utils.logistic import (  # noqa: E402
    get_x_for_quantile,
    logistic_regression,
)
from token_efficiency import (  # noqa: E402
    HEADLINE_REGULARIZATION,
    HEADLINE_WEIGHTING,
    LINEAGES,
    SHORT_LABELS,
    all_lineage_agents,
)

logger = logging.getLogger(__name__)

BUCKET_LO, BUCKET_HI, BUCKET_LABEL = 15, 60, "15-60 min"
MIN_RUNS = 5


def fit_horizons(runs: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for agent, agent_runs in runs.groupby("alias"):
        x = np.log2(agent_runs["human_minutes"].values).reshape(-1, 1)
        y = agent_runs["score_binarized"].values.astype(float)
        w = agent_runs[HEADLINE_WEIGHTING].values
        if np.all(y == 0) or np.all(y == 1):
            continue
        model = logistic_regression(
            x, y, sample_weight=w, regularization=HEADLINE_REGULARIZATION
        )
        out[str(agent)] = float(np.exp2(get_x_for_quantile(model, 0.50)))
    return out


def median_tokens_in_bucket(
    runs: pd.DataFrame, lo: float, hi: float
) -> dict[str, float]:
    sub = runs[(runs["human_minutes"] >= lo) & (runs["human_minutes"] < hi)]
    out: dict[str, float] = {}
    for agent, g in sub.groupby("alias"):
        toks = g["tokens_count"].dropna()
        if len(toks) >= MIN_RUNS:
            out[str(agent)] = float(np.median(toks))
    return out


def build_table(
    runs: pd.DataFrame, release_dates: dict[str, str]
) -> pd.DataFrame:
    horizons = fit_horizons(runs)
    tokens = median_tokens_in_bucket(runs, BUCKET_LO, BUCKET_HI)
    rows = []
    for lineage in LINEAGES:
        for agent in lineage["agents"]:
            if agent not in horizons or agent not in tokens:
                continue
            rows.append(
                {
                    "agent": agent,
                    "lineage": lineage["key"],
                    "lineage_label": lineage["label"],
                    "color": lineage["color"],
                    "marker": lineage["marker"],
                    "linestyle": lineage.get("linestyle", "-"),
                    "horizon_50": horizons[agent],
                    "tokens_at_fixed_difficulty": tokens[agent],
                    "release_date": pd.Timestamp(release_dates.get(agent)),
                }
            )
    return pd.DataFrame(rows)


def make_plot(per_agent: pd.DataFrame, out_path: pathlib.Path) -> None:
    plt.style.use(str(REPO_ROOT / "reports/time-horizon-1-1/matplotlibrc"))
    fig, ax = plt.subplots(figsize=(13, 8))

    summary: list[tuple[str, str, float]] = []
    for lineage in LINEAGES:
        sub = per_agent[per_agent["lineage"] == lineage["key"]].sort_values(
            "release_date"
        )
        if sub.empty:
            continue

        # Faint trajectory in release order + bold markers.
        ax.plot(
            sub["horizon_50"], sub["tokens_at_fixed_difficulty"],
            color=lineage["color"], marker=lineage["marker"], markersize=9,
            linewidth=1.2, linestyle=":", alpha=0.55,
            label=lineage["label"],
        )

        # OLS fit in log-log space => slope is "elasticity of T wrt H".
        if len(sub) >= 2:
            log_h = np.log(sub["horizon_50"].values)
            log_t = np.log(sub["tokens_at_fixed_difficulty"].values)
            slope, intercept = np.polyfit(log_h, log_t, 1)
            xs = np.linspace(log_h.min(), log_h.max(), 50)
            ys = intercept + slope * xs
            ax.plot(
                np.exp(xs), np.exp(ys),
                color=lineage["color"], linewidth=2.5,
                linestyle=lineage.get("linestyle", "-"),
            )
            summary.append((lineage["color"], lineage["label"], float(slope)))

        # Label every point.
        for _, row in sub.iterrows():
            ax.annotate(
                SHORT_LABELS.get(row["agent"], row["agent"]),
                (row["horizon_50"], row["tokens_at_fixed_difficulty"]),
                textcoords="offset points", xytext=(7, 5),
                fontsize=9, color=lineage["color"],
            )

    # Reference: Tom's plot has slope = -1 baked in by algebra.  Draw a faint
    # comparison line so the reader can see what "the artifact alone" would
    # look like.
    if not per_agent.empty:
        h_range = np.array([per_agent["horizon_50"].min(),
                            per_agent["horizon_50"].max()])
        # Anchor the slope=-1 reference at the geometric midpoint of the
        # Anthropic line for visual alignment.
        anth = per_agent[per_agent["lineage"] == "anthropic_react"]
        anchor_h = float(np.exp(np.mean(np.log(anth["horizon_50"]))))
        anchor_t = float(np.exp(np.mean(np.log(anth["tokens_at_fixed_difficulty"]))))
        ref_t = anchor_t * (anchor_h / h_range)
        ax.plot(
            h_range, ref_t,
            color="#888", linewidth=1.2, linestyle=(0, (4, 4)), alpha=0.7,
        )
        ax.text(
            h_range[1], ref_t[1] * 1.05,
            "slope -1: what Tom's T/H-vs-H plot bakes in",
            color="#666", fontsize=9, ha="right", va="bottom",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("50% time horizon (minutes of human work)")
    ax.set_ylabel(f"Median tokens spent on a {BUCKET_LABEL} task")
    ax.grid(True, which="major", alpha=0.25)

    minute_ticks = [4, 15, 60, 240, 60 * 24]
    minute_labels = ["4m", "15m", "1h", "4h", "1d"]
    ax.set_xticks(minute_ticks)
    ax.set_xticklabels(minute_labels)

    fig.suptitle(
        "Tokens vs capability, both at fixed difficulty",
        x=0.06, y=0.985, ha="left", fontsize=16,
    )
    fig.text(
        0.06, 0.91,
        f"Each point: one model.  Y = median tokens on a {BUCKET_LABEL} task.  "
        "X = 50% time horizon.\n"
        "A real efficiency story would need a clear negative slope.  "
        "The dashed grey line is the slope = -1 that Tom's plot bakes in by algebra.",
        fontsize=10.5, color="#444",
    )

    # In-axes summary: slope per lineage, with interpretation.
    box_x, box_y = 0.985, 0.98
    ax.text(
        box_x, box_y, "Fitted slope (elasticity of T wrt H):",
        transform=ax.transAxes, fontsize=10.5, color="#222",
        ha="right", va="top", weight="bold",
    )
    for i, (color, label, slope) in enumerate(summary):
        if abs(slope) < 0.1:
            tag = f"{slope:+.2f}  ~ flat"
        elif slope < 0:
            tag = f"{slope:+.2f}  some efficiency"
        else:
            tag = f"{slope:+.2f}  anti-efficient"
        ax.text(
            box_x, box_y - 0.05 - i * 0.045, f"{label}: {tag}",
            transform=ax.transAxes, fontsize=10.5, color=color,
            ha="right", va="top",
        )

    ax.legend(loc="lower left", frameon=False, fontsize=10.5)
    fig.text(
        0.99, 0.005,
        "Source: METR eval-analysis-public . runs.jsonl",
        ha="right", fontsize=9, color="#888",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.85))
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
    logger.info("Wrote %s", out_table)

    for lineage in LINEAGES:
        sub = per_agent[per_agent["lineage"] == lineage["key"]].sort_values(
            "release_date"
        )
        if len(sub) < 2:
            continue
        log_h = np.log(sub["horizon_50"].values)
        log_t = np.log(sub["tokens_at_fixed_difficulty"].values)
        slope, _ = np.polyfit(log_h, log_t, 1)
        logger.info("%-32s n=%d  log-log slope = %+.3f", lineage["label"], len(sub), slope)

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
        / "reports/time-horizon-1-1/analysis/output/tokens_vs_horizon.png",
    )
    parser.add_argument(
        "--out-table", type=pathlib.Path,
        default=REPO_ROOT
        / "reports/time-horizon-1-1/analysis/output/tokens_vs_horizon.csv",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
    main(args.runs_file, args.release_dates, args.out_plot, args.out_table)
