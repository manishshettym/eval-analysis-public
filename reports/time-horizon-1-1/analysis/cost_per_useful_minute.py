"""Cost per successful human-minute (Tom's intended metric, done right).

For each model m on a fixed task distribution:

    C_m =  (sum_i  w_i * tokens_{m,i})
           ----------------------------------------------
           (sum_i  w_i * success_{m,i} * human_minutes_i)

Numerator: total weighted tokens the model spent (failed attempts count).
Denominator: total weighted successfully-completed human-minutes (failed
attempts give zero credit).

This is the "tokens per minute of human work successfully completed"
unit-economics number that Tom's plot was a noisy proxy for.

We plot C over time per lineage, then decompose its per-year change into
the numerator (token spend) and denominator (capability) contributions.
Tom's "two compounding curves" claim implies BOTH should help -- the
numerator should fall (real efficiency) AND the denominator should rise
(real capability).  If only the denominator is moving, the drop in C is
all capability.
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

WEIGHT_COL = "invsqrt_task_weight"


def per_agent_metric(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent, g in runs.groupby("alias"):
        w = g[WEIGHT_COL].values
        tok = g["tokens_count"].values
        succ = g["score_binarized"].values
        mins = g["human_minutes"].values
        numerator = float(np.sum(w * tok))
        denominator = float(np.sum(w * succ * mins))
        if denominator <= 0:
            cost_per_minute = float("nan")
        else:
            cost_per_minute = numerator / denominator
        rows.append(
            {
                "agent": str(agent),
                "weighted_tokens": numerator,
                "weighted_successful_minutes": denominator,
                "tokens_per_successful_minute": cost_per_minute,
                "n_runs": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def yearly_factor(values: pd.Series, dates: pd.Series) -> float:
    mask = values.notna() & dates.notna() & (values > 0)
    v, d = values[mask], dates[mask]
    if len(v) < 2:
        return float("nan")
    t = (d - d.min()).dt.days.values / 365.0
    slope, _ = np.polyfit(t, np.log(v.values), 1)
    return float(np.exp(slope))


def make_plot(per_agent: pd.DataFrame, out_path: pathlib.Path) -> None:
    plt.style.use(str(REPO_ROOT / "reports/time-horizon-1-1/matplotlibrc"))
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.6], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_box = fig.add_subplot(gs[0, 1])
    ax_box.axis("off")

    decomposition_lines: list[tuple[str, str]] = []
    for lineage in LINEAGES:
        sub = per_agent[per_agent["lineage"] == lineage["key"]].sort_values(
            "release_date"
        ).dropna(subset=["tokens_per_successful_minute"])
        if sub.empty:
            continue

        ax.plot(
            sub["release_date"],
            sub["tokens_per_successful_minute"],
            color=lineage["color"], marker=lineage["marker"],
            markersize=8, linewidth=2,
            linestyle=lineage.get("linestyle", "-"),
            label=lineage["label"],
        )
        for _, row in sub.iterrows():
            ax.annotate(
                SHORT_LABELS.get(row["agent"], row["agent"]),
                (row["release_date"], row["tokens_per_successful_minute"]),
                textcoords="offset points", xytext=(6, 5),
                fontsize=9, color=lineage["color"],
            )

        if len(sub) >= 2:
            c_factor = yearly_factor(sub["tokens_per_successful_minute"], sub["release_date"])
            num_factor = yearly_factor(sub["weighted_tokens"], sub["release_date"])
            den_factor = yearly_factor(sub["weighted_successful_minutes"], sub["release_date"])
            decomposition_lines.append(
                (
                    lineage["color"],
                    f"{lineage['label']}",
                    f"  C  drops {1 / c_factor:5.2f}x / yr",
                    f"  tokens (num):  {num_factor:5.2f}x / yr  "
                    f"{'(rising, anti-efficiency)' if num_factor > 1.05 else '(roughly flat)' if num_factor > 0.95 else '(falling, helps)'}",
                    f"  successful-mins (den):  {den_factor:5.2f}x / yr  (capability)",
                )
            )

    ax.set_yscale("log")
    ax.set_ylabel("Tokens per successfully completed human-minute (log)")
    ax.set_xlabel("Release date")
    ax.grid(True, which="major", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)

    fig.suptitle(
        "Cost per successful human-minute -- and where the drop comes from",
        x=0.06, y=0.985, ha="left", fontsize=15,
    )
    fig.text(
        0.06, 0.93,
        "Per-agent: weighted total tokens / weighted total successfully-completed human-minutes "
        "(invsqrt task weighting).\n"
        "Tom's \"two curves\" claim needs BOTH numerator falling (efficiency) AND denominator "
        "rising (capability).",
        fontsize=10.5, color="#444",
    )

    # Decomposition box rendered into the right panel.
    ax_box.text(
        0.0, 0.95,
        "Per-year decomposition\n(least-squares on log values)",
        transform=ax_box.transAxes, fontsize=11.5, color="#222",
        ha="left", va="top", weight="bold",
    )
    cursor = 0.84
    for color, label, *lines in decomposition_lines:
        ax_box.text(
            0.0, cursor, label,
            transform=ax_box.transAxes, fontsize=11, color=color,
            ha="left", va="top", weight="bold",
        )
        cursor -= 0.05
        for line in lines:
            ax_box.text(
                0.0, cursor, line,
                transform=ax_box.transAxes, fontsize=10, color=color,
                ha="left", va="top", family="monospace",
            )
            cursor -= 0.04
        cursor -= 0.025

    ax.legend(loc="lower left", frameon=False, fontsize=10.5)
    fig.text(
        0.99, 0.005,
        "Source: METR eval-analysis-public . runs.jsonl  (tokens used as cost proxy)",
        ha="right", fontsize=9, color="#888",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
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
    runs = runs[runs["tokens_count"].notna() & runs["score_binarized"].notna()
                & runs["human_minutes"].notna()]
    runs = runs.drop(
        columns=[c for c in ("equal_task_weight", "invsqrt_task_weight") if c in runs]
    )
    runs = add_task_weight_columns(runs)

    release_dates = yaml.safe_load(release_dates_file.read_text())["date"]
    metrics = per_agent_metric(runs)

    rows = []
    for lineage in LINEAGES:
        for agent in lineage["agents"]:
            m = metrics[metrics["agent"] == agent]
            if m.empty:
                continue
            rows.append(
                {
                    "agent": agent,
                    "lineage": lineage["key"],
                    "lineage_label": lineage["label"],
                    "color": lineage["color"],
                    "marker": lineage["marker"],
                    "linestyle": lineage.get("linestyle", "-"),
                    "tokens_per_successful_minute": float(m["tokens_per_successful_minute"].iloc[0]),
                    "weighted_tokens": float(m["weighted_tokens"].iloc[0]),
                    "weighted_successful_minutes": float(m["weighted_successful_minutes"].iloc[0]),
                    "n_runs": int(m["n_runs"].iloc[0]),
                    "release_date": pd.Timestamp(release_dates.get(agent)),
                }
            )
    per_agent = pd.DataFrame(rows)

    out_table.parent.mkdir(parents=True, exist_ok=True)
    per_agent.drop(columns=["color", "marker", "linestyle"]).sort_values(
        ["lineage", "release_date"]
    ).to_csv(out_table, index=False)
    logger.info("Wrote %s", out_table)

    for lineage in LINEAGES:
        sub = per_agent[per_agent["lineage"] == lineage["key"]].sort_values(
            "release_date"
        ).dropna(subset=["tokens_per_successful_minute"])
        if len(sub) < 2:
            continue
        c = yearly_factor(sub["tokens_per_successful_minute"], sub["release_date"])
        n = yearly_factor(sub["weighted_tokens"], sub["release_date"])
        d = yearly_factor(sub["weighted_successful_minutes"], sub["release_date"])
        logger.info(
            "%-32s  C %.2fx/yr (drops %.2fx)  num %.2fx/yr  den %.2fx/yr",
            lineage["label"], c, 1 / c, n, d,
        )

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
        / "reports/time-horizon-1-1/analysis/output/cost_per_useful_minute.png",
    )
    parser.add_argument(
        "--out-table", type=pathlib.Path,
        default=REPO_ROOT
        / "reports/time-horizon-1-1/analysis/output/cost_per_useful_minute.csv",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
    main(args.runs_file, args.release_dates, args.out_plot, args.out_table)
