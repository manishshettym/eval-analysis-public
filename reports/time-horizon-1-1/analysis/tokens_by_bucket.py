"""Tokens-per-task at fixed difficulty, faceted by task-length bucket.

Companion to tokens_at_fixed_difficulty.py.  That script picks a single
bucket (15-60 min) and concludes "tokens per task at fixed difficulty
are flat".  Honest pushback: maybe newer models *are* more token-efficient
on the longer tasks they were built for, and the 15-60 min slice misses it.

So we facet across all buckets the data supports, and look at the per-year
multiplicative change at fixed difficulty for each lineage in each bucket.

We also compute a "success-conditional" version: median tokens on tasks
the agent actually passed.  This removes the confound of older models
quitting fast (low tokens) on tasks far beyond their horizon.
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

BUCKETS = [
    (1, 4, "1-4 min"),
    (4, 15, "4-15 min"),
    (15, 60, "15-60 min"),
    (60, 240, "1-4 hr"),
    (240, 1e9, "4 hr+"),
]
MIN_RUNS = 5  # don't trust a median computed on fewer than this many runs


def yearly_factor(values: pd.Series, dates: pd.Series) -> float:
    mask = values.notna() & dates.notna() & (values > 0)
    v, d = values[mask], dates[mask]
    if len(v) < 2:
        return float("nan")
    t = (d - d.min()).dt.days.values / 365.0
    slope, _ = np.polyfit(t, np.log(v.values), 1)
    return float(np.exp(slope))


def median_in_bucket(
    runs: pd.DataFrame, lo: float, hi: float, *, success_only: bool = False
) -> dict[str, tuple[float, int]]:
    sub = runs[(runs["human_minutes"] >= lo) & (runs["human_minutes"] < hi)]
    if success_only:
        sub = sub[sub["score_binarized"] >= 1.0]
    out: dict[str, tuple[float, int]] = {}
    for agent, g in sub.groupby("alias"):
        toks = g["tokens_count"].dropna()
        if len(toks) >= MIN_RUNS:
            out[str(agent)] = (float(np.median(toks)), int(len(toks)))
    return out


def per_agent_per_bucket(
    runs: pd.DataFrame,
    release_dates: dict[str, str],
    *,
    success_only: bool,
) -> pd.DataFrame:
    rows = []
    for lo, hi, label in BUCKETS:
        m = median_in_bucket(runs, lo, hi, success_only=success_only)
        for lineage in LINEAGES:
            for agent in lineage["agents"]:
                if agent not in m:
                    continue
                tokens, n = m[agent]
                rows.append(
                    {
                        "agent": agent,
                        "lineage": lineage["key"],
                        "lineage_label": lineage["label"],
                        "color": lineage["color"],
                        "marker": lineage["marker"],
                        "linestyle": lineage.get("linestyle", "-"),
                        "bucket": label,
                        "bucket_lo": lo,
                        "median_tokens": tokens,
                        "n_runs": n,
                        "release_date": pd.Timestamp(release_dates.get(agent)),
                    }
                )
    return pd.DataFrame(rows)


def make_faceted_plot(
    per_agent: pd.DataFrame, out_path: pathlib.Path, *, suffix: str
) -> None:
    plt.style.use(str(REPO_ROOT / "reports/time-horizon-1-1/matplotlibrc"))
    fig, axes = plt.subplots(1, len(BUCKETS), figsize=(20, 6), sharey=True)

    for ax, (lo, hi, label) in zip(axes, BUCKETS):
        sub_bucket = per_agent[per_agent["bucket"] == label]
        per_lineage_factor: list[tuple[str, str, float]] = []
        for lineage in LINEAGES:
            sub = sub_bucket[sub_bucket["lineage"] == lineage["key"]].sort_values(
                "release_date"
            )
            if sub.empty:
                continue
            ax.plot(
                sub["release_date"], sub["median_tokens"],
                color=lineage["color"], marker=lineage["marker"],
                markersize=7, linewidth=1, linestyle=":", alpha=0.55,
                label=lineage["label"],
            )
            if len(sub) >= 2:
                t = (sub["release_date"] - sub["release_date"].min()).dt.days.values / 365.0
                y = np.log(sub["median_tokens"].values)
                slope, intercept = np.polyfit(t, y, 1)
                ax.plot(
                    sub["release_date"], np.exp(intercept + slope * t),
                    color=lineage["color"], linewidth=2.5,
                    linestyle=lineage.get("linestyle", "-"),
                )
                per_lineage_factor.append(
                    (lineage["color"], lineage["label"], float(np.exp(slope)))
                )

        ax.set_yscale("log")
        ax.set_title(f"{label}  (n agents = {sub_bucket['agent'].nunique()})")
        ax.grid(True, which="major", alpha=0.25)
        ax.tick_params(axis="x", rotation=30)

        # In-panel summary of per-year factor.
        for i, (color, label_, f) in enumerate(per_lineage_factor):
            if 0.95 <= f <= 1.05:
                tag = "flat"
            elif f > 1:
                tag = f"x{f:.2f}/yr"
            else:
                tag = f"x{f:.2f}/yr"
            ax.text(
                0.02, 0.98 - i * 0.06, f"{label_}: {tag}",
                transform=ax.transAxes, fontsize=8.5, color=color, va="top",
            )

    axes[0].set_ylabel("Median tokens per task (log)")
    axes[-1].legend(loc="lower right", frameon=False, fontsize=9)

    title = (
        "Tokens per task at fixed difficulty: across all task-length buckets"
        if suffix == "all"
        else "Tokens per task at fixed difficulty (success-only): across all buckets"
    )
    fig.suptitle(title, x=0.06, y=0.99, ha="left", fontsize=15)
    sub_text = (
        "Median tokens-per-task within each human-baseline bucket vs release date.  "
        "Trend lines = least-squares on log-tokens.  Bucket panels with too few "
        f"agents (n_runs < {MIN_RUNS}) are dropped per agent."
    )
    if suffix == "success":
        sub_text = (
            "Median tokens on tasks the agent actually passed (binarized score = 1).  "
            "Removes the artifact where older models quit cheaply on tasks beyond their horizon."
        )
    fig.text(0.06, 0.945, sub_text, fontsize=10, color="#444")

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info("Wrote %s", out_path)


def write_summary_table(
    all_runs: pd.DataFrame, success_runs: pd.DataFrame, out_path: pathlib.Path
) -> None:
    rows = []
    for label_kind, df in (("all", all_runs), ("success_only", success_runs)):
        for bucket in [b[2] for b in BUCKETS]:
            sub_bucket = df[df["bucket"] == bucket]
            for lineage in LINEAGES:
                sub = sub_bucket[sub_bucket["lineage"] == lineage["key"]].sort_values(
                    "release_date"
                )
                if len(sub) < 2:
                    continue
                f = yearly_factor(sub["median_tokens"], sub["release_date"])
                rows.append(
                    {
                        "kind": label_kind,
                        "bucket": bucket,
                        "lineage": lineage["label"],
                        "n_agents": len(sub),
                        "yearly_token_factor": round(f, 3),
                        "first_agent": sub["agent"].iloc[0],
                        "last_agent": sub["agent"].iloc[-1],
                        "first_tokens": int(sub["median_tokens"].iloc[0]),
                        "last_tokens": int(sub["median_tokens"].iloc[-1]),
                    }
                )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)


def main(
    runs_file: pathlib.Path,
    release_dates_file: pathlib.Path,
    out_dir: pathlib.Path,
) -> None:
    runs = pd.read_json(runs_file, lines=True, convert_dates=False)
    runs = runs[runs["alias"].isin(all_lineage_agents())].copy()
    runs = runs[runs["tokens_count"].notna() & runs["score_binarized"].notna()]
    runs = runs.drop(
        columns=[c for c in ("equal_task_weight", "invsqrt_task_weight") if c in runs]
    )
    runs = add_task_weight_columns(runs)

    release_dates = yaml.safe_load(release_dates_file.read_text())["date"]

    all_table = per_agent_per_bucket(runs, release_dates, success_only=False)
    success_table = per_agent_per_bucket(runs, release_dates, success_only=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    make_faceted_plot(all_table, out_dir / "tokens_by_bucket_all.png", suffix="all")
    make_faceted_plot(
        success_table, out_dir / "tokens_by_bucket_success.png", suffix="success"
    )
    write_summary_table(all_table, success_table, out_dir / "tokens_by_bucket.csv")

    # Console summary.
    for kind, df in (("all runs", all_table), ("success only", success_table)):
        logger.info("--- %s ---", kind)
        for bucket in [b[2] for b in BUCKETS]:
            for lineage in LINEAGES:
                sub = df[(df["bucket"] == bucket) & (df["lineage"] == lineage["key"])]
                sub = sub.sort_values("release_date")
                if len(sub) < 2:
                    continue
                f = yearly_factor(sub["median_tokens"], sub["release_date"])
                logger.info(
                    "  %-7s %-32s n=%d  %.2fx/yr  (%d -> %d tokens)",
                    bucket, lineage["label"], len(sub), f,
                    int(sub["median_tokens"].iloc[0]),
                    int(sub["median_tokens"].iloc[-1]),
                )


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
        "--out-dir", type=pathlib.Path,
        default=REPO_ROOT / "reports/time-horizon-1-1/analysis/output",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
    main(args.runs_file, args.release_dates, args.out_dir)
