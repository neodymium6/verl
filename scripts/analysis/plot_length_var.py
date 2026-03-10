#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_stats(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out: list[float] = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        denom = min(i + 1, window)
        out.append(acc / denom)
    return out


def plot_metric(ax, stats: dict, key: str, title: str, window: int):
    for exp_name, steps in stats.items():
        step_nums = sorted(int(s) for s in steps.keys())
        vals = [steps[str(s)][key] for s in step_nums]
        vals = moving_average(vals, window)
        ax.plot(step_nums, vals, label=exp_name)

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=1)


def main() -> int:
    stats_path = Path("scripts/analysis/output") / "length_var_stats.json"
    if not stats_path.exists():
        raise RuntimeError(f"missing stats file: {stats_path}")
    stats = load_stats(stats_path)
    window = 20
    fig, axes = plt.subplots(8, 1, figsize=(12, 30), sharex=False)
    plot_metric(axes[0], stats, "cv_all_outputs", "cv_all_outputs (MA20)", window)
    plot_metric(axes[1], stats, "mean_prompt_cv", "mean_prompt_cv (MA20)", window)
    plot_metric(axes[2], stats, "cv_prompt_mean", "cv_prompt_mean (MA20)", window)
    plot_metric(
        axes[3],
        stats,
        "mean_abs_log_r_token",
        "mean_abs_log_r_token (MA20) Prompt≈Token",
        window,
    )
    plot_metric(
        axes[4],
        stats,
        "mean_abs_log_r_sample",
        "mean_abs_log_r_sample (MA20) Prompt≈Sample",
        window,
    )
    plot_metric(
        axes[5],
        stats,
        "mean_abs_log_r_token_sample",
        "mean_abs_log_r_token_sample (MA20) Token≈Sample",
        window,
    )
    plot_metric(
        axes[6],
        stats,
        "rel_prompt_token_minus_sample",
        "rel_prompt_token_minus_sample (MA20) Prompt Token-Sample",
        window,
    )
    plot_metric(
        axes[7],
        stats,
        "rel_prompt_token_ratio",
        "rel_prompt_token_ratio (MA20) Prompt Token/(Token+Sample)",
        window,
    )
    fig.tight_layout()

    out_path = Path("scripts/analysis/output") / "length_var_plots.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved\t{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
