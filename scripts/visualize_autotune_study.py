#!/usr/bin/env python3
"""
Plot autotune study results.

Requires:
  evaluation_results/autotune_study/sweep_records.csv
  evaluation_results/autotune_study/selection_summary.csv

Usage:
  .venv/bin/python scripts/visualize_autotune_study.py
  .venv/bin/python scripts/visualize_autotune_study.py --exemplar --model hybrid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autotune_study_common import (
    CONFIG_JSON,
    FIGURES_DIR,
    MODEL_TYPES,
    SELECTION_CSV,
    STUDY_DIR,
    SWEEP_CSV,
    load_study_csv,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def _ensure_model_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "model_type" not in out.columns:
        out["model_type"] = "t-frex"
    return out


def _filter_model(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    return _ensure_model_type(df)[_ensure_model_type(df)["model_type"] == model_type]


def _ordered_sample_sizes(values) -> list:
    """Ascending numeric sizes, with 0 (= all reviews) last."""
    return sorted(values, key=lambda s: (s == 0, s))


def _sample_size_label(sample_size: int) -> str:
    return "all" if sample_size == 0 else str(sample_size)


MODEL_PLOT_ORDER = ["hybrid", "t-frex", "transfeatex"]
MODEL_COLORS = dict(zip(MODEL_PLOT_ORDER, sns.color_palette("tab10", len(MODEL_PLOT_ORDER))))


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


def fig_sample_size_stability(selection: pd.DataFrame) -> None:
    """Selected metrics vs reviews per app (mean ± std over apps)."""
    selection = _ensure_model_type(selection)
    ok = selection[selection["selected_balanced_score"].notna()].copy()
    if ok.empty:
        return

    model_types = [m for m in MODEL_PLOT_ORDER if m in ok["model_type"].unique()]
    sample_sizes = _ordered_sample_sizes(ok["sample_size"].unique())
    x_labels = [_sample_size_label(s) for s in sample_sizes]
    xs = np.arange(len(sample_sizes))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Stability across review sample size (mean ± std over apps)", fontsize=12)

    metric_specs = [
        (axes[0, 0], "selected_silhouette", "Selected silhouette"),
        (axes[0, 1], "selected_davies_bouldin", "Davies--Bouldin index"),
        (axes[1, 0], "selected_n_clusters", "Selected # clusters"),
        (axes[1, 1], "selected_threshold", "Selected threshold"),
    ]
    for ax, col, title in metric_specs:
        for mt in model_types:
            sub = ok[ok["model_type"] == mt]
            means, stds = [], []
            for ss in sample_sizes:
                vals = sub.loc[sub["sample_size"] == ss, col].dropna()
                means.append(vals.mean() if len(vals) else np.nan)
                stds.append(vals.std() if len(vals) > 1 else 0.0)
            ax.errorbar(
                xs, means, yerr=stds, fmt="o-", capsize=4, linewidth=2,
                color=MODEL_COLORS[mt], label=mt, markersize=6,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_xlabel("Reviews per app")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "sample_size_stability")


def fig_baseline_comparison(selection: pd.DataFrame) -> None:
    """Selected configuration vs fixed/random τ baselines."""
    selection = _ensure_model_type(selection)
    ok = selection.dropna(subset=["selected_balanced_score"])
    if ok.empty:
        return

    methods = {
        "Selected\n(balanced)": "selected_balanced_score",
        "Median τ": "baseline_median_threshold_balanced_score",
        "Max silhouette": "baseline_max_silhouette_balanced_score",
        "Fixed τ=0.5": "baseline_fixed_0.5_balanced_score",
        "Random τ": "baseline_random_mean_balanced_score",
    }

    model_types = sorted(ok["model_type"].unique())
    method_labels = list(methods.keys())
    n_methods = len(method_labels)
    n_models = len(model_types)
    width = 0.8 / max(n_models, 1)
    x = np.arange(n_methods)

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("muted", n_models)

    for i, mt in enumerate(model_types):
        sub = ok[ok["model_type"] == mt]
        means, stds, wins = [], [], []
        for col in methods.values():
            vals = sub[col].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std() if len(vals) > 1 else 0.0)
            if col == "selected_balanced_score":
                wins.append(np.nan)
            else:
                beat_col = col.replace("_balanced_score", "")
                beat_col = "selected_beats_baseline_" + beat_col.replace("baseline_", "")
                wins.append(sub[beat_col].mean() * 100 if beat_col in sub.columns else np.nan)

        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width=width, yerr=stds, capsize=3,
            color=palette[i], edgecolor="white", label=mt,
        )
        for bar, w in zip(bars, wins):
            if not np.isnan(w):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{w:.0f}%", ha="center", va="bottom", fontsize=7, color="#333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.set_ylabel("Balanced score (mean ± std over all app × sample runs)")
    ax.set_title("Auto-selected configuration vs baselines")
    ax.legend(title="Extractor")
    fig.tight_layout()
    _save(fig, "baseline_comparison")


def _pick_exemplar_app(sweep: pd.DataFrame, selection: pd.DataFrame, model_type: str) -> str:
    selection = _filter_model(selection, model_type)
    full = selection[selection["sample_size"] == 0]
    if full.empty:
        full = selection
    if full.empty:
        sweep = _filter_model(sweep, model_type)
        return sweep["app_name"].iloc[0]
    return full.loc[full["n_unique_features"].idxmax(), "app_name"]


def fig_threshold_landscape(
    sweep: pd.DataFrame, selection: pd.DataFrame, exemplar_app: str, model_type: str,
) -> None:
    """Metrics vs height_threshold for one exemplar app."""
    sweep = _filter_model(sweep, model_type)
    selection = _filter_model(selection, model_type)
    sub = sweep[(sweep["app_name"] == exemplar_app) & (sweep["sample_size"] == 0)].copy()
    if sub.empty:
        sub = sweep[sweep["app_name"] == exemplar_app].groupby("sample_size").max().reset_index()
        sub = sweep[(sweep["app_name"] == exemplar_app) & (sweep["sample_size"] == sub["sample_size"].max())]

    sel = selection[(selection["app_name"] == exemplar_app) & (selection["sample_size"] == sub["sample_size"].iloc[0])]
    sel_thr = sel["selected_threshold"].iloc[0] if len(sel) else None

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    fig.suptitle(
        f"Threshold sweep — {exemplar_app} [{model_type}]\n"
        f"(sample={sub['sample_size'].iloc[0]} reviews/app)",
        fontsize=12,
    )

    x = sub["height_threshold"]
    panels = [
        (axes[0, 0], sub["silhouette_score"], "Silhouette", "#2a6fbb"),
        (axes[0, 1], sub["davies_bouldin_score"], "Davies–Bouldin", "#c44e52"),
        (axes[1, 0], sub["n_clusters"], "# clusters", "#55a868"),
        (axes[1, 1], sub["singleton_ratio"], "Singleton ratio", "#8172b3"),
    ]
    for ax, y, label, color in panels:
        ax.plot(x, y, "o-", color=color, linewidth=2, markersize=6)
        ax.set_ylabel(label)
        if sel_thr is not None:
            ax.axvline(sel_thr, color="black", linestyle="--", linewidth=1.2, label="selected")
        ax.legend(loc="best", fontsize=8)

    axes[1, 0].set_xlabel("Height threshold (linkage distance)")
    axes[1, 1].set_xlabel("Height threshold (linkage distance)")
    fig.tight_layout()
    _save(fig, f"threshold_landscape_{model_type}")


def fig_singleton_vs_silhouette(
    sweep: pd.DataFrame, selection: pd.DataFrame, exemplar_app: str, model_type: str,
) -> None:
    """Silhouette vs singleton ratio for one exemplar app."""
    sweep = _filter_model(sweep, model_type)
    selection = _filter_model(selection, model_type)
    sub = sweep[(sweep["app_name"] == exemplar_app) & (sweep["sample_size"] == 0)]
    if sub.empty:
        sample_size = sweep[sweep["app_name"] == exemplar_app]["sample_size"].max()
        sub = sweep[(sweep["app_name"] == exemplar_app) & (sweep["sample_size"] == sample_size)]

    sel_row = selection[(selection["app_name"] == exemplar_app) & (selection["sample_size"] == sub["sample_size"].iloc[0])]
    if sel_row.empty:
        return

    sel_thr = sel_row["selected_threshold"].iloc[0]
    max_sil_thr = sel_row.get("baseline_max_silhouette_threshold", pd.Series([np.nan])).iloc[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        sub["singleton_ratio"], sub["silhouette_score"],
        c=sub["height_threshold"], cmap="viridis", s=80, edgecolors="white", linewidth=0.5,
    )
    plt.colorbar(sc, ax=ax, label="height threshold")

    sel_pt = sub[np.isclose(sub["height_threshold"], sel_thr)]
    if not sel_pt.empty:
        ax.scatter(sel_pt["singleton_ratio"], sel_pt["silhouette_score"],
                   s=200, facecolors="none", edgecolors="red", linewidths=2.5, label="selected (balanced)")

    if pd.notna(max_sil_thr):
        ms_pt = sub[np.isclose(sub["height_threshold"], max_sil_thr)]
        if not ms_pt.empty:
            ax.scatter(ms_pt["singleton_ratio"], ms_pt["silhouette_score"],
                       s=200, facecolors="none", edgecolors="orange", linewidths=2, linestyle="--",
                       label="max silhouette")

    ax.set_xlabel("Singleton ratio")
    ax.set_ylabel("Silhouette score")
    ax.set_title(f"Silhouette vs singleton trade-off\n{exemplar_app} [{model_type}]")
    ax.legend()
    fig.tight_layout()
    _save(fig, f"silhouette_singleton_tradeoff_{model_type}")


def fig_balanced_components(sweep: pd.DataFrame, exemplar_app: str, model_type: str) -> None:
    """Balanced score breakdown for top-3 autotune candidates."""
    sweep = _filter_model(sweep, model_type)
    sub = sweep[
        (sweep["app_name"] == exemplar_app)
        & (sweep["sample_size"] == 0)
        & (sweep["autotune_rank"].notna())
        & (sweep["autotune_rank"] <= 3)
        & (sweep["balanced_score"].notna())
    ].sort_values("autotune_rank")

    if sub.empty:
        sample_size = sweep[sweep["app_name"] == exemplar_app]["sample_size"].max()
        sub = sweep[
            (sweep["app_name"] == exemplar_app)
            & (sweep["sample_size"] == sample_size)
            & (sweep["autotune_rank"].notna())
            & (sweep["autotune_rank"] <= 3)
            & (sweep["balanced_score"].notna())
        ].sort_values("autotune_rank")

    if sub.empty:
        return

    components = ["sil_component", "db_component", "cluster_penalty_component", "size_penalty_component"]
    labels = ["Silhouette (40%)", "DB inv (30%)", "Cluster # (15%)", "Avg size (15%)"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sub))
    bottom = np.zeros(len(sub))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]

    for comp, label, color in zip(components, labels, colors):
        vals = sub[comp].values
        ax.bar(x, vals, bottom=bottom, label=label, color=color, width=0.6)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Rank {int(r)}\nτ={t:.2f}" for r, t in zip(sub["autotune_rank"], sub["height_threshold"])],
        fontsize=9,
    )
    ax.set_ylabel("Balanced score (sum of components)")
    ax.set_title(
        f"Balanced selection among top-3 autotune candidates\n{exemplar_app} [{model_type}]"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, f"balanced_components_{model_type}")


def main():
    parser = argparse.ArgumentParser(description="Plot autotune study results.")
    parser.add_argument(
        "--exemplar",
        action="store_true",
        help="Also plot per-app diagnostic figures (threshold sweep, trade-off, balanced components)",
    )
    parser.add_argument(
        "--model",
        default="hybrid",
        choices=MODEL_TYPES,
        help="Extractor for --exemplar plots (default: hybrid)",
    )
    args = parser.parse_args()

    if not SWEEP_CSV.exists() or not SELECTION_CSV.exists():
        print(f"Missing data in {STUDY_DIR}. Run scripts/run_autotune_study.py first.")
        sys.exit(1)

    sweep = _ensure_model_type(load_study_csv(SWEEP_CSV))
    selection = _ensure_model_type(load_study_csv(SELECTION_CSV))

    print(f"Generating figures in {FIGURES_DIR}...")
    fig_sample_size_stability(selection)
    fig_baseline_comparison(selection)

    if args.exemplar:
        models_in_data = sorted(selection["model_type"].unique())
        exemplar_model = args.model if args.model in models_in_data else models_in_data[0]
        exemplar = _pick_exemplar_app(sweep, selection, exemplar_model)
        print(f"  exemplar: {exemplar} [{exemplar_model}]")
        fig_threshold_landscape(sweep, selection, exemplar, exemplar_model)
        fig_singleton_vs_silhouette(sweep, selection, exemplar, exemplar_model)
        fig_balanced_components(sweep, exemplar, exemplar_model)

    if CONFIG_JSON.exists():
        print(f"Config: {CONFIG_JSON}")
    print("Done.")


if __name__ == "__main__":
    main()
