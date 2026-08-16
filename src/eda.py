"""Exploratory data analysis for the diabetes readmission features.

Model-centric EDA over the 16 features the baseline/AutoML models actually use plus the
`readmitted_binary` target. Produces three deliverables under eda/:

  * eda/figures/*.png       - seaborn/matplotlib charts (target balance, distributions,
                              correlations, readmission rate by key features)
  * eda/EDA_SUMMARY.md      - written summary: shape, target balance, missingness, numeric
                              stats, categorical levels, top correlations with the target
  * eda/eda_report.html     - Evidently DataQualityPreset report (same classic look as the
                              monitoring dashboards) for interactive per-column exploration

Run:  venv/bin/python src/eda.py    (from the repo root)
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render straight to PNG, no display needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import load_features
from train_baseline import FEATURE_COLS, CATEGORICAL_COLS, TARGET_COL

from evidently.legacy.metric_preset import DataQualityPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

EDA_DIR = "eda"
FIG_DIR = "eda/figures"
SUMMARY_MD = "eda/EDA_SUMMARY.md"
REPORT_HTML = "eda/eda_report.html"

NUMERICAL_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]
AGE_LABELS = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
              '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']

sns.set_theme(style="whitegrid")


def _savefig(fig, name):
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_balance(df):
    counts = df[TARGET_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=["No readmission (0)", "Readmitted (1)"], y=counts.values, ax=ax,
                palette=["#4C9F70", "#D9534F"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n{v/len(df):.1%}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Target balance: readmitted_binary")
    ax.set_ylabel("Encounters")
    ax.margins(y=0.15)
    return _savefig(fig, "target_balance.png")


def plot_numeric_distributions(df):
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))
    for ax, col in zip(axes.ravel(), NUMERICAL_COLS[:12]):
        sns.histplot(df[col], bins=30, ax=ax, color="#3B7DD8")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    fig.suptitle("Numeric feature distributions", fontsize=13, y=1.01)
    fig.tight_layout()
    return _savefig(fig, "numeric_distributions.png")


def plot_categorical_distributions(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, CATEGORICAL_COLS):
        order = df[col].value_counts().index[:10]
        sns.countplot(y=col, data=df, order=order, ax=ax, color="#7E63B8")
        ax.set_title(col, fontsize=11)
        ax.set_ylabel("")
    fig.suptitle("Categorical feature distributions (top levels)", fontsize=13, y=1.03)
    fig.tight_layout()
    return _savefig(fig, "categorical_distributions.png")


def plot_correlation_heatmap(df):
    corr = df[NUMERICAL_COLS + [TARGET_COL]].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation matrix (numeric features + target)")
    return _savefig(fig, "correlation_heatmap.png")


def plot_readmission_rate_by_feature(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # by age band
    by_age = df.groupby("age_ordinal")[TARGET_COL].mean().reindex(range(10))
    sns.barplot(x=[AGE_LABELS[i] for i in by_age.index], y=by_age.values, ax=axes[0], color="#D9534F")
    axes[0].set_title("Readmission rate by age band"); axes[0].set_ylabel("readmit rate")
    axes[0].tick_params(axis="x", rotation=60)
    # by primary diagnosis category
    by_diag = df.groupby("diag_1_category")[TARGET_COL].mean().sort_values(ascending=False)
    sns.barplot(x=by_diag.values, y=by_diag.index, ax=axes[1], color="#D9534F")
    axes[1].set_title("Readmission rate by diagnosis category"); axes[1].set_xlabel("readmit rate")
    # by number of prior inpatient visits (capped)
    tmp = df.copy()
    tmp["inpatient_cap"] = tmp["number_inpatient"].clip(upper=5)
    by_inp = tmp.groupby("inpatient_cap")[TARGET_COL].mean()
    sns.barplot(x=by_inp.index.astype(int), y=by_inp.values, ax=axes[2], color="#D9534F")
    axes[2].set_title("Readmission rate by prior inpatient visits (0-5+)")
    axes[2].set_xlabel("number_inpatient (capped at 5)"); axes[2].set_ylabel("readmit rate")
    fig.suptitle("Readmission rate vs. key features", fontsize=13, y=1.03)
    fig.tight_layout()
    return _savefig(fig, "readmission_rate_by_feature.png")


def write_summary(df, figures):
    pos = df[TARGET_COL].mean()
    miss = (df[FEATURE_COLS].isna().mean() * 100).round(2)
    miss = miss[miss > 0].sort_values(ascending=False)
    corr_t = df[NUMERICAL_COLS + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    top_corr = corr_t.reindex(corr_t.abs().sort_values(ascending=False).index).head(8)

    lines = [
        "# Exploratory Data Analysis — Diabetes 30-Day Readmission",
        "",
        f"Dataset: **{df.shape[0]:,} encounters × {len(FEATURE_COLS)} model features** "
        f"(+ `{TARGET_COL}` target). Source: engineered feature set "
        "(`data/processed/diabetic_data_features.csv`).",
        "",
        "## Target balance",
        f"- Readmitted (`{TARGET_COL}=1`): **{pos:.1%}**  ·  Not readmitted: **{1-pos:.1%}**",
        "- Mild class imbalance — worth noting for threshold/metric choices "
        "(the models report F1 alongside accuracy for this reason).",
        "",
        "## Missingness (model features)",
    ]
    if len(miss):
        lines.append("| Feature | % missing |")
        lines.append("|---|---|")
        lines += [f"| `{c}` | {v}% |" for c, v in miss.items()]
    else:
        lines.append("- No missing values in the 16 model features (cleaning upstream fills/encodes them).")

    lines += [
        "",
        "## Strongest linear correlations with the target",
        "| Feature | corr with readmit |",
        "|---|---|",
    ]
    lines += [f"| `{c}` | {v:+.3f} |" for c, v in top_corr.items()]

    lines += [
        "",
        "> Correlations are weak and roughly linear — consistent with the models' modest AUC "
        "(~0.65): readmission is driven by many small signals rather than one dominant feature. "
        "`number_inpatient` (prior admissions) is the most informative single feature.",
        "",
        "## Numeric feature summary",
        "",
        df[NUMERICAL_COLS].describe().round(2).T.to_markdown(),
        "",
        "## Figures",
    ]
    lines += [f"- `{os.path.relpath(f, EDA_DIR)}`" for f in figures]
    lines += [
        "",
        "See `eda/eda_report.html` for the full interactive Evidently data-quality report "
        "(per-column distributions, quantiles, missing values, correlations).",
    ]

    Path(EDA_DIR).mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    return SUMMARY_MD


def evidently_quality_report(df):
    cm = ColumnMapping()
    cm.numerical_features = list(NUMERICAL_COLS)
    cm.categorical_features = list(CATEGORICAL_COLS)
    cm.target = TARGET_COL
    report = Report(metrics=[DataQualityPreset()])
    report.run(reference_data=None, current_data=df[FEATURE_COLS + [TARGET_COL]], column_mapping=cm)
    Path(EDA_DIR).mkdir(parents=True, exist_ok=True)
    report.save_html(REPORT_HTML)
    return REPORT_HTML


def main():
    df = load_features()
    print(f"[eda] loaded {df.shape[0]:,} rows")

    figures = [
        plot_target_balance(df),
        plot_numeric_distributions(df),
        plot_categorical_distributions(df),
        plot_correlation_heatmap(df),
        plot_readmission_rate_by_feature(df),
    ]
    for f in figures:
        print(f"[eda] figure -> {f}")

    print(f"[eda] summary -> {write_summary(df, figures)}")
    print(f"[eda] evidently data-quality report -> {evidently_quality_report(df)}")
    print("[eda] done")


if __name__ == "__main__":
    main()
