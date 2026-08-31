"""
Command to run:
    python -m src.basepipeline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import build_dataset, make_splits, overlap_report
from src.evaluate import evaluate_all, plot_confusion_matrix, run_cv
from src.train import train_all


def plot_class_distribution(dataset) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    counts = dataset["genre"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="steelblue")
    ax.set_title("Class distribution")
    ax.set_ylabel("# tracks")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main() -> None:
    sns.set_context("notebook")

    dataset, feature_cols = build_dataset()
    print(f"dataset: {dataset.shape} | classes: {dataset['genre'].nunique()}")
    print("\nsplit sizes:")
    print(dataset["fma_split"].value_counts())
    print("\ngenres × split (counts):")
    print(pd.crosstab(dataset["genre"], dataset["fma_split"]))
    overlap_report(dataset)
    plot_class_distribution(dataset)

    splits = make_splits(dataset, feature_cols)
    print(
        f"Train: {splits.Xtrain.shape} | Val: {splits.Xval.shape} | Test: {splits.Xtest.shape}"
    )

    trained = train_all(splits, feature_cols)
    results, summary = evaluate_all(trained, splits)
    print("\nTest summary:")
    print(summary.to_string(index=False))

    best = summary.iloc[0]["name"]
    best_entry = trained[best]
    X_eval = (
        best_entry["selector"].transform(splits.Xtest)
        if best_entry["selector"] is not None
        else splits.Xtest
    )
    plot_confusion_matrix(
        best, best_entry["model"], X_eval, splits.ytest, splits.label_encoder
    )

    cv_summary = run_cv(splits)
    print("\nGrouped CV summary:")
    print(cv_summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip artist-grouped 5-fold CV (much faster; holdout only).",
    )
    args = parser.parse_args()
    main()
