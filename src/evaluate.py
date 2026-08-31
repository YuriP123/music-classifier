"""Evaluation: holdout reports, confusion matrices, and grouped cross-validation."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CV_FOLDS, CV_SUMMARY_CSV, KBEST_K, OUTPUTS_DIR, RNG, TEST_SUMMARY_CSV
from src.data import Splits
from src.train import SELECTED_MODEL, build_models


def evaluate_model(name, model, X_eval, y_eval_enc, label_encoder, verbose=True):
    y_pred = model.predict(X_eval)
    acc = accuracy_score(y_eval_enc, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_eval_enc, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(y_eval_enc, y_pred, average="weighted", zero_division=0)

    report_kwargs = dict(
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    if verbose:
        print(
            f"\n=== {name} - accuracy: {acc:.4f} | macro F1: {macro_f1:.4f} "
            f"| weighted F1: {weighted_f1:.4f} ==="
        )
        print(classification_report(y_eval_enc, y_pred, **report_kwargs))

    per_class = classification_report(y_eval_enc, y_pred, output_dict=True, **report_kwargs)
    return {
        "name": name,
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "y_pred": y_pred,
        "per_class": per_class,
    }


def evaluate_all(trained: dict, splits: Splits):
    results = []
    for name, entry in trained.items():
        X_eval = (
            entry["selector"].transform(splits.Xtest)
            if entry["selector"] is not None
            else splits.Xtest
        )
        results.append(
            evaluate_model(name, entry["model"], X_eval, splits.ytest, splits.label_encoder)
        )

    summary = (
        pd.DataFrame(
            [
                {k: r[k] for k in ("name", "accuracy", "macro_f1", "weighted_f1")}
                for r in results
            ]
        )
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(TEST_SUMMARY_CSV, index=False)
    return results, summary


def plot_confusion_matrix(name, model, X_eval, y_eval_enc, label_encoder, normalize="true"):
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_eval_enc,
        model.predict(X_eval),
        labels=np.arange(len(label_encoder.classes_)),
        display_labels=label_encoder.classes_,
        normalize=normalize,
        xticks_rotation=45,
        values_format=".2f",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion matrix - {name}" + (" (row-normalized)" if normalize else ""))
    fig.tight_layout()
    path = OUTPUTS_DIR / f"confusion_{name}.png"
    fig.savefig(path, dpi=150)
    plt.show()
    return path


def _pipeline(estimator, with_select=False) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if with_select:
        steps.append(("select", SelectKBest(score_func=f_classif, k=KBEST_K)))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def run_cv(splits: Splits) -> pd.DataFrame:
    models = build_models()
    cv_models = [
        (name, _pipeline(clone(est), with_select=(name == SELECTED_MODEL)))
        for name, est in models.items()
    ]

    X_dev = splits.X[splits.dev_mask]
    y_dev = splits.y_enc[splits.dev_mask]
    groups_dev = splits.groups[splits.dev_mask]

    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RNG)
    rows = []
    for name, pipe in cv_models:
        scores = cross_val_score(
            pipe, X_dev, y_dev, groups=groups_dev, scoring="accuracy", cv=cv, n_jobs=-1
        )
        rows.append({
            "model": name,
            "cv_mean": scores.mean(),
            "cv_std": scores.std(),
            "cv_min": scores.min(),
            "cv_max": scores.max(),
            "cv_folds": ", ".join(f"{s:.4f}" for s in scores),
        })
        print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

    cv_summary = (
        pd.DataFrame(rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)
    )
    cv_summary.to_csv(CV_SUMMARY_CSV, index=False)
    return cv_summary
