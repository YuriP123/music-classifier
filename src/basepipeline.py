from src.config import OUTPUTS_DIR, MODELS_DIR, RNG
from src.data import build_dataset, make_splits, overlap_report, build_subgenres
from src.evaluate import evaluate_all, evaluate_model, plot_confusion_matrix
from src.train import train_all
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
sns.set_context("notebook")

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

def print_genreinfo(dataset: pd.DataFrame):
    print(f"dataset: {dataset.shape} | classes: {dataset['genre'].nunique()}")
    print("\nsplit sizes:")
    print(dataset["fma_split"].value_counts())
    print("\ngenres × split (counts):")
    print(pd.crosstab(dataset["genre"], dataset["fma_split"]))


def main() -> None:
    sns.set_context("notebook")
    dataset, feature_cols = build_dataset(subgenre=True)
    rockdf = dataset[(dataset["genre"] == "Rock") & dataset["subgenre"].notna()].copy()
    edmdf = dataset[(dataset["genre"] == "Electronic") & dataset["subgenre"].notna()].copy()

    for name, frame in [("Rock", rockdf), ("Electronic", edmdf)]:
        training_rows = frame["fma_split"] == "training"
        counts = frame.loc[training_rows, "subgenre"].value_counts()
        print(f"\n{name}: final training counts")
        print(counts)

    for name, frame in [("Rock", rockdf), ("Electronic", edmdf)]:
      counts = pd.crosstab(frame["subgenre"], frame["fma_split"])
      print(f"\n{name}: subgenre counts by split")
      print(counts)

    print_genreinfo(rockdf)
    print_genreinfo(edmdf)
    rocksplits = make_splits(rockdf, feature_cols, subgenre=True)
    edmsplits = make_splits(edmdf, feature_cols, subgenre=True)

    print(
        f"Train: {rocksplits.Xtrain.shape} | Val: {rocksplits.Xval.shape} | Test: {rocksplits.Xtest.shape}"
    )
    print(
        f"Train: {edmsplits.Xtrain.shape} | Val: {edmsplits.Xval.shape} | Test: {edmsplits.Xtest.shape}"
    )

    for genre, splits in [("rock", rocksplits), ("edm", edmsplits)]:
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(splits.Xtrain, splits.ytrain)

        evaluate_model(
            name="majority_baseline_validation",
            genre=genre,
            model=baseline,
            X_eval=splits.Xval,
            y_eval_enc=splits.yval,
            label_encoder=splits.label_encoder,
    )

    rocktrained = train_all(rocksplits, feature_cols, genre="rock")
    edmtrained = train_all(edmsplits, feature_cols, genre="edm")

    for genre, trained, splits in [
        ("rock", rocktrained, rocksplits),
        ("edm", edmtrained, edmsplits),
    ]:
        evaluate_model(
                name="M7_lgbm_validation",
                genre=genre,
                model=trained["M7_lgbm"]["model"],
                X_eval=splits.Xval,
                y_eval_enc=splits.yval,
                label_encoder=splits.label_encoder,
        )

    rock_results, rock_summary = evaluate_all(rocktrained, rocksplits, genre="rock")
    edm_results, edm_summary = evaluate_all(edmtrained, edmsplits, genre="edm")

    print(rock_summary)
    print(edm_summary)

    print("\nTest summary:")
    print(rock_summary.to_string(index=False))
    print(edm_summary.to_string(index=False))

    bestrock = rock_summary.iloc[0]["name"]
    bestedm = edm_summary.iloc[0]["name"]
    best_Rockentry = rocktrained[bestrock]
    best_Edmentry = edmtrained[bestedm]

    X_Rockeval = (
        best_Rockentry["selector"].transform(rocksplits.Xtest)
        if best_Rockentry["selector"] is not None
        else rocksplits.Xtest
    )

    X_Edmeval = (
        best_Edmentry["selector"].transform(edmsplits.Xtest)
        if best_Edmentry["selector"] is not None
        else edmsplits.Xtest
    )

    plot_confusion_matrix(
        f"{bestrock}_rock", best_Rockentry["model"],
        X_Rockeval, rocksplits.ytest, rocksplits.label_encoder,
    )

    plot_confusion_matrix(
        f"{bestedm}_edm", best_Edmentry["model"],
        X_Edmeval, edmsplits.ytest, edmsplits.label_encoder,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip artist-grouped 5-fold CV (much faster; holdout only).",
    )
    args = parser.parse_args()
    main()

