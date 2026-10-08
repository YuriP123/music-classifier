"""Model definitions, training, and artifact persistence."""
import pickle
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.config import KBEST_K, KNN_K, MODELS_DIR, RNG, SELECTED_FEATURES_CSV
from src.data import Splits

# The one model that trains on SelectKBest-reduced features.
SELECTED_MODEL = "M2_knn_top10"


def build_models() -> dict:
    return {
        "M1_knn": KNeighborsClassifier(n_neighbors=KNN_K),
        SELECTED_MODEL: KNeighborsClassifier(n_neighbors=KNN_K),
        "M3_logreg": LogisticRegression(
            max_iter=2000, random_state=RNG, class_weight="balanced", n_jobs=-1
        ),
        "M4_rf": RandomForestClassifier(
            n_estimators=300, random_state=RNG, class_weight="balanced", n_jobs=-1
        ),
        "M5_mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=RNG,
        ),
        "M6_xgb": XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            tree_method="hist",
            n_jobs=-1,
            random_state=RNG,
        ),
        "M7_lgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=RNG,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def fit_selector(splits: Splits, feature_cols: list[str], genre: str) -> SelectKBest:
    selector = SelectKBest(score_func=f_classif, k=KBEST_K).fit(splits.Xtrain, splits.ytrain)
    report = (
        pd.DataFrame({
            "feature": feature_cols,
            "f_score": selector.scores_,
            "p_value": selector.pvalues_,
            "selected": selector.get_support(),
        })
        .sort_values("f_score", ascending=False)
        .reset_index(drop=True)
    )
    report.to_csv(
         SELECTED_FEATURES_CSV.with_name(
             f"selected_features_top10_{genre}.csv"
         ),
         index=False,
    )
    return selector


def save_artifact(name: str, model, splits: Splits, genre: str, feature_cols: list[str], selector=None):
    payload = {
        "model": model,
        "scaler": splits.scaler,
        "label_encoder": splits.label_encoder,
        "features": feature_cols,
    }
    if selector is not None:
        payload["selector"] = selector
        payload["features"] = [
            c for c, keep in zip(feature_cols, selector.get_support()) if keep
        ]
    path = MODELS_DIR / f"{name}_{genre}.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def train_all(splits: Splits, feature_cols: list[str], genre: str) -> dict:
    selector = fit_selector(splits, feature_cols, genre)
    trained = {}
    for name, model in build_models().items():
        if name == "M7_lgbm":
            uses_selector = name == SELECTED_MODEL
            Xfit = selector.transform(splits.Xtrain) if uses_selector else splits.Xtrain
            print(f"Training {name}_{genre}...")
            model.fit(Xfit, splits.ytrain)
            save_artifact(name, model, splits, genre, feature_cols, selector if uses_selector else None)
            trained[name] = {"model": model, "selector": selector if uses_selector else None}
        else:
            pass
    return trained
