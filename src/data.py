from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src import utils
from src.config import DATASET_CSV, IDS_PATH, METADATA_DIR, X_PATH
from src.features import feature_column_names, run_extraction

@dataclass
class Splits:
    X: np.ndarray
    y: np.ndarray
    y_enc: np.ndarray
    label_encoder: LabelEncoder
    scaler: StandardScaler
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    dev_mask: np.ndarray
    groups: np.ndarray
    Xtrain: np.ndarray
    Xval: np.ndarray
    Xtest: np.ndarray
    ytrain: np.ndarray
    yval: np.ndarray
    ytest: np.ndarray

def load_tracks() -> pd.DataFrame:
    return utils.load(METADATA_DIR / "tracks.csv")

def build_genre_labels(tracks: pd.DataFrame) -> pd.DataFrame:
    in_medium = tracks[("set", "subset")] <= "medium"
    has_genre = tracks[("track", "genre_top")].notna()
    subset_df = tracks[in_medium & has_genre]

    genre_labels = (
        subset_df[("track", "genre_top")]
        .astype(str)
        .rename("genre")
        .to_frame()
    )
    genre_labels.index.name = "track_id"
    return genre_labels

def load_or_extract_features(track_ids: np.ndarray):
    if X_PATH.exists() and IDS_PATH.exists():
        return np.load(X_PATH), np.load(IDS_PATH)
    return run_extraction(track_ids)

def make_splits(dataset: pd.DataFrame, feature_cols: list[str], subgenre=False) -> Splits:
    X = dataset[feature_cols].to_numpy()
    if subgenre:
        y = dataset["subgenre"].to_numpy()
    else:
        y = dataset["genre"].to_numpy()

    label_encoder = LabelEncoder().fit(y)
    y_enc = label_encoder.transform(y)
    print(X,y)

    train_mask = (dataset["fma_split"] == "training").to_numpy()
    val_mask = (dataset["fma_split"] == "validation").to_numpy()
    test_mask = (dataset["fma_split"] == "test").to_numpy()

    train_artists = set(dataset.loc[train_mask, "artist_id"])
    test_artists = set(dataset.loc[test_mask, "artist_id"])
    assert not (train_artists & test_artists)

    scaler = StandardScaler().fit(X[train_mask])
    return Splits(
        X=X,
        y=y,
        y_enc=y_enc,
        label_encoder=label_encoder,
        scaler=scaler,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        dev_mask=~test_mask,
        groups=dataset["artist_id"].to_numpy(),
        Xtrain=scaler.transform(X[train_mask]),
        Xval=scaler.transform(X[val_mask]),
        Xtest=scaler.transform(X[test_mask]),
        ytrain=y_enc[train_mask],
        yval=y_enc[val_mask],
        ytest=y_enc[test_mask],
    )


def overlap_report(dataset: pd.DataFrame, split_col: str = "fma_split") -> None:
    train_artists = set(dataset.loc[dataset[split_col] == "training", "artist_id"])
    val_artists = set(dataset.loc[dataset[split_col] == "validation", "artist_id"])
    test_artists = set(dataset.loc[dataset[split_col] == "test", "artist_id"])

    pairs = {
        "train ∩ val": train_artists & val_artists,
        "train ∩ test": train_artists & test_artists,
        "val ∩ test": val_artists & test_artists,
    }
    for name, shared in pairs.items():
        print(f"{name}: {len(shared)} artists")

    test_df = dataset[dataset[split_col] == "test"]
    leaked = test_df["artist_id"].isin(train_artists).sum()
    print(
        f"test tracks with artist also in train: {leaked} / {len(test_df)} "
        f"({leaked / len(test_df):.1%})"
    )

def pick_subgenre(genre_ids: int, allowed: set):
    matches = set(genre_ids) & allowed
    if len(matches) != 1:
        return None
    genre = pd.read_csv(METADATA_DIR/"genres.csv").set_index("genre_id")
    kept = [gid for gid in genre_ids if gid in allowed]
    if len(kept) == 0:
        return None
    return genre.loc[kept[0], "title"]

def build_subgenres(tracks: pd.DataFrame | None = None):
    genre = pd.read_csv(METADATA_DIR/"genres.csv").set_index("genre_id")
    in_med = tracks[("set", "subset")] <= "medium"
    has_genre = tracks[("track", "genre_top")].notna()
    subset_df = tracks[in_med & has_genre]

    rock_df = subset_df[subset_df[("track", "genre_top")] == "Rock"]
    rock_train = rock_df[rock_df[("set", "split")] == "training"]

    edm_df = subset_df[subset_df[("track", "genre_top")] == "Electronic"]
    edm_train = edm_df[edm_df[("set","split")] == "training"]

    rock_ids = rock_train[("track", "genres")].dropna().explode()
    rock_ids = rock_ids[rock_ids != 12]

    edm_ids = edm_train[("track", "genres")].dropna().explode()
    edm_ids = edm_ids[edm_ids != 15]
    rock_counts = rock_ids.value_counts()
    edm_counts = edm_ids.value_counts()
    filtered_rock = rock_counts[rock_counts >= 200]
    filtered_edm = edm_counts[edm_counts >= 200]
    rock_branch = set(
        genre.index[
            (genre["top_level"] == 12) & (genre.index != 12)
        ]
    )
    electronic_branch = set(
        genre.index[
            (genre["top_level"] == 15) & (genre.index != 15)
        ]
    )
    allowed_rock = set(filtered_rock.index) & rock_branch
    allowed_edm = set(filtered_edm.index) & electronic_branch
    rock_df = rock_df.copy()
    edm_df = edm_df.copy()
    rock_df["subgenre"] = rock_df[("track","genres")].map(
            lambda ids: pick_subgenre(ids, allowed_rock)
    )
    edm_df["subgenre"] = edm_df[("track","genres")].map(
            lambda ids: pick_subgenre(ids, allowed_edm)
    )
    
    print(rock_df["subgenre"].isna().mean())
    print(rock_df.loc[rock_df[("set","split")] == "training" , "subgenre"].value_counts())

    print(edm_df["subgenre"].isna().mean())
    print(edm_df.loc[edm_df[("set","split")] == "training" , "subgenre"].value_counts())

    return rock_df, edm_df

def build_dataset(tracks: pd.DataFrame | None = None, subgenre=False):
    if tracks is None:
        tracks = load_tracks()

    genre_labels = build_genre_labels(tracks)
    subset_ids = np.asarray(genre_labels.index, dtype=np.int32)
    X_raw, track_ids = load_or_extract_features(subset_ids)

    feature_cols = feature_column_names()
    assert len(feature_cols) == X_raw.shape[1], (len(feature_cols), X_raw.shape[1])

    features_df = pd.DataFrame(X_raw, columns=feature_cols)
    features_df["track_id"] = track_ids

    dataset = features_df.merge(genre_labels.reset_index(), on="track_id", how="inner")

    artist_split = pd.DataFrame({
        "track_id": tracks.index.astype(int),
        "artist_id": tracks[("artist", "id")].to_numpy(),
        "fma_split": tracks[("set", "split")].astype(str).to_numpy(),
    })

    dataset = dataset.merge(artist_split, on="track_id", how="left")
    if subgenre == True:
        print("----------SUBGENRE TARGETED------------")
        subRockdf, subEdmdf= build_subgenres(tracks)
        rock_labels = (
            subRockdf["subgenre"]
            .rename_axis("track_id")
            .reset_index()
            .dropna(subset="subgenre")
        )
        edm_labels = (
                subEdmdf["subgenre"]
                .rename_axis("track_id")
                .reset_index()
                .dropna(subset="subgenre")
        )
        labels = pd.concat([rock_labels,edm_labels], ignore_index=True)
        subgenreDataset = dataset.merge(labels,on="track_id", how="left")
        return subgenreDataset, feature_cols

    dataset.to_csv(DATASET_CSV, index=False)
    return dataset, feature_cols
