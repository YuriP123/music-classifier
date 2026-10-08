# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
from src.data import *


tracks = load_tracks()
genre = pd.read_csv(METADATA_DIR/"genres.csv").set_index("genre_id")
rock = genre.loc[ (genre["parent"] != 0) & (genre["top_level"] == 12)]
electronic = genre.loc[ (genre["parent"] != 0) & (genre["top_level"] == 15)]

def pick_subgenre(genre_ids, allowed):
    kept = [gid for gid in genre_ids if gid in allowed]
    if len(kept) == 0:
        return None
    return genre.loc[kept[0], "title"]

def build_subgenres(tracks: pd.DataFrame):
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
    allowed_rock = set(filtered_rock.index)
    allowed_edm = set(filtered_edm.index)
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





    # print(row[("track", "genre_top")])
    # for gid in genreidx:
    #     print(genre.loc[gid, "title"], genre.loc[gid, "parent"])

build_subgenres(tracks)



