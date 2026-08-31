"""Audio feature extraction: 84 summary statistics per 30-second clip."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src import utils
from src.config import (
    AUDIO_DIR,
    BAD_FILES_CSV,
    DURATION,
    HOP_LENGTH,
    IDS_PATH,
    N_FFT,
    N_JOBS,
    N_MFCC,
    SR,
    X_PATH,
)

def summarize(features: np.ndarray) -> np.ndarray:
    feat = np.atleast_2d(features)
    return np.concatenate([feat.mean(axis=1), feat.std(axis=1)])

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    return np.concatenate([
        summarize(mfcc),
        summarize(centroid),
        summarize(bandwidth),
        summarize(contrast),
        summarize(zcr),
        summarize(chroma),
    ]).astype(np.float32)

def feature_column_names() -> list[str]:
    names = []
    names += [f"mfcc_{i+1}_mean" for i in range(N_MFCC)] + [f"mfcc_{i+1}_std" for i in range(N_MFCC)]
    names += ["centroid_mean", "centroid_std"]
    names += ["bandwidth_mean", "bandwidth_std"]
    names += [f"contrast_{i+1}_mean" for i in range(7)] + [f"contrast_{i+1}_std" for i in range(7)]
    names += ["zcr_mean", "zcr_std"]
    names += [f"chroma_{i+1}_mean" for i in range(12)] + [f"chroma_{i+1}_std" for i in range(12)]
    return names

def _extract_one(tid: int):
    import os

    # Must run inside the worker, before librosa does BLAS-heavy work.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    import librosa

    audio_path = utils.get_audio_path(str(AUDIO_DIR), tid)
    try:
        y, sr = librosa.load(audio_path, sr=SR, duration=DURATION)
        if y is None or len(y) < sr:
            raise ValueError("Audio file too short")
        return tid, extract_features(y, sr), None
    except Exception as e:
        return tid, None, (audio_path, repr(e))

def run_extraction(target_track_ids: np.ndarray):
    X, ids, bad = [], [], []
    todo = [int(t) for t in target_track_ids]
    print(f"Extracting features for {len(todo):,} tracks (n_jobs={N_JOBS})")

    results = Parallel(n_jobs=N_JOBS, backend="loky", return_as="generator")(
        delayed(_extract_one)(tid) for tid in todo
    )

    for tid, feat, err in tqdm(results, total=len(todo), desc="Features"):
        if err is not None:
            bad.append(err)
            continue
        X.append(feat)
        ids.append(tid)

    X = np.asarray(X, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int32)
    np.save(X_PATH, X)
    np.save(IDS_PATH, ids)
    if bad:
        pd.DataFrame(bad, columns=["path", "error"]).to_csv(BAD_FILES_CSV, index=False)
    print(f"Extracted {len(X):,} tracks (skipped {len(bad):,}).")
    return X, ids
