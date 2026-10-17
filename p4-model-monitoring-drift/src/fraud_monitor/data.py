"""Data loading: OpenML creditcard -> parquet cache; time-ordered split."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def load_raw() -> pd.DataFrame:
    p = config.path("raw")
    if p.exists():
        return pd.read_parquet(p)
    from sklearn.datasets import fetch_openml

    cfg = config.load()
    log.info("downloading OpenML creditcard (cached to %s)", p)
    df = fetch_openml(data_id=cfg["data"]["openml_id"], as_frame=True, parser="auto").frame
    df["Class"] = df["Class"].astype(int)
    df = df.astype({c: "float64" for c in df.columns if c != "Class"})
    df.reset_index(drop=True).to_parquet(p)  # OpenML 1597 has no Time column; rows are already time-ordered
    return pd.read_parquet(p)


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop = set(config.load()["data"]["drop_cols"]) | {config.load()["data"]["target"]}
    return [c for c in df.columns if c not in drop]


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First 60% (time order) = reference/training; rest = production stream."""
    n_ref = int(len(df) * config.load()["data"]["ref_frac"])
    return df.iloc[:n_ref].reset_index(drop=True), df.iloc[n_ref:].reset_index(drop=True)


def make_stream(stream: pd.DataFrame) -> pd.DataFrame:
    """Small profile: keep every fraud + a random sample of non-fraud, preserving time order,
    and assign `day` (equal-size consecutive chunks)."""
    cfg = config.load()
    rng = np.random.default_rng(cfg["seed"])
    neg = stream.index[stream["Class"] == 0].to_numpy()
    keep_neg = rng.choice(neg, size=min(cfg["stream_neg_keep"], len(neg)), replace=False)
    keep = np.sort(np.concatenate([stream.index[stream["Class"] == 1].to_numpy(), keep_neg]))
    s = stream.loc[keep].reset_index(drop=True)
    s["day"] = np.minimum(np.arange(len(s)) * cfg["days"] // len(s), cfg["days"] - 1) + 1
    return s
