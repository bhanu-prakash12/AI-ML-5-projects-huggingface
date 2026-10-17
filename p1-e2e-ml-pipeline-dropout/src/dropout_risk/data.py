"""Fetch, validate, split the UCI 697 dataset."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "target"
BINARY_TARGET = "dropout"  # D2: Dropout vs rest
CLASSES = {"Dropout", "Enrolled", "Graduate"}
N_ROWS, N_FEATURES = 4424, 36

# Inclusive plausible ranges (observed range widened a little). Binary flags are 0/1.
_BIN = (0, 1)
RANGES: dict[str, tuple[float, float]] = {
    "marital_status": (1, 6), "application_mode": (1, 60), "application_order": (0, 9),
    "course": (1, 10000), "daytime_evening_attendance": _BIN, "previous_qualification": (1, 50),
    "previous_qualification_grade": (0, 200), "nacionality": (1, 120),
    "mother_s_qualification": (1, 50), "father_s_qualification": (1, 50),
    "mother_s_occupation": (0, 200), "father_s_occupation": (0, 200),
    "admission_grade": (0, 200), "displaced": _BIN, "educational_special_needs": _BIN,
    "debtor": _BIN, "tuition_fees_up_to_date": _BIN, "gender": _BIN, "scholarship_holder": _BIN,
    "age_at_enrollment": (16, 100), "international": _BIN,
    "unemployment_rate": (0, 30), "inflation_rate": (-5, 15), "gdp": (-10, 10),
}
for _sem in ("1st", "2nd"):
    for _k, _hi in (("credited", 40), ("enrolled", 40), ("evaluations", 60), ("approved", 40),
                    ("without_evaluations", 40)):
        RANGES[f"curricular_units_{_sem}_sem_{_k}"] = (0, _hi)
    RANGES[f"curricular_units_{_sem}_sem_grade"] = (0, 20)


class ValidationError(ValueError):
    """Raised when the dataset violates the expected schema or ranges."""


def snake(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def fetch_raw(uci_id: int = 697) -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=uci_id)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    return normalize(df)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """snake_case columns, string target, derived binary target."""
    df = df.rename(columns={c: snake(c) for c in df.columns})
    df[TARGET] = df[TARGET].astype(str).str.strip()
    return df


def validate(df: pd.DataFrame, expect_rows: int | None = N_ROWS) -> pd.DataFrame:
    """Fail loudly on schema, null, target or range problems."""
    problems: list[str] = []
    missing = (set(RANGES) | {TARGET}) - set(df.columns)
    if missing:
        problems.append(f"missing columns: {sorted(missing)}")
    extra = set(df.columns) - set(RANGES) - {TARGET, BINARY_TARGET}
    if extra:
        problems.append(f"unexpected columns: {sorted(extra)}")
    if expect_rows is not None and len(df) != expect_rows:
        problems.append(f"expected {expect_rows} rows, got {len(df)}")
    if problems:
        raise ValidationError("; ".join(problems))
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        problems.append(f"nulls: {nulls.to_dict()}")
    bad = set(df[TARGET].dropna().unique()) - CLASSES
    if bad:
        problems.append(f"unknown target labels: {sorted(bad)}")
    for col, (lo, hi) in RANGES.items():
        out = int(((df[col] < lo) | (df[col] > hi)).sum())
        if out:
            problems.append(f"{col}: {out} values outside [{lo}, {hi}]")
    if problems:
        raise ValidationError("; ".join(problems))
    return df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[BINARY_TARGET] = (df[TARGET] == "Dropout").astype("int8")
    return df


def split(df: pd.DataFrame, fractions: dict[str, float], seed: int = 42) -> dict[str, pd.DataFrame]:
    """Deterministic stratified train/val/test split (stratified on the 3-class target)."""
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    train, rest = train_test_split(
        df, train_size=fractions["train"], stratify=df[TARGET], random_state=seed
    )
    val_share = fractions["val"] / (fractions["val"] + fractions["test"])
    val, test = train_test_split(rest, train_size=val_share, stratify=rest[TARGET], random_state=seed)
    return {k: v.reset_index(drop=True) for k, v in (("train", train), ("val", val), ("test", test))}


def write_splits(splits: dict[str, pd.DataFrame], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(out / f"{name}.parquet", index=False)
