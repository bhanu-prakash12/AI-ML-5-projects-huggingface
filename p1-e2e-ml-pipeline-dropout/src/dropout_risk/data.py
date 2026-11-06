"""Fetch, clean, validate and split the UCI dropout dataset.

This module is the data entry point for the project. It downloads the public UCI
student dataset via `ucimlrepo.fetch_ucirepo(...)`, normalizes the column names to a
consistent format, validates the schema and value ranges, adds the binary target used
for risk scoring, and creates the train/validation/test splits used by the model.

External references:
- `ucimlrepo.fetch_ucirepo`: public dataset loader from the UCI ML Repository.
- `pandas.concat`: combines the feature and target DataFrames into one table.
- `sklearn.model_selection.train_test_split`: creates deterministic train/val/test sets.
"""

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
    "marital_status": (1, 6),
    "application_mode": (1, 60),
    "application_order": (0, 9),
    "course": (1, 10000),
    "daytime_evening_attendance": _BIN,
    "previous_qualification": (1, 50),
    "previous_qualification_grade": (0, 200),
    "nacionality": (1, 120),
    "mother_s_qualification": (1, 50),
    "father_s_qualification": (1, 50),
    "mother_s_occupation": (0, 200),
    "father_s_occupation": (0, 200),
    "admission_grade": (0, 200),
    "displaced": _BIN,
    "educational_special_needs": _BIN,
    "debtor": _BIN,
    "tuition_fees_up_to_date": _BIN,
    "gender": _BIN,
    "scholarship_holder": _BIN,
    "age_at_enrollment": (16, 100),
    "international": _BIN,
    "unemployment_rate": (0, 30),
    "inflation_rate": (-5, 15),
    "gdp": (-10, 10),
}
for _sem in ("1st", "2nd"):
    for _k, _hi in (
        ("credited", 40),
        ("enrolled", 40),
        ("evaluations", 60),
        ("approved", 40),
        ("without_evaluations", 40),
    ):
        RANGES[f"curricular_units_{_sem}_sem_{_k}"] = (0, _hi)
    RANGES[f"curricular_units_{_sem}_sem_grade"] = (0, 20)


class ValidationError(ValueError):
    """Raised when the dataset violates the expected schema or ranges."""


def snake(name: str) -> str:
    """Convert a raw column name into a clean Python-friendly identifier.

    Example:
        "Mother's Qualification" -> "mother_s_qualification"
    """
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def fetch_raw(uci_id: int = 697) -> pd.DataFrame:
    """Download and normalize the raw UCI dataset.

    This uses `ucimlrepo.fetch_ucirepo(id=697)`, which returns a dataset object with
    separate `features` and `targets` DataFrames. We combine them into one table and
    then normalize the column names so the rest of the project can work with a stable
    schema.
    """
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=uci_id)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    return normalize(df)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the dataset columns and target labels.

    The target column is kept as a string so it is easy to validate against the known
    class names (`Dropout`, `Enrolled`, `Graduate`).
    """
    df = df.rename(columns={c: snake(c) for c in df.columns})
    df[TARGET] = df[TARGET].astype(str).str.strip()
    return df


def validate(df: pd.DataFrame, expect_rows: int | None = N_ROWS) -> pd.DataFrame:
    """Fail loudly when the dataset is malformed or out of a plausible range.

    This is intentionally strict so data issues show up early. The validation checks:
    - required columns are present,
    - there are no unexpected columns,
    - the row count matches the dataset expectation,
    - nulls are not silently accepted,
    - the target labels match the known classes,
    - numeric features stay within the expected domains.
    """
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
    """Create the binary target used for dropout-vs-rest classification.

    This is the label used to train the risk score, while the original three-class label
    remains available for analysis and fairness checks.
    """
    df = df.copy()
    df[BINARY_TARGET] = (df[TARGET] == "Dropout").astype("int8")
    return df


def split(
    df: pd.DataFrame, fractions: dict[str, float], seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Split the data into train, validation and test sets.

    `train_test_split` from scikit-learn keeps the class balance stable by stratifying on
    the original target. This reduces leakage and makes the validation signal more
    representative.
    """
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("split fractions must sum to 1")

    train, rest = train_test_split(
        df,
        train_size=fractions["train"],
        stratify=df[TARGET],
        random_state=seed,
    )

    val_share = fractions["val"] / (fractions["val"] + fractions["test"])
    val, test = train_test_split(
        rest,
        train_size=val_share,
        stratify=rest[TARGET],
        random_state=seed,
    )

    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def write_splits(splits: dict[str, pd.DataFrame], out: Path) -> None:
    """Persist the processed splits to parquet files for downstream training steps."""
    out.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(out / f"{name}.parquet", index=False)
