"""Fetch UCI 697 -> data/raw, validate, stratified split -> data/processed/*.parquet."""
from __future__ import annotations

import _boot  # noqa: F401  (adds src/ to sys.path)
import argparse
import logging

from dropout_risk import console, data
from dropout_risk.config import load_config, project_path

log = logging.getLogger("make_data")


def main() -> None:
    cfg = load_config()
    root = project_path("data")
    raw_path = root / "raw" / "students.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    console.step(1, 3, "downloading UCI dataset 697 (needs network; ~4.4k rows)")
    df = data.fetch_raw(cfg["data"]["uci_id"])
    console.step(2, 3, "validating schema and ranges")
    data.validate(df)
    data.add_binary_target(df).to_parquet(raw_path, index=False)
    console.ok(f"raw rows={len(df)} -> {raw_path}")
    console.step(3, 3, "stratified train/val/test split")
    splits = data.split(data.add_binary_target(df), cfg["data"]["split"], cfg["seed"])
    data.write_splits(splits, root / "processed")
    console.table([(k, len(v), f"{v[data.BINARY_TARGET].mean():.3f}") for k, v in splits.items()],
                  headers=("split", "rows", "dropout rate"))
    assert sum(len(v) for v in splits.values()) == data.N_ROWS


def cli() -> None:
    argparse.ArgumentParser(
        description="Fetch the UCI student-dropout dataset, validate it and write splits.",
        epilog="Input: network access to UCI (id 697). Output: data/raw/students.parquet and "
               "data/processed/{train,val,test}.parquet. Sample: python scripts/make_data.py"
    ).parse_args()
    console.setup_logging()
    console.banner("Make data", "Fetch -> validate -> split (70/15/15, stratified, seed 42).",
                   "three split sizes with near-identical dropout rates.")
    main()
    console.note(
        "stratified splits keep the dropout rate equal across train/val/test.",
        "live fetch from UCI: fails offline; dataset is a single Portuguese institution.",
        "swap fetch_raw for a warehouse query; keep validate() as the contract check.")


if __name__ == "__main__":
    cli()
