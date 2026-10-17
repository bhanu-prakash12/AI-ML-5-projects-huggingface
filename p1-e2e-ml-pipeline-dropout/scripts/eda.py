"""Write reports/eda.md: class balance, missingness, top correlations."""
from __future__ import annotations

import _boot  # noqa: F401  (adds src/ to sys.path)
import argparse
import logging

import pandas as pd

from dropout_risk import console, data
from dropout_risk.config import project_path


def main() -> None:
    root = project_path("data") / "processed"
    parts = {k: pd.read_parquet(root / f"{k}.parquet") for k in ("train", "val", "test")}
    df = pd.concat(parts.values(), ignore_index=True)
    feats = [c for c in df.columns if c in data.RANGES]
    counts = df[data.TARGET].value_counts()
    miss = df[feats].isna().sum()
    corr = df[feats].corrwith(df[data.BINARY_TARGET]).dropna()
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(10)
    lines = ["# EDA: UCI 697 student dropout", "", f"Rows: {len(df)}, features: {len(feats)}", "",
             "## Split sizes", "| split | rows | dropout rate |", "|---|---|---|"]
    lines += [f"| {k} | {len(v)} | {v[data.BINARY_TARGET].mean():.3f} |" for k, v in parts.items()]
    lines += ["", "## Class balance (3-class target)", "| class | rows | share |", "|---|---|---|"]
    lines += [f"| {k} | {n} | {n / len(df):.3f} |" for k, n in counts.items()]
    lines += ["", f"Binary target (D2) base rate Dropout vs rest: {df[data.BINARY_TARGET].mean():.3f}",
              "", "## Missingness", f"Total missing cells in features: {int(miss.sum())}",
              "", "## Top 10 correlations with dropout (Pearson)", "| feature | r |", "|---|---|"]
    lines += [f"| {k} | {v:+.3f} |" for k, v in top.items()]
    lines += ["", "> Note: semester features are end-of-year-2 data (D1), so this is a late-stage model,"
              " not an early warning."]
    out = project_path("reports") / "eda.md"
    out.write_text("\n".join(lines) + "\n")
    logging.info("wrote %s", out)
    console.ok(f"wrote {out}")
    console.table([(k, f"{v:+.3f}") for k, v in top.head(5).items()],
                  headers=("top-5 feature", "Pearson r with dropout"))


def cli() -> None:
    argparse.ArgumentParser(
        description="Write reports/eda.md: class balance, missingness, top correlations.",
        epilog="Input: data/processed/*.parquet. Output: reports/eda.md. "
               "Sample: python scripts/eda.py").parse_args()
    console.setup_logging()
    console.banner("EDA", "Summarise class balance, missing cells and strongest correlations.",
                   "late-semester grades/approvals dominate (a late-stage signal).")
    main()
    console.note("cheap sanity check before modelling.",
                 "linear correlation only; says nothing about causes.",
                 "automate as a data-quality report per ingest.")


if __name__ == "__main__":
    cli()
