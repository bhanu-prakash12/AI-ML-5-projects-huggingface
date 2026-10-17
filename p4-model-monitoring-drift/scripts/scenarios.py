"""CLI view of the drift scenarios: `python scripts/scenarios.py [A|B|C]` prints what each does to the data.
The implementation lives in src/fraud_monitor/scenarios.py (importable by tests/simulator)."""
import argparse
import sys

sys.path.insert(0, "src")
import pandas as pd  # noqa: E402

from fraud_monitor import config, console, data, scenarios  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Inputs: data/*.parquet. Outputs: a table on screen (nothing is written).")
    ap.add_argument("names", nargs="*", default=["A", "B", "C"], help="scenarios to show (default: all)")
    a = ap.parse_args()
    console.banner("drift scenarios", (scenarios.__doc__ or "").strip(), "one row per scenario: day 8 before -> after")
    ref = pd.read_parquet(config.path("data") / "reference.parquet")
    s = pd.read_parquet(config.path("stream"))
    feats = data.feature_cols(ref)
    rows = []
    for name in a.names:
        d = s[s.day == 8]
        o = scenarios.apply(d, 8, ref, feats, name, 1)
        rows.append({"scenario": name, "rows": len(d), "fraud before": int(d.Class.sum()), "fraud after": int(o.Class.sum()),
                     "Amount mean before": f"{d.Amount.mean():.1f}", "after": f"{o.Amount.mean():.1f}"})
    console.table(rows, title="effect on stream day 8")
