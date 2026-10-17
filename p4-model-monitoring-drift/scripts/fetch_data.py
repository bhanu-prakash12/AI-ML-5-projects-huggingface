"""Download creditcard (cached) and write the reference + stream parquet files."""
import argparse
import sys

sys.path.insert(0, "src")
from fraud_monitor import config, console, data  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        epilog="Inputs: OpenML data_id 1597 (only on first run; then data/creditcard.parquet is reused).\n"
               "Outputs: data/reference.parquet (training period), data/stream.parquet (10 simulated days).\n"
               "Example: python run.py data",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    console.setup_logging()
    console.banner("data", "Load the creditcard fraud data, split by row order 60/40 into reference and stream, "
                   "sample the stream into 10 days (all frauds kept).", "a summary table and the per-day fraud counts")
    console.step(1, 3, "loading raw data (cached after the first download)")
    raw = data.load_raw()
    console.step(2, 3, "splitting reference / stream")
    ref, stream = data.split(raw)
    s = data.make_stream(stream)
    console.step(3, 3, "writing parquet files")
    s.to_parquet(config.path("stream"))
    ref.to_parquet(config.path("data") / "reference.parquet")
    console.table([{"part": n, "rows": len(d), "fraud rate": f"{d.Class.mean():.5f}"}
                   for n, d in (("raw", raw), ("reference", ref), ("stream (full)", stream), ("stream (sampled)", s))])
    console.table([{"day": int(d), "rows": int(r["size"]), "frauds": int(r["sum"])}
                   for d, r in s.groupby("day").Class.agg(["size", "sum"]).iterrows()], title="stream days")
    console.note("time-ordered split means no future leakage into the reference",
                 "the stream is enriched with frauds (higher base rate than real life) so per-window PR-AUC is measurable",
                 "replace the parquet files with a warehouse table and a streaming ingest")
