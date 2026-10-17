"""Point the `champion` alias back at the baseline model (start the demo story from the same model every time)."""
import sys

sys.path.insert(0, "src")
from fraud_monitor import console, model  # noqa: E402

if __name__ == "__main__":
    console.banner("reset champion", "Move the MLflow alias `champion` to the baseline version in reports/baseline.json.",
                   "one line: champion = baseline version")
    console.ok(f"champion is now v{model.reset_champion()} (reload a running API with POST /reload)")
