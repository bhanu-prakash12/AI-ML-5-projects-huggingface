"""Gradio dashboard: run scenarios, monitor and retrain by clicking; metrics over time + alert markers."""
import sys

sys.path.insert(0, "src")
from fraud_monitor import config, console, dashboard  # noqa: E402

if __name__ == "__main__":
    port = config.load()["ports"]["ui"]
    console.banner("dashboard", f"Gradio UI on http://127.0.0.1:{port}", "buttons: 1 Simulate, 2 Monitor, 3 Retrain")
    dashboard.build().queue(default_concurrency_limit=1).launch(
        server_port=port, allowed_paths=[str(config.path("reports"))])
