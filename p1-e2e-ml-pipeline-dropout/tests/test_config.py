from dropout_risk.config import load_config, project_path


def test_config_profile():
    cfg = load_config()
    assert cfg["seed"] == 42 and cfg["runtime"]["n_jobs"] == 2
    assert project_path("reports").is_dir()
