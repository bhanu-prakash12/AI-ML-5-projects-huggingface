import pandas as pd
import pytest

from dropout_risk import data

FR = {"train": 0.70, "val": 0.15, "test": 0.15}


def make_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    cols = {c: rng.integers(int(lo), int(lo) + 2, n) for c, (lo, _hi) in data.RANGES.items()}
    df = pd.DataFrame(cols)
    df[data.TARGET] = rng.choice(["Dropout", "Enrolled", "Graduate"], n, p=[0.32, 0.18, 0.5])
    return df


def test_snake_names():
    assert data.snake("Mother's qualification") == "mother_s_qualification"
    assert data.snake("Curricular units 1st sem (credited)") == "curricular_units_1st_sem_credited"
    assert data.snake("Daytime/evening attendance") == "daytime_evening_attendance"


def test_validate_ok():
    data.validate(make_df(), expect_rows=None)


def test_validate_range_fails():
    df = make_df()
    df.loc[0, "age_at_enrollment"] = 5
    with pytest.raises(data.ValidationError, match="age_at_enrollment"):
        data.validate(df, expect_rows=None)


def test_validate_null_missing_and_label_fail():
    df = make_df()
    df.loc[0, "gdp"] = None
    with pytest.raises(data.ValidationError, match="nulls"):
        data.validate(df, expect_rows=None)
    with pytest.raises(data.ValidationError, match="missing columns"):
        data.validate(make_df().drop(columns=["gdp"]), expect_rows=None)
    bad = make_df()
    bad.loc[0, data.TARGET] = "Quit"
    with pytest.raises(data.ValidationError, match="unknown target"):
        data.validate(bad, expect_rows=None)


def test_row_count_check():
    with pytest.raises(data.ValidationError, match="expected"):
        data.validate(make_df())


def test_split_sizes_stratified_deterministic_disjoint():
    df = data.add_binary_target(make_df(1000)).reset_index(names="rid")
    a, b = data.split(df, FR, 42), data.split(df, FR, 42)
    assert sum(len(v) for v in a.values()) == 1000
    assert [len(a[k]) for k in ("train", "val", "test")] == [700, 150, 150]
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])
    ids = [set(v["rid"]) for v in a.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    base = df[data.BINARY_TARGET].mean()
    for v in a.values():
        assert abs(v[data.BINARY_TARGET].mean() - base) < 0.02


def test_binary_target():
    df = data.add_binary_target(make_df())
    assert ((df[data.TARGET] == "Dropout") == (df[data.BINARY_TARGET] == 1)).all()
