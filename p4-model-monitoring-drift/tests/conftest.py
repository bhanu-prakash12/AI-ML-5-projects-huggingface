import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"V{i}" for i in range(1, 6)])
    df["Amount"] = rng.exponential(50, n)
    df["Class"] = (df.V1 + df.V2 > 2.6).astype(int)
    return df
