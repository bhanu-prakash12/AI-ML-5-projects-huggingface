"""Drift scenarios applied to a simulated day of traffic (all seeded, all synthetic).

A  covariate shift : Amount x3 (+ multiplicative lognormal noise) for every row, labels untouched.
B  new fraud pattern: `frac` of the day's legit rows are overwritten by a synthetic fraud cluster
                      (label=1). The cluster sits `shift_sigma` legit-std away from the legit mean, in
                      the direction OPPOSITE to where known fraud lives, on the `n_feats` most
                      fraud-discriminative features -> a region absent from training that the model
                      scores as clearly legitimate (a fraudster who learned to look "more normal").
C  concept drift    : in the segment V1 > reference-median, `flip_frac` of fraud labels become 0 and
                      the same number of legit rows become 1 (same base rate, changed P(y|x)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def cluster_spec(ref: pd.DataFrame, feats: list[str], n_feats: int) -> pd.DataFrame:
    """Mean/std of legit rows and the sign pointing away from fraud, for the top-n discriminative features."""
    leg, fr = ref[ref.Class == 0][feats], ref[ref.Class == 1][feats]
    std = leg.std().replace(0, 1.0)
    gap = (fr.mean() - leg.mean()) / std
    top = gap.abs().sort_values(ascending=False).index[:n_feats]
    return pd.DataFrame({"mean": leg.mean()[top], "std": std[top], "away": -np.sign(gap[top])})


def apply(day_df: pd.DataFrame, day: int, ref: pd.DataFrame, feats: list[str],
          scenario: str | None = None, start_day: int | None = None) -> pd.DataFrame:
    cfg = config.load()
    sc = cfg["scenario"]
    name = sc["name"] if scenario is None else scenario
    start = sc["start_day"] if start_day is None else start_day
    out = day_df.copy()
    if name in (None, "none", "") or day < start:
        return out
    rng = np.random.default_rng(cfg["seed"] + day)
    if name == "A":
        p = sc["A"]
        out["Amount"] = out["Amount"] * p["amount_mult"] * rng.lognormal(0, p["noise"], len(out))
    elif name == "B":
        p = sc["B"]
        spec = cluster_spec(ref, feats, p["n_feats"])
        legit_idx = out.index[out["Class"] == 0].to_numpy()
        pick = rng.choice(legit_idx, size=int(round(p["frac"] * len(out))), replace=False)
        for f, r in spec.iterrows():
            out.loc[pick, f] = r["mean"] + r["away"] * p["shift_sigma"] * r["std"] + rng.normal(0, 0.3 * r["std"], len(pick))
        out.loc[pick, "Class"] = 1
    elif name == "C":
        p = sc["C"]
        seg = out[p["segment_feature"]] > ref[p["segment_feature"]].median()
        fr = out.index[seg & (out.Class == 1)].to_numpy()
        lg = out.index[seg & (out.Class == 0)].to_numpy()
        k = int(round(p["flip_frac"] * len(fr)))
        if k:
            out.loc[rng.choice(fr, k, replace=False), "Class"] = 0
            out.loc[rng.choice(lg, k, replace=False), "Class"] = 1
    else:
        raise ValueError(f"unknown scenario {name}")
    return out
