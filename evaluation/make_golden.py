"""Generate the golden set: datasets with known injected issues plus hard negatives."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent / "golden_sets"
N = 400
REGISTRY: dict = {}


def golden(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def ann(target, *expected, task="predictive modelling"):
    return {"target": target, "task": task,
            "expected": [{"issue_type": t, "column": c} for t, c in expected]}


def base(seed, n=N, k=4):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({"customer_id": np.arange(1000, 1000 + n)})
    for i in range(1, k + 1):
        df[f"f{i}"] = r.normal(50 + 10 * i, 5 + i, n)
    df["segment"] = r.choice(["retail", "sme", "corporate"], n, p=[0.5, 0.3, 0.2])
    p = 1 / (1 + np.exp(-0.5 * (df["f1"] - df["f1"].mean()) / df["f1"].std()))   # weak, legitimate signal
    df["y"] = (r.random(n) < p).astype(int)
    return df, r


def inject_missing(df, r, col, frac):
    idx = r.choice(len(df), int(round(frac * len(df))), replace=False)
    df.loc[idx, col] = np.nan


# ------------------------------ positives: one issue each ------------------------------
@golden("missing_moderate")
def _():
    df, r = base(1); inject_missing(df, r, "f2", 0.12)
    return df, ann("y", ("missing_values", "f2"))


@golden("missing_heavy_categorical")
def _():
    df, r = base(2); inject_missing(df, r, "segment", 0.45)
    return df, ann("y", ("missing_values", "segment"))


@golden("missing_low_but_real")
def _():
    df, r = base(3); inject_missing(df, r, "f3", 0.02)
    return df, ann("y", ("missing_values", "f3"))


@golden("outliers_extreme")
def _():
    df, r = base(4); idx = r.choice(N, 8, replace=False); df.loc[idx, "f2"] = df["f2"].mean() + 40 * df["f2"].std()
    return df, ann("y", ("outliers", "f2"))


@golden("outliers_on_skewed_feature")
def _():
    df, r = base(5); df["income"] = r.lognormal(10, 0.5, N)
    idx = r.choice(N, 8, replace=False); df.loc[idx, "income"] *= 60
    return df, ann("y", ("outliers", "income"))


@golden("leakage_numeric")
def _():
    df, r = base(6); df["refund_flag_score"] = df["y"] * 4 + r.normal(0, 0.05, N)
    return df, ann("y", ("leakage", "refund_flag_score"))


@golden("leakage_categorical")
def _():
    df, r = base(7); df["outcome_label"] = np.where(df["y"] == 1, "closed_won", "closed_lost")
    return df, ann("y", ("leakage", "outcome_label"))


@golden("leakage_regression_target")
def _():
    df, r = base(8); df["target"] = 3 * df["f1"] + r.normal(0, 4, N)   # strong but LEGITIMATE dependence on f1
    df = df.drop(columns=["y"]); df["leak"] = 2 * df["target"] + r.normal(0, 0.05, N)
    # leak derives from target = 3*f1 + noise, so corr(f1, leak) = 0.976 by construction: a second true issue
    return df, ann("target", ("leakage", "leak"), ("multicollinearity", "f1~leak"))


@golden("imbalance_severe")
def _():
    df, r = base(9); df["y"] = 0; df.loc[r.choice(N, 12, replace=False), "y"] = 1
    return df, ann("y", ("imbalance", "y"))


@golden("imbalance_moderate")
def _():
    df, r = base(10); df["y"] = 0; df.loc[r.choice(N, 52, replace=False), "y"] = 1   # 13%
    return df, ann("y", ("imbalance", "y"))


@golden("duplicates_heavy")
def _():
    df, r = base(11); df = pd.concat([df, df.iloc[r.choice(N, 60, replace=False)]], ignore_index=True)
    return df, ann("y", ("duplicates", None))


@golden("collinear_pair")
def _():
    df, r = base(12); df["f1_scaled"] = df["f1"] * 3 + r.normal(0, 0.8, N)
    return df, ann("y", ("multicollinearity", "f1~f1_scaled"))


@golden("constant_column")
def _():
    df, _r = base(13); df["country"] = "FR"
    return df, ann("y", ("constant_column", "country"))


# ------------------------------ mixed / wide ------------------------------
@golden("mixed_A")
def _():
    df, r = base(14); inject_missing(df, r, "f2", 0.18)
    df.loc[r.choice(N, 6, replace=False), "f3"] = 900.0
    df["leak"] = df["y"] * 2.5 + r.normal(0, 0.02, N)
    df = pd.concat([df, df.iloc[r.choice(N, 40, replace=False)]], ignore_index=True)
    return df, ann("y", ("missing_values", "f2"), ("outliers", "f3"), ("leakage", "leak"), ("duplicates", None))


@golden("mixed_B")
def _():
    df, r = base(15); df["f4_copy"] = df["f4"] * 1.5 + r.normal(0, 0.3, N)
    df["y"] = 0; df.loc[r.choice(N, 16, replace=False), "y"] = 1
    df["region"] = "EU"; inject_missing(df, r, "segment", 0.08)
    return df, ann("y", ("multicollinearity", "f4~f4_copy"), ("imbalance", "y"),
                   ("constant_column", "region"), ("missing_values", "segment"))


@golden("wide_60_columns")
def _():
    r = np.random.default_rng(16)
    df = pd.DataFrame({f"c{i}": r.normal(i, 1 + (i % 5), N) for i in range(60)})
    df.insert(0, "customer_id", np.arange(N))
    df["y"] = (r.random(N) < 0.5).astype(int)
    inject_missing(df, r, "c7", 0.2)
    df.loc[r.choice(N, 10, replace=False), "c33"] = 5000.0
    df["c50"] = df["y"] * 3 + r.normal(0, 0.02, N)           # replaces a random column with a leaky one
    return df, ann("y", ("missing_values", "c7"), ("outliers", "c33"), ("leakage", "c50"))


# ------------------------------ hard negatives: expected NO findings ------------------------------
@golden("clean_basic")
def _():
    df, _r = base(20); return df, ann("y")


@golden("clean_naturally_skewed")
def _():
    df, r = base(21); df["wait_time"] = r.exponential(5, N); df["spend"] = r.lognormal(4, 0.6, N)
    return df, ann("y")


@golden("clean_near_thresholds")
def _():
    df, r = base(22)
    inject_missing(df, r, "f2", 0.005)                                     # 0.5% < 1% flag level
    df["f1_noisy"] = df["f1"] + r.normal(0, 4.5, N)                        # correlated ~0.75, not redundant
    df["y"] = 0; df.loc[r.choice(N, 100, replace=False), "y"] = 1          # 25% minority > 20%
    df = pd.concat([df, df.iloc[r.choice(N, 2, replace=False)]], ignore_index=True)   # 0.5% duplicates
    return df, ann("y")


@golden("clean_ids_text_dates")
def _():
    df, r = base(23)
    df["order_uuid"] = [f"ord-{i:06d}" for i in range(N)]
    df["comment"] = [f"free text note number {i} about the customer visit and follow up" for i in range(N)]
    df["signup_date"] = pd.date_range("2022-01-01", periods=N, freq="D").strftime("%Y-%m-%d")
    return df, ann("y")


@golden("clean_strong_but_legit_signal")
def _():
    df, r = base(24); df["y"] = (df["f2"] + r.normal(0, 7, N) > df["f2"].mean()).astype(int)   # f2 explains y partly
    return df, ann("y")


@golden("clean_regression")
def _():
    df, r = base(25); df["price"] = 2 * df["f1"] + 0.5 * df["f2"] + r.normal(0, 8, N); df = df.drop(columns=["y"])
    return df, ann("price")


# ------------------------------ real data (scikit-learn bundled, no network) ------------------------------
@golden("real_wine_untouched")
def _():
    from sklearn.datasets import load_wine
    df = load_wine(as_frame=True).frame
    return df, ann("target")


@golden("real_wine_injected")
def _():
    from sklearn.datasets import load_wine
    df = load_wine(as_frame=True).frame; r = np.random.default_rng(30)
    inject_missing(df, r, "alcohol", 0.10)
    df.loc[r.choice(len(df), 6, replace=False), "magnesium"] = 400.0
    return df, ann("target", ("missing_values", "alcohol"), ("outliers", "magnesium"))


@golden("real_iris_known_collinearity")
def _():
    from sklearn.datasets import load_iris
    df = load_iris(as_frame=True).frame       # petal length vs width: documented r = 0.963
    return df, ann("target", ("multicollinearity", "petal length (cm)~petal width (cm)"))


@golden("real_diabetes_untouched")
def _():
    from sklearn.datasets import load_diabetes
    df = load_diabetes(as_frame=True).frame   # regression; most correlated pair s1~s2 is r = 0.90
    return df, ann("target")


def main():
    (ROOT / "datasets").mkdir(parents=True, exist_ok=True)
    (ROOT / "annotations").mkdir(parents=True, exist_ok=True)
    for name, fn in REGISTRY.items():
        df, a = fn()
        df.to_csv(ROOT / "datasets" / f"{name}.csv", index=False)
        (ROOT / "annotations" / f"{name}.json").write_text(json.dumps(a, indent=2), encoding="utf-8")
    print(f"wrote {len(REGISTRY)} golden datasets to {ROOT}")


if __name__ == "__main__":
    main()
