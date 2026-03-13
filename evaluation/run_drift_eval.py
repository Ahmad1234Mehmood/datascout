"""Drift detector evaluation on scikit-learn datasets: null split and planted shift."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine  # noqa: E402

from datascout.tools.drift import detect_drift  # noqa: E402

DATASETS = {"wine": load_wine, "iris": load_iris, "diabetes": load_diabetes, "breast_cancer": load_breast_cancer}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=100)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    false_alarms = comparisons = planted_found = planted = 0
    rows = []
    for name, loader in DATASETS.items():
        df = loader(as_frame=True).frame
        num = [c for c in df.columns if df[c].nunique() > 10]
        fa = hit = 0
        for _ in range(args.splits):
            idx = rng.permutation(len(df))
            a, b = df.iloc[idx[: len(df) // 2]], df.iloc[idx[len(df) // 2:]].copy()
            r = detect_drift(a, b)
            assert r.verified, r.verification_errors
            fa += bool(r.findings)
            col = num[int(rng.integers(len(num)))]
            b[col] = b[col] + 0.5 * df[col].std()
            r2 = detect_drift(a, b)
            assert r2.verified, r2.verification_errors
            hit += any(f.column == col for f in r2.findings)
        rows.append((name, len(df), fa, hit))
        false_alarms += fa; comparisons += args.splits; planted_found += hit; planted += args.splits
    print("| dataset | rows | false alarms (null splits) | planted +0.5 std shift found |")
    print("|---|---|---|---|")
    for name, n, fa, hit in rows:
        print(f"| {name} | {n} | {fa} / {args.splits} | {hit} / {args.splits} |")
    fa_rate, recall = false_alarms / comparisons, planted_found / planted
    print(f"\nfalse alarm rate {fa_rate:.3f}   planted-shift recall {recall:.3f}")
    ok = fa_rate <= 0.02 and recall >= 0.5
    print("gate:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
