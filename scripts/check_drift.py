"""Compare two datasets for distribution drift."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datascout.tools.drift import detect_drift  # noqa: E402
from datascout.tools.profiling import load_dataframe  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python scripts/check_drift.py <reference> <current> [target]")
        return 2
    r = detect_drift(load_dataframe(sys.argv[1]), load_dataframe(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
    print(f"rows: {r.n_reference_rows} (reference) vs {r.n_current_rows} (current)   compared: {len(r.columns_compared)} columns")
    print(f"{r.summary}   statistics re-verified: {r.verified}")
    for w in r.warnings:
        print(f"warning: {w}")
    for f in r.findings:
        print(f"drift: {f.column} | {f.severity.value} | {f.statistic_name}={f.statistic:.3f} p={f.p_value:.2g} | {f.recommendation}")
    for c, why in r.columns_skipped.items():
        print(f"skipped: {c} ({why})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
