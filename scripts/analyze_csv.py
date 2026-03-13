"""Run DataScout on a CSV, XLSX or Hugging Face dataset and print the findings."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datascout.agents.graph import analyze_dataframe, analyze_file  # noqa: E402
from datascout.tools.hf_loader import load_hf_dataset  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print('usage: python scripts/analyze_csv.py <file> "<task>" [target_column]')
        return 2
    path, task = sys.argv[1], sys.argv[2]
    target = sys.argv[3] if len(sys.argv) > 3 else None
    if path.startswith("hf:"):
        report, clean = analyze_dataframe(load_hf_dataset(path[3:]), task, target, name=path[3:])
    else:
        report, clean = analyze_file(path, task, target)
    print(f"provider: {report.llm_provider}   rows: {report.n_rows_before} -> {report.n_rows_after}   "
          f"cols: {report.n_cols_before} -> {report.n_cols_after}")
    print(f"quality score: {report.quality_score_before} -> {report.quality_score_after}   "
          f"human review: {report.needs_human_review}")
    print(f"summary: {report.summary}")
    for w in report.warnings:
        print(f"warning: {w}")
    for f in report.findings:
        print(f"finding: {f.issue_type} | {f.column} | {f.severity} | {f.evidence}")
    for s in report.preparation_steps:
        print(f"step: {s.action.value} | {s.column}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
