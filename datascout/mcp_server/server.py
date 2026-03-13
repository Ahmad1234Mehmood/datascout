"""MCP server exposing DataScout tools; files are readable only under DATASCOUT_MCP_ROOT."""
from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from ..agents.graph import analyze_dataframe
from ..agents.quality_checker import run_checks
from ..llm import HeuristicClient
from ..tools.drift import detect_drift
from ..tools.profiling import compute_stats as _compute_stats
from ..tools.profiling import load_dataframe, profile_columns
from ..tools.quality import check_leakage as _check_leakage

mcp = MCPServer("datascout", instructions="Profile datasets, detect ML data-quality issues and produce a verified, cleaned dataset.")


def _safe_path(p: str) -> Path:
    root = Path(os.getenv("DATASCOUT_MCP_ROOT", ".")).resolve()
    path = (root / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"path must be inside {root}")
    return path


@mcp.tool()
def profile_dataset(file_path: str) -> list[dict]:
    """Column-level profile (type, missing %, uniqueness, basic stats) for a CSV/XLSX file."""
    df = load_dataframe(_safe_path(file_path))
    return [p.model_dump(mode="json") for p in profile_columns(df)]


@mcp.tool()
def check_quality(file_path: str, target: str | None = None) -> dict:
    """Detect missing values, outliers, leakage, class imbalance, duplicates, collinearity, constant columns.
    Pass the ML target column to enable leakage and imbalance checks."""
    df = load_dataframe(_safe_path(file_path))
    plan = HeuristicClient().plan_checks(profile_columns(df), "", target if target in df.columns else None)
    findings, errors = run_checks(df, plan, target if target in df.columns else None)
    return {"findings": [f.model_dump(mode="json") for f in findings], "errors": errors}


@mcp.tool()
def check_leakage(file_path: str, target: str) -> dict:
    """Find features that (almost) determine the target: correlation ratio, Pearson r or category purity >= 0.99."""
    df = load_dataframe(_safe_path(file_path))
    if target not in df.columns:
        raise ValueError(f"target '{target}' is not a column")
    return {"findings": [f.model_dump(mode="json") for f in _check_leakage(df, target=target)]}


@mcp.tool()
def compute_stats(file_path: str, column: str) -> dict:
    """Detailed statistics for one column (rows, missing, unique, mean, std, min, max, median)."""
    df = load_dataframe(_safe_path(file_path))
    if column not in df.columns:
        raise ValueError(f"column '{column}' not found")
    return _compute_stats(df, column)


@mcp.tool()
def check_drift(reference_path: str, current_path: str, target: str | None = None) -> dict:
    """Compare a reference dataset with a current one, column by column (KS test for numeric columns,
    chi-square for categories, with an effect-size floor). Every statistic is recomputed by separate code."""
    ref, cur = load_dataframe(_safe_path(reference_path)), load_dataframe(_safe_path(current_path))
    return detect_drift(ref, cur, target).model_dump(mode="json")


@mcp.tool()
def prepare_data(file_path: str, task: str, target: str | None = None, output_path: str | None = None) -> dict:
    """Run the full agent pipeline (profile, check, fix, verify). Returns the quality report and, if
    output_path is given, writes the cleaned CSV there."""
    src = _safe_path(file_path)
    report, clean = analyze_dataframe(load_dataframe(src), task, target, name=src.name)
    out = None
    if output_path:
        out = _safe_path(output_path)
        clean.to_csv(out, index=False)
    return {"report": report.model_dump(mode="json"), "clean_path": str(out) if out else None}


if __name__ == "__main__":
    mcp.run()
