from __future__ import annotations

import difflib

import pandas as pd

from ..llm import HeuristicClient, LLMClient
from ..models.findings import CheckPlan, QualityFinding, Severity
from ..tools.quality import CHECKS

SEV_ORDER = {Severity.critical: 0, Severity.high: 1, Severity.medium: 2, Severity.low: 3}


def run_checks(df: pd.DataFrame, plan: CheckPlan, target: str | None, skip_missing_columns: bool = False
               ) -> tuple[list[QualityFinding], list[str]]:
    """Execute the planned tool calls; a failing call is reported and the run continues."""
    findings: dict[str, QualityFinding] = {}
    errors: list[str] = []
    done: set[tuple] = set()
    for call in plan.calls:
        if call.tool not in CHECKS:
            errors.append(f"unknown tool '{call.tool}' ignored")
            continue
        fn, scope = CHECKS[call.tool]
        col = call.column
        if scope == "column":
            if col not in df.columns:
                close = difflib.get_close_matches(col or "", list(df.columns), n=1, cutoff=0.8)
                if close:
                    errors.append(f"{call.tool}: column '{col}' not found, using '{close[0]}'")
                    col = close[0]
                else:
                    if not skip_missing_columns:
                        errors.append(f"{call.tool}: column '{col}' not found, skipped")
                    continue
        if (call.tool, col) in done:
            continue
        done.add((call.tool, col))
        try:
            for f in fn(df, column=col, target=target):
                findings.setdefault(f.key, f)
        except Exception as e:
            errors.append(f"{call.tool}({col}) failed: {type(e).__name__}: {e}")
    return sorted(findings.values(), key=lambda f: (SEV_ORDER[f.severity], f.key)), errors


def _ensure_coverage(plan: CheckPlan, profiles, task: str, target: str | None) -> tuple[CheckPlan, list[str]]:
    """Add every check the rule-based plan would run that the model skipped, and report the omission."""
    have = {(c.tool, c.column) for c in plan.calls}
    omitted = [c for c in HeuristicClient().plan_checks(profiles, task, target).calls if (c.tool, c.column) not in have]
    if not omitted:
        return plan, []
    sample = ", ".join(f"{c.tool}:{c.column}" for c in omitted[:4]) + (" ..." if len(omitted) > 4 else "")
    note = f"coverage: the model's check plan omitted {len(omitted)} applicable check(s) ({sample}); added automatically"
    return CheckPlan(calls=plan.calls + omitted), [note]


def quality_node(state: dict, llm: LLMClient) -> dict:
    df, target = state["original_df"], state.get("target")
    plan = llm.plan_checks(state["profiles"], state["task"], target)
    plan, coverage_note = _ensure_coverage(plan, state["profiles"], state["task"], target)
    findings, errors = run_checks(df, plan, target)
    recs = llm.recommend_many(findings, state["task"])
    for f in findings:
        f.recommendation = recs.get(f.key, "")
    warnings = list(state.get("warnings", [])) + errors + llm.pop_events() + coverage_note
    return {"plan": plan, "findings": findings, "warnings": warnings,
            "_note": f"{len(plan.calls)} checks planned, {len(findings)} findings"}
