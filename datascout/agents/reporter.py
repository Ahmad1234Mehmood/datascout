from __future__ import annotations

from ..models.findings import Severity
from ..models.report import QualityReport, TraceEvent
from ..tools.quality import quality_score
from .quality_checker import run_checks


def reporter_node(state: dict, provider: str) -> dict:
    orig = state["original_df"]
    clean = state.get("clean_df", orig)
    findings = state.get("findings", [])
    steps = state.get("steps", [])
    target = state.get("target")
    errors = list(state.get("errors", []))
    needs_review = bool(state.get("needs_review")) or bool(state.get("fatal"))

    # "After" score: re-run the same checks on the cleaned data (imbalance is advisory, so it stays).
    residual = []
    if state.get("plan") is not None and not state.get("fatal"):
        residual, _ = run_checks(clean, state["plan"], target, skip_missing_columns=True)

    counts = {s: sum(f.severity == s for f in findings) for s in Severity}
    validations = state.get("validations", [])
    verified = sum(v.passed for v in validations if v.subject.startswith("step:"))
    total_steps = sum(v.subject.startswith("step:") for v in validations)
    if not findings and not errors:
        summary = "No data quality issues detected."
    else:
        summary = (f"{len(findings)} issue(s) found ({counts[Severity.critical]} critical, {counts[Severity.high]} high, "
                   f"{counts[Severity.medium]} medium, {counts[Severity.low]} low). "
                   f"{len(steps)} fix(es) applied, {verified}/{total_steps} step checks passed after "
                   f"{state.get('retry_count', 0)} retr{'y' if state.get('retry_count', 0) == 1 else 'ies'}.")
    if needs_review:
        summary += " Flagged for human review."
    if errors:
        summary += " Run ended early: " + "; ".join(errors)

    report = QualityReport(
        dataset_name=state.get("dataset_name", "dataset"), task_description=state["task"], target=target,
        llm_provider=provider,
        n_rows_before=len(orig), n_cols_before=orig.shape[1], n_rows_after=len(clean), n_cols_after=clean.shape[1],
        profiles=state.get("profiles", []), findings=findings, preparation_steps=steps, validations=validations,
        residual_findings=residual,
        quality_score_before=quality_score(findings), quality_score_after=quality_score(residual),
        retries=state.get("retry_count", 0), needs_human_review=needs_review,
        unresolved=state.get("unresolved", []) + errors, warnings=state.get("warnings", []),
        summary=summary, trace=[TraceEvent(**t) for t in state.get("trace", [])],
    )
    return {"report": report, "clean_df": clean}
