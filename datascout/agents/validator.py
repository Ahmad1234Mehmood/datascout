from __future__ import annotations

from collections import defaultdict

from ..llm import LLMClient
from ..models.findings import IssueType, QualityFinding, ValidationResult
from ..recommendations import template_recommendation
from ..tools.validation import check_numeric_grounding, verify_finding, verify_steps


def _key_of(subject: str) -> str:
    # "step:missing_values:age:impute_median" -> "missing_values:age"
    return subject.split(":", 1)[1].rsplit(":", 1)[0]


def _resolved_by_removal(f: QualityFinding, clean) -> bool:
    if f.issue_type == IssueType.multicollinearity:
        return f.evidence["column_a"] not in clean.columns or f.evidence["column_b"] not in clean.columns
    return f.column is not None and f.column not in clean.columns


def validator_node(state: dict, llm: LLMClient) -> dict:
    """Judge: nothing the agents claim reaches the report unless it is re-computed and confirmed."""
    orig, clean, target = state["original_df"], state["clean_df"], state.get("target")
    findings, steps = list(state["findings"]), list(state.get("steps", []))
    warnings = list(state.get("warnings", []))
    unresolved = list(state.get("unresolved", []))
    validations: list[ValidationResult] = []
    needs_review = False

    # 1) every finding is recomputed independently from the original data
    rejected = set()
    for f in findings:
        r = verify_finding(orig, f)
        validations.append(r)
        if not r.passed:
            rejected.add(f.key)
    if rejected:
        findings = [f for f in findings if f.key not in rejected]
        steps = [s for s in steps if s.finding_key not in rejected]
        unresolved += [f"finding {k} rejected: claimed evidence does not match the data" for k in sorted(rejected)]
        warnings.append(f"{len(rejected)} finding(s) rejected by the validator")
        needs_review = True

    # 2) numbers in LLM-written advice must come from the evidence; otherwise repair with a template
    for f in findings:
        g = check_numeric_grounding(f.recommendation, f)
        if not g.passed:
            f.recommendation = template_recommendation(f)
            g.detail = "recommendation quoted numbers not in the evidence; replaced with a deterministic template"
        validations.append(g)

    # 3) steps verified against the cleaned data, plus global invariants
    step_vals = verify_steps(orig, clean, steps, target)
    validations += step_vals

    # 4) LLM judge: is each chosen fix appropriate for the finding and the task?
    judgements = llm.judge_steps(findings, steps, state["task"])
    for i, s in enumerate(steps):
        j = judgements.get(i)
        validations.append(ValidationResult(subject=f"judge:{s.finding_key}:{s.action.value}",
                                            passed=True if j is None else j.appropriate,
                                            detail="" if j is None else j.reason))

    # 5) coverage: every fixable finding must be resolved by a verified step or by the column being dropped
    blocking: dict[str, list[str]] = defaultdict(list)
    judge_only: dict[str, bool] = {}
    step_keys = {s.finding_key for s in steps}
    for f in findings:
        if f.issue_type == IssueType.imbalance:
            continue                                   # advisory by design
        if f.key not in step_keys and not _resolved_by_removal(f, clean):
            blocking[f.key].append("no fix was applied")
    for v in validations:
        if v.passed or v.subject.startswith(("finding:", "grounding:")):
            continue
        if v.subject.startswith(("step:", "judge:")):
            k = _key_of(v.subject)
            blocking[k].append(v.detail or f"{v.subject} failed (claimed {v.claimed}, actual {v.actual})")
            judge_only[k] = judge_only.get(k, True) and v.subject.startswith("judge:")
        elif v.subject.startswith("invariant:"):
            for k in step_keys:
                blocking[k].append(f"{v.subject} violated {v.detail}".strip())

    retry, max_retry = state.get("retry_count", 0), state.get("max_retries", 2)
    levels, feedback, route = dict(state.get("levels", {})), [], "report"

    # A judge-only objection gets one retry; after that keep the verified fix and flag the run for review.
    for k in [k for k in blocking if judge_only.get(k) and levels.get(k, 0) >= 1]:
        unresolved.append(f"{k}: judge disagrees after a retry, verified fix kept ({'; '.join(sorted(set(blocking.pop(k))))})")
        warnings.append(f"judge disagreement on {k}: fix kept, flagged for review")
        needs_review = True

    if blocking:
        if retry < max_retry:
            route, retry = "retry", retry + 1
            for k, reasons in blocking.items():
                levels[k] = levels.get(k, 0) + 1
                feedback.append(f"{k}: {'; '.join(sorted(set(reasons)))}")
        else:
            needs_review = True
            unresolved += [f"{k}: {'; '.join(sorted(set(r)))}" for k, r in blocking.items()]
            warnings.append("retry budget exhausted: flagged for human review")

    return {"findings": findings, "steps": steps, "validations": validations, "route": route,
            "retry_count": retry, "levels": levels, "feedback": feedback, "needs_review": needs_review,
            "unresolved": unresolved, "warnings": warnings,
            "_note": f"{sum(not v.passed for v in validations)} failed checks -> {route}"}
