"""Validation tools: recompute every claim with separate numpy code and compare."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ..models.findings import ActionType, IssueType, PreparationStep, QualityFinding, ValidationResult

TOL = 0.011  # absolute tolerance for rounded evidence values


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(float(a) - float(b)) <= max(tol, 1e-3 * abs(float(b)))


def verify_finding(df: pd.DataFrame, f: QualityFinding) -> ValidationResult:
    """Recompute the finding's key number from the ORIGINAL data."""
    e, subject = f.evidence, f"finding:{f.key}"
    try:
        if f.issue_type == IssueType.missing_values:
            actual = int(len(df) - df[f.column].count())   # count() = non-null; independent of isna().sum()
            ok = actual == e["missing_count"] and _close(100 * actual / len(df), e["missing_pct"])
            return ValidationResult(subject=subject, passed=ok, claimed=e["missing_count"], actual=actual)

        if f.issue_type == IssueType.outliers:
            x = df[f.column].dropna().to_numpy(dtype=float)
            actual = int(np.sum((x < e["lower_bound"] - 1e-3) | (x > e["upper_bound"] + 1e-3)))
            # fences must equal 3 IQR: lower from the raw scale; upper from raw or log1p scale (as stated)
            q1, q3 = np.percentile(x, [25, 75])
            lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
            if e["method"] == "iqr_log1p":
                lq1, lq3 = np.percentile(np.log1p(x), [25, 75])
                hi = np.expm1(lq3 + 3 * (lq3 - lq1))
            bounds_ok = _close(lo, e["lower_bound"], 0.02) and _close(hi, e["upper_bound"], 0.02)
            # rounding of bounds to 4 d.p. can move a borderline point; allow +-1
            ok = bounds_ok and abs(actual - e["outlier_count"]) <= 1
            return ValidationResult(subject=subject, passed=ok, claimed=e["outlier_count"], actual=actual,
                                    detail="" if bounds_ok else "fences do not match 3xIQR")

        if f.issue_type == IssueType.imbalance:
            y = df[e["target"] if "target" in e else f.column]
            counts = {str(k): int(v) for k, v in y.dropna().astype(str).value_counts().items()}
            claimed = {str(k): int(v) for k, v in e["class_counts"].items()}
            return ValidationResult(subject=subject, passed=counts == claimed, claimed=claimed, actual=counts)

        if f.issue_type == IssueType.duplicates:
            actual = len(df) - len(df.drop_duplicates())
            return ValidationResult(subject=subject, passed=actual == e["duplicate_count"],
                                    claimed=e["duplicate_count"], actual=actual)

        if f.issue_type == IssueType.constant_column:
            actual = df[f.column].dropna().nunique()
            return ValidationResult(subject=subject, passed=actual <= 1, claimed=1, actual=int(actual))

        if f.issue_type == IssueType.multicollinearity:
            a, b = df[e["column_a"]], df[e["column_b"]]
            m = a.notna() & b.notna()
            r = float(np.corrcoef(a[m].astype(float), b[m].astype(float))[0, 1])
            return ValidationResult(subject=subject, passed=_close(r, e["correlation"], 0.002),
                                    claimed=e["correlation"], actual=round(r, 4))

        if f.issue_type == IssueType.leakage:
            return _verify_leakage(df, f)
    except Exception as ex:  # a claim about a missing column etc. is simply unverifiable -> rejected
        return ValidationResult(subject=subject, passed=False, detail=f"verification error: {ex}")
    return ValidationResult(subject=subject, passed=False, detail="no verifier for this issue type")


def _verify_leakage(df: pd.DataFrame, f: QualityFinding) -> ValidationResult:
    e = f.evidence
    x, y = df[f.column], df[e["target"]]
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    metric = e["metric"]
    if metric == "pearson_abs":
        actual = abs(float(np.corrcoef(x.astype(float), y.astype(float))[0, 1]))
    elif metric == "correlation_ratio":
        xv = x.to_numpy(dtype=float)
        grand = xv.mean()
        ssb = 0.0
        for cls in np.unique(y.astype(str)):
            g = xv[(y.astype(str) == cls).to_numpy()]
            ssb += len(g) * (g.mean() - grand) ** 2
        actual = float(np.sqrt(ssb / ((xv - grand) ** 2).sum()))
    elif metric == "purity":
        hits = 0
        for cat in x.astype(str).unique():
            hits += y[(x.astype(str) == cat).to_numpy()].astype(str).value_counts().iloc[0]
        actual = hits / len(x)
    else:
        return ValidationResult(subject=f"finding:{f.key}", passed=False, detail=f"unknown metric {metric}")
    return ValidationResult(subject=f"finding:{f.key}", passed=_close(actual, e["value"], 0.002),
                            claimed=e["value"], actual=round(float(actual), 4))


# ---------------- step + invariant verification (against the cleaned data) ----------------

def verify_steps(original: pd.DataFrame, clean: pd.DataFrame, steps: list[PreparationStep],
                 target: str | None) -> list[ValidationResult]:
    out: list[ValidationResult] = []
    for s in steps:
        subject = f"step:{s.finding_key}:{s.action.value}"
        col = s.column
        if s.action == ActionType.drop_duplicates:
            n = int(clean.duplicated().sum())
            out.append(ValidationResult(subject=subject, passed=n == 0, claimed=0, actual=n))
        elif s.action == ActionType.drop_column:
            out.append(ValidationResult(subject=subject, passed=col not in clean.columns,
                                        claimed="absent", actual="present" if col in clean.columns else "absent"))
        elif col not in clean.columns:
            out.append(ValidationResult(subject=subject, passed=True, detail="column dropped by a later step"))
        elif s.action in (ActionType.impute_median, ActionType.impute_mode, ActionType.drop_rows_missing):
            n = int(clean[col].isna().sum())
            out.append(ValidationResult(subject=subject, passed=n == 0, claimed=0, actual=n))
        elif s.action in (ActionType.clip_outliers, ActionType.remove_outlier_rows):
            x = clean[col].dropna().to_numpy(dtype=float)
            lo, hi = s.parameters["lower"], s.parameters["upper"]
            bad = int(np.sum((x < lo - 1e-3) | (x > hi + 1e-3)))
            out.append(ValidationResult(subject=subject, passed=bad == 0, claimed=0, actual=bad))

    # global invariants: a "fix" must not break something else
    out.append(ValidationResult(subject="invariant:target_kept", passed=(target is None or target in clean.columns),
                                detail="target column must survive preparation"))
    out.append(ValidationResult(subject="invariant:row_floor", passed=len(clean) >= 0.5 * len(original),
                                claimed=">=50% rows", actual=f"{len(clean)}/{len(original)}"))
    worse = [c for c in clean.columns if c in original.columns and clean[c].isna().sum() > original[c].isna().sum()]
    out.append(ValidationResult(subject="invariant:no_new_missing", passed=not worse, detail=", ".join(worse)))
    return out


# ---------------- numeric grounding of LLM-written text ----------------

_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_STRUCTURAL = {0.0, 1.0, 1.5, 2.0, 3.0}   # "1 of 2", "1.5 IQR", etc.


def _flatten(obj: Any) -> list[float]:
    nums: list[float] = []
    if isinstance(obj, bool):
        return nums
    if isinstance(obj, (int, float)):
        nums.append(float(obj))
    elif isinstance(obj, str):
        try:
            nums.append(float(obj))
        except ValueError:
            pass
    elif isinstance(obj, dict):
        for k, v in obj.items():
            nums += _flatten(k) + _flatten(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            nums += _flatten(v)
    return nums


_PCT = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*%")


def _pct_values(evidence: dict) -> list[float]:
    return [float(v) for k, v in evidence.items() if "pct" in k and isinstance(v, (int, float))]


def _matches(v: float, pool: list[float]) -> bool:
    return any(abs(v - a) <= max(0.051, 0.01 * abs(a)) for a in pool)


def check_numeric_grounding(text: str, f: QualityFinding) -> ValidationResult:
    """Every number the LLM writes must appear in the finding's computed evidence.
    A number followed by '%' must match a *_pct field (so 'a count of 30' can't pass as '30%')."""
    scrub = text
    names = [f.column or ""] + str(f.column or "").split("~") + [str(v) for v in f.evidence.values() if isinstance(v, str)]
    for n in sorted({n for n in names if n}, key=len, reverse=True):
        scrub = scrub.replace(n, " ")
    allowed = _flatten(f.evidence)
    pct_allowed = _pct_values(f.evidence)
    ungrounded = []
    for m in _PCT.findall(scrub):
        if not _matches(float(m), pct_allowed):
            ungrounded.append(m + "%")
    bare = _PCT.sub(" ", scrub)
    for m in _NUM.findall(bare):
        v = float(m)
        if v in _STRUCTURAL:
            continue
        if not _matches(v, allowed):
            ungrounded.append(m)
    return ValidationResult(subject=f"grounding:{f.key}", passed=not ungrounded,
                            claimed=ungrounded or None, detail="numbers not found in evidence" if ungrounded else "")
