"""Distribution drift: a statistical test plus an effect-size floor per column, Bonferroni-split."""
from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from ..models.drift import DriftFinding, DriftReport
from ..models.findings import Severity
from .profiling import detect_semantic_type

KS_MIN_D = 0.10
TVD_MIN = 0.10
ALPHA = 0.01
MIN_ROWS = 30
MAX_CATEGORIES = 50


def _py(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def _severity(stat: float) -> Severity:
    if stat >= 0.5:
        return Severity.critical
    if stat >= 0.3:
        return Severity.high
    if stat >= 0.2:
        return Severity.medium
    return Severity.low


def _numeric_finding(col: str, a: pd.Series, b: pd.Series, alpha: float) -> DriftFinding | None:
    x, y = a.dropna().astype(float).to_numpy(), b.dropna().astype(float).to_numpy()
    if len(x) < MIN_ROWS or len(y) < MIN_ROWS:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # scipy falls back to the asymptotic p-value on big samples
        res = stats.ks_2samp(x, y)
    if res.statistic < KS_MIN_D or res.pvalue >= alpha:
        return None
    ev = {"reference_mean": _py(x.mean()), "current_mean": _py(y.mean()),
          "reference_median": _py(np.median(x)), "current_median": _py(np.median(y)),
          "reference_std": _py(x.std(ddof=1)), "current_std": _py(y.std(ddof=1))}
    rec = (f"'{col}' shifted (KS D={res.statistic:.3f}, mean {ev['reference_mean']:.4g} -> {ev['current_mean']:.4g}). "
           "Check the data source and time window, then retrain or re-weight if the shift is real.")
    return DriftFinding(column=col, kind="numeric", test="ks_2samp", statistic_name="ks_d",
                        statistic=float(res.statistic), p_value=float(res.pvalue),
                        severity=_severity(float(res.statistic)), evidence=ev, recommendation=rec)


def _freqs(s: pd.Series) -> Counter:
    return Counter(s.dropna().astype(str))


def _tvd(ca: Counter, cb: Counter) -> float:
    na, nb = sum(ca.values()), sum(cb.values())
    keys = set(ca) | set(cb)
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in keys)


def _categorical_finding(col: str, a: pd.Series, b: pd.Series, alpha: float) -> DriftFinding | None:
    ca, cb = _freqs(a), _freqs(b)
    if sum(ca.values()) < MIN_ROWS or sum(cb.values()) < MIN_ROWS:
        return None
    keys = sorted(set(ca) | set(cb))
    table = np.array([[ca[k] for k in keys], [cb[k] for k in keys]])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return None
    _, p, _, _ = stats.chi2_contingency(table)
    d = _tvd(ca, cb)
    if d < TVD_MIN or p >= alpha:
        return None
    na, nb = sum(ca.values()), sum(cb.values())
    moves = sorted(((k, ca[k] / na, cb[k] / nb) for k in keys), key=lambda t: -abs(t[2] - t[1]))[:3]
    ev = {"largest_shifts": [{"category": k, "reference_pct": round(100 * r, 2), "current_pct": round(100 * c, 2)}
                             for k, r, c in moves],
          "new_categories": sorted(k for k in cb if k not in ca)[:10],
          "missing_categories": sorted(k for k in ca if k not in cb)[:10]}
    k, r, c = moves[0]
    rec = (f"'{col}' category mix changed (total variation distance {d:.3f}; '{k}' {100 * r:.1f}% -> {100 * c:.1f}%). "
           "Check the data source and time window, then retrain or re-weight if the shift is real.")
    return DriftFinding(column=col, kind="categorical", test="chi2_contingency", statistic_name="total_variation_distance",
                        statistic=float(d), p_value=float(p), severity=_severity(float(d)), evidence=ev, recommendation=rec)


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, target: str | None = None) -> DriftReport:
    report = DriftReport(n_reference_rows=len(reference), n_current_rows=len(current), target=target)
    common = [c for c in reference.columns if c in current.columns]
    for c in reference.columns:
        if c not in current.columns:
            report.columns_skipped[c] = "not in the current dataset"
    for c in current.columns:
        if c not in reference.columns:
            report.columns_skipped[c] = "not in the reference dataset"
    if len(reference) < MIN_ROWS or len(current) < MIN_ROWS:
        report.warnings.append(f"fewer than {MIN_ROWS} rows in one dataset: drift tests are not reliable, nothing tested")
        report.summary = "Not enough rows to test drift."
        report.verified = True
        return report

    def _testable(c: str) -> bool:
        ta, tb = detect_semantic_type(reference[c]), detect_semantic_type(current[c])
        if ta == "numeric" and tb == "numeric":
            return True
        return (ta in ("categorical", "boolean") and tb in ("categorical", "boolean")
                and max(reference[c].nunique(), current[c].nunique()) <= MAX_CATEGORIES)

    # Bonferroni: one test per column, so the significance level is split across them
    alpha = ALPHA / max(1, sum(_testable(c) for c in common))
    report.alpha_used = alpha

    for c in common:
        sa, sb = reference[c], current[c]
        ta, tb = detect_semantic_type(sa), detect_semantic_type(sb)
        if ta != tb and {ta, tb} != {"categorical", "boolean"}:
            report.columns_skipped[c] = f"type differs ({ta} vs {tb})"
            continue
        if ta == "numeric":
            f = _numeric_finding(c, sa, sb, alpha)
        elif ta in ("categorical", "boolean"):
            if max(sa.nunique(), sb.nunique()) > MAX_CATEGORIES:
                report.columns_skipped[c] = f"more than {MAX_CATEGORIES} categories"
                continue
            f = _categorical_finding(c, sa, sb, alpha)
        else:
            report.columns_skipped[c] = f"{ta} column"
            continue
        report.columns_compared.append(c)
        if f:
            report.findings.append(f)

    report.findings.sort(key=lambda f: -f.statistic)
    report.verified, report.verification_errors = verify_drift(reference, current, report.findings, alpha=alpha)
    n = len(report.findings)
    report.summary = (f"No drift detected in {len(report.columns_compared)} compared column(s)." if n == 0 else
                      f"Drift in {n} of {len(report.columns_compared)} compared column(s).")
    return report


# ---------------------------------------------------------------- independent verification

def _ks_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """KS statistic from the two empirical CDFs (no scipy)."""
    x, y = np.sort(x), np.sort(y)
    grid = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, grid, side="right") / len(x)
    cdf_y = np.searchsorted(y, grid, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def verify_drift(reference: pd.DataFrame, current: pd.DataFrame, findings: list[DriftFinding],
                 tol: float = 1e-9, alpha: float = ALPHA) -> tuple[bool, list[str]]:
    """Recompute every reported statistic and evidence value with separate code."""
    errors: list[str] = []
    for f in findings:
        a, b = reference[f.column], current[f.column]
        if f.kind == "numeric":
            x, y = a.dropna().astype(float).to_numpy(), b.dropna().astype(float).to_numpy()
            d = _ks_numpy(x, y)
            if abs(d - f.statistic) > tol:
                errors.append(f"{f.column}: KS D {f.statistic} != recomputed {d}")
            for key, val in (("reference_mean", x.mean()), ("current_mean", y.mean())):
                if abs(f.evidence[key] - val) > 1e-6 * max(1.0, abs(val)):
                    errors.append(f"{f.column}: {key} mismatch")
        else:
            d = _tvd(_freqs(a), _freqs(b))
            if abs(d - f.statistic) > tol:
                errors.append(f"{f.column}: TVD {f.statistic} != recomputed {d}")
        if not (f.statistic >= (KS_MIN_D if f.kind == "numeric" else TVD_MIN) and f.p_value < alpha):
            errors.append(f"{f.column}: finding does not meet its own thresholds")
    return (not errors), errors
