"""Quality-check tools: deterministic detection returning findings with numeric evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.findings import IssueType, QualityFinding, Severity
from .profiling import detect_semantic_type, infer_task_type, is_numeric_feature

FAR_IQR = 3.0              # outlier fence multiplier (far outliers only)
LOG_SCALE_MARGIN = 0.05    # log1p scale is used only if it makes the central 50% clearly more symmetric
MISSING_FLAG_PCT = 1.0
IMBALANCE_FLAG_PCT = 20.0
# Leakage = (near-)deterministic: 0.99 <=> R^2 >= 0.98; lower values flag strong legitimate predictors.
LEAK_ETA = 0.99
LEAK_PURITY = 0.99
LEAK_REGRESSION_R = 0.99
COLLINEAR_R = 0.95
DUP_FLAG_PCT = 1.0


def _quartile_skew(v: pd.Series) -> float:
    """Bowley skewness from quartiles only: robust, so outliers cannot influence it."""
    q1, med, q3 = v.quantile(0.25), v.quantile(0.5), v.quantile(0.75)
    return float((q3 + q1 - 2 * med) / (q3 - q1)) if q3 > q1 else 0.0


def outlier_bounds(x: pd.Series, k: float = FAR_IQR) -> tuple[float, float, str] | None:
    """Far-outlier fences in original units; the scale is chosen from the central 50% so outliers cannot bias it."""
    x = x.dropna().astype(float)
    if len(x) < 20 or x.nunique() <= 10:
        return None
    rq1, rq3 = x.quantile(0.25), x.quantile(0.75)
    riqr = rq3 - rq1
    if riqr == 0:
        return None
    lo, hi, method = float(rq1 - k * riqr), float(rq3 + k * riqr), "iqr"
    if x.min() >= 0:
        lx = np.log1p(x)
        if abs(_quartile_skew(lx)) < abs(_quartile_skew(x)) - LOG_SCALE_MARGIN:
            lq1, lq3 = lx.quantile(0.25), lx.quantile(0.75)
            # log scale only for the upper fence: its lower fence is too tight and flags ordinary values
            hi, method = float(np.expm1(lq3 + k * (lq3 - lq1))), "iqr_log1p"
    return lo, hi, method


def _sev_by_pct(pct: float, cuts: tuple[float, float, float]) -> Severity:
    low, high, crit = cuts
    if pct >= crit:
        return Severity.critical
    if pct >= high:
        return Severity.high
    if pct >= low:
        return Severity.medium
    return Severity.low


def check_missing(df: pd.DataFrame, column: str, **_) -> list[QualityFinding]:
    miss = int(df[column].isna().sum())
    pct = 100 * miss / len(df)
    if pct < MISSING_FLAG_PCT:
        return []
    sev = Severity.critical if pct >= 40 else Severity.high if pct >= 20 else Severity.medium if pct >= 5 else Severity.low
    return [QualityFinding(column=column, issue_type=IssueType.missing_values, severity=sev,
                           evidence={"missing_count": miss, "missing_pct": round(pct, 2), "total_rows": len(df)})]


def detect_outliers(df: pd.DataFrame, column: str, **_) -> list[QualityFinding]:
    s = df[column]
    if not is_numeric_feature(s):
        return []
    b = outlier_bounds(s)
    if b is None:
        return []
    lo, hi, method = b
    x = s.dropna().astype(float)
    n = int(((x < lo) | (x > hi)).sum())
    if n == 0:
        return []
    pct = 100 * n / len(x)
    sev = Severity.high if pct >= 5 else Severity.medium if pct >= 1 else Severity.low
    return [QualityFinding(column=column, issue_type=IssueType.outliers, severity=sev,
                           evidence={"outlier_count": n, "outlier_pct": round(pct, 2),
                                     "lower_bound": round(lo, 4), "upper_bound": round(hi, 4),
                                     "method": method, "observed_min": round(float(x.min()), 4),
                                     "observed_max": round(float(x.max()), 4)})]


def check_imbalance(df: pd.DataFrame, column: str | None = None, target: str | None = None, **_) -> list[QualityFinding]:
    target = target or column
    if not target or target not in df or infer_task_type(df, target) != "classification":
        return []
    counts = df[target].value_counts()
    if len(counts) < 2:
        return []
    minority_pct = 100 * counts.min() / counts.sum()
    if minority_pct >= IMBALANCE_FLAG_PCT:
        return []
    sev = Severity.critical if minority_pct < 5 else Severity.high if minority_pct < 10 else Severity.medium
    return [QualityFinding(column=target, issue_type=IssueType.imbalance, severity=sev,
                           evidence={"minority_pct": round(float(minority_pct), 2),
                                     "class_counts": {str(k): int(v) for k, v in counts.items()},
                                     "imbalance_ratio": round(float(counts.max() / counts.min()), 2)})]


def _correlation_ratio(values: pd.Series, labels: pd.Series) -> float:
    """eta: share of feature variance explained by class membership (0..1)."""
    d = pd.DataFrame({"v": values, "y": labels}).dropna()
    if d["v"].var() == 0 or d["y"].nunique() < 2:
        return 0.0
    grand = d["v"].mean()
    ssb = d.groupby("y")["v"].agg(lambda g: len(g) * (g.mean() - grand) ** 2).sum()
    sst = ((d["v"] - grand) ** 2).sum()
    return float(np.sqrt(ssb / sst)) if sst > 0 else 0.0


def _purity(feature: pd.Series, labels: pd.Series) -> tuple[float, float]:
    d = pd.DataFrame({"f": feature, "y": labels}).dropna()
    if d.empty:
        return 0.0, 0.0
    per_cat_max = d.groupby("f")["y"].agg(lambda g: g.value_counts().iloc[0]).sum()
    baseline = d["y"].value_counts().iloc[0] / len(d)
    return float(per_cat_max / len(d)), float(baseline)


def check_leakage(df: pd.DataFrame, column: str | None = None, target: str | None = None, **_) -> list[QualityFinding]:
    """Flag features that (almost) determine the target on their own."""
    target = target or column
    if not target or target not in df:
        return []
    task = infer_task_type(df, target)
    y = df[target]
    out: list[QualityFinding] = []
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        sem = detect_semantic_type(s)
        if sem in ("id", "constant", "empty", "text", "datetime"):
            continue
        metric = value = None
        if is_numeric_feature(s):
            if task == "classification":
                metric, value = "correlation_ratio", _correlation_ratio(s, y)
                hit = value >= LEAK_ETA
            else:
                d = pd.DataFrame({"a": s, "b": y}).dropna()
                value = abs(float(np.corrcoef(d["a"], d["b"])[0, 1])) if len(d) > 2 and d["a"].std() > 0 else 0.0
                metric, hit = "pearson_abs", value >= LEAK_REGRESSION_R
        elif task == "classification" and sem in ("categorical", "boolean") and s.nunique() / max(s.notna().sum(), 1) < 0.5:
            purity, baseline = _purity(s, y)
            metric, value = "purity", purity
            hit = purity >= LEAK_PURITY and (purity - baseline) >= 0.04
        else:
            continue
        if hit:
            out.append(QualityFinding(column=col, issue_type=IssueType.leakage, severity=Severity.critical,
                                      evidence={"target": target, "metric": metric, "value": round(float(value), 4)}))
    return out


def check_correlation(df: pd.DataFrame, target: str | None = None, **_) -> list[QualityFinding]:
    cols = [c for c in df.columns if c != target and is_numeric_feature(df[c])
            and detect_semantic_type(df[c]) == "numeric"]
    if len(cols) < 2:
        return []
    corr = df[cols].corr()
    out: list[QualityFinding] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= COLLINEAR_R:
                out.append(QualityFinding(column=f"{a}~{b}", issue_type=IssueType.multicollinearity,
                                          severity=Severity.high if abs(r) >= 0.99 else Severity.medium,
                                          evidence={"column_a": a, "column_b": b, "correlation": round(float(r), 4)}))
    return out


def check_duplicates(df: pd.DataFrame, **_) -> list[QualityFinding]:
    n = int(df.duplicated().sum())
    pct = 100 * n / len(df)
    if pct < DUP_FLAG_PCT:
        return []
    sev = Severity.high if pct >= 10 else Severity.medium if pct >= 3 else Severity.low
    return [QualityFinding(column=None, issue_type=IssueType.duplicates, severity=sev,
                           evidence={"duplicate_count": n, "duplicate_pct": round(pct, 2), "total_rows": len(df)})]


def check_constant(df: pd.DataFrame, column: str, **_) -> list[QualityFinding]:
    s = df[column]
    if len(s) and s.notna().any() and s.nunique(dropna=True) <= 1:
        return [QualityFinding(column=column, issue_type=IssueType.constant_column, severity=Severity.medium,
                               evidence={"unique_count": int(s.nunique(dropna=True)), "value": str(s.dropna().iloc[0])})]
    return []


# name -> (function, scope). Scope tells the executor how to call it.
CHECKS = {
    "check_missing": (check_missing, "column"),
    "detect_outliers": (detect_outliers, "column"),
    "check_constant": (check_constant, "column"),
    "check_imbalance": (check_imbalance, "target"),
    "check_leakage": (check_leakage, "target"),
    "check_correlation": (check_correlation, "dataset"),
    "check_duplicates": (check_duplicates, "dataset"),
}

SEVERITY_PENALTY = {Severity.low: 3, Severity.medium: 8, Severity.high: 15, Severity.critical: 25}


def quality_score(findings: list[QualityFinding]) -> float:
    return float(max(0, 100 - sum(SEVERITY_PENALTY[f.severity] for f in findings)))
