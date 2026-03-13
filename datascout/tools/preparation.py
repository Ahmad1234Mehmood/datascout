"""Preparation tools: execute an LLM-proposed PrepPlan safely (class imbalance is advisory only)."""
from __future__ import annotations

import pandas as pd

from ..models.findings import ActionType, IssueType, PrepAction, PreparationStep, QualityFinding
from .profiling import compute_stats, is_numeric_feature
from .quality import outlier_bounds

# Canonical execution order. Deduplication runs last because imputing/clipping can create new duplicates.
ORDER = {
    ActionType.drop_column: 0,
    ActionType.impute_median: 1, ActionType.impute_mode: 1, ActionType.drop_rows_missing: 1,
    ActionType.clip_outliers: 2, ActionType.remove_outlier_rows: 2,
    ActionType.drop_duplicates: 3,
}

ACTION_ISSUES = {
    ActionType.impute_median: {IssueType.missing_values},
    ActionType.impute_mode: {IssueType.missing_values},
    ActionType.drop_rows_missing: {IssueType.missing_values},
    ActionType.clip_outliers: {IssueType.outliers},
    ActionType.remove_outlier_rows: {IssueType.outliers},
    ActionType.drop_duplicates: {IssueType.duplicates},
    ActionType.drop_column: {IssueType.leakage, IssueType.constant_column,
                             IssueType.multicollinearity, IssueType.missing_values},
}


class PreparationError(ValueError):
    pass


class SkipAction(Exception):
    """Action is moot (e.g. column already dropped). Not an error."""


def derive_finding_key(action: PrepAction, findings: list[QualityFinding]) -> str:
    if action.finding_key and action.finding_key in {f.key for f in findings}:
        return action.finding_key           # an LLM-invented key is ignored and re-derived below
    wanted = ACTION_ISSUES[action.action]
    for f in findings:
        if f.issue_type not in wanted:
            continue
        if f.issue_type == IssueType.duplicates and action.action == ActionType.drop_duplicates:
            return f.key
        if f.column == action.column or (f.issue_type == IssueType.multicollinearity
                                         and action.column in str(f.column).split("~")):
            return f.key
    return f"{next(iter(wanted)).value}:{action.column}"


def apply_action(df: pd.DataFrame, action: PrepAction, target: str | None, finding_key: str) -> tuple[pd.DataFrame, PreparationStep]:
    a, col = action.action, action.column
    step = PreparationStep(finding_key=finding_key, action=a, column=col, rationale=action.rationale)

    if a == ActionType.drop_duplicates:
        step.before_stats = {"rows": len(df), "duplicate_rows": int(df.duplicated().sum())}
        out = df.drop_duplicates().reset_index(drop=True)
        step.after_stats = {"rows": len(out), "duplicate_rows": int(out.duplicated().sum())}
        step.parameters = {"rows_removed": len(df) - len(out)}
        return out, step

    if not col:
        raise PreparationError(f"{a.value} requires a column")
    if col not in df.columns:
        raise SkipAction(f"column '{col}' no longer exists")

    if a == ActionType.drop_column:
        if col == target:
            raise PreparationError("refusing to drop the target column")
        step.before_stats = {"columns": df.shape[1]}
        out = df.drop(columns=[col])
        step.after_stats = {"columns": out.shape[1]}
        return out, step

    step.before_stats = compute_stats(df, col)
    s = df[col]

    if a in (ActionType.impute_median, ActionType.impute_mode):
        if col == target:
            raise PreparationError("refusing to impute the target column; drop rows instead")
        if s.notna().sum() == 0:
            raise PreparationError(f"'{col}' has no observed values to impute from")
        if a == ActionType.impute_median:
            if not is_numeric_feature(s):
                raise PreparationError(f"median imputation needs a numeric column, '{col}' is {s.dtype}")
            value = float(s.median())
        else:
            value = s.mode(dropna=True).iloc[0]
            value = value.item() if hasattr(value, "item") else value
        out = df.copy()
        out[col] = out[col].fillna(value)
        step.parameters = {"value": value}

    elif a == ActionType.drop_rows_missing:
        out = df[df[col].notna()].reset_index(drop=True)
        step.parameters = {"rows_removed": len(df) - len(out)}

    elif a in (ActionType.clip_outliers, ActionType.remove_outlier_rows):
        if not is_numeric_feature(s):
            raise PreparationError(f"outlier handling needs a numeric column, '{col}' is {s.dtype}")
        b = outlier_bounds(s)
        if b is None:
            raise SkipAction(f"no outlier fences computable for '{col}'")
        lo, hi, method = b
        if a == ActionType.clip_outliers:
            out = df.copy()
            out[col] = out[col].clip(lower=lo, upper=hi)
        else:
            keep = s.isna() | ((s >= lo) & (s <= hi))
            out = df[keep].reset_index(drop=True)
        step.parameters = {"lower": round(lo, 4), "upper": round(hi, 4), "method": method}
        if a == ActionType.remove_outlier_rows:
            step.parameters["rows_removed"] = len(df) - len(out)
    else:  # pragma: no cover
        raise PreparationError(f"unknown action {a}")

    step.after_stats = compute_stats(out, col)
    return out, step


def run_plan(df: pd.DataFrame, actions: list[PrepAction], target: str | None,
             findings: list[QualityFinding]) -> tuple[pd.DataFrame, list[PreparationStep], list[str]]:
    """Execute actions in canonical order. Returns (new df, steps, errors/skips)."""
    steps: list[PreparationStep] = []
    notes: list[str] = []
    seen: set[tuple] = set()
    for action in sorted(actions, key=lambda x: ORDER[x.action]):
        sig = (action.action, action.column)
        if sig in seen:
            continue
        seen.add(sig)
        key = derive_finding_key(action, findings)
        try:
            df, step = apply_action(df, action, target, key)
            steps.append(step)
        except SkipAction as e:
            notes.append(f"skipped {action.action.value}({action.column}): {e}")
        except PreparationError as e:
            notes.append(f"error {action.action.value}({action.column}): {e}")
    return df, steps, notes
