"""Deterministic recommendation text, also used to repair LLM advice that quotes unsupported numbers."""
from __future__ import annotations

from .models.findings import IssueType, QualityFinding


def template_recommendation(f: QualityFinding) -> str:
    e = f.evidence
    t = f.issue_type
    if t == IssueType.missing_values:
        return (f"{e['missing_pct']}% of values are missing ({e['missing_count']} of {e['total_rows']} rows). "
                "Impute with the median for numeric columns or the most frequent value for categorical ones; "
                "drop the column if most of it is missing.")
    if t == IssueType.outliers:
        return (f"{e['outlier_count']} values ({e['outlier_pct']}%) fall outside [{e['lower_bound']}, {e['upper_bound']}]. "
                "Clip them to these bounds, or remove the rows if they are data-entry errors.")
    if t == IssueType.leakage:
        return (f"'{f.column}' predicts the target '{e['target']}' almost perfectly ({e['metric']}={e['value']}). "
                "It probably encodes the outcome or is only known after it. Drop it before training.")
    if t == IssueType.imbalance:
        return (f"The minority class is {e['minority_pct']}% of rows (imbalance ratio {e['imbalance_ratio']}). "
                "Do not resample the full dataset: use a stratified split plus class weights, "
                "or resample only the training fold.")
    if t == IssueType.duplicates:
        return (f"{e['duplicate_count']} rows ({e['duplicate_pct']}% of {e['total_rows']}) are exact duplicates. "
                "Drop them so they cannot leak across the train/test split.")
    if t == IssueType.multicollinearity:
        return (f"'{e['column_a']}' and '{e['column_b']}' are almost redundant (correlation {e['correlation']}). "
                "Drop one of them, keeping the one with fewer missing values.")
    if t == IssueType.constant_column:
        return f"'{f.column}' has a single value ({e['value']}) and carries no information. Drop it."
    return "Review this finding manually."
