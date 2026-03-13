from __future__ import annotations

from typing import Any, Optional, TypedDict

import pandas as pd

from ..models.findings import CheckPlan, ColumnProfile, PreparationStep, QualityFinding, ValidationResult


class AgentState(TypedDict, total=False):
    # inputs
    dataset_name: str
    task: str
    target: Optional[str]
    original_df: pd.DataFrame
    # produced by nodes
    profiles: list[ColumnProfile]
    plan: CheckPlan
    findings: list[QualityFinding]
    steps: list[PreparationStep]
    clean_df: pd.DataFrame
    validations: list[ValidationResult]
    # control flow
    levels: dict[str, int]        # finding key -> strategy level (0 = preferred, 1+ = fallback)
    feedback: list[str]           # why the previous attempt failed (fed back to the preparer)
    retry_count: int
    route: str                    # validator decision: "retry" | "report"
    needs_review: bool
    unresolved: list[str]
    warnings: list[str]
    errors: list[str]
    trace: list[dict[str, Any]]
    fatal: bool
    report: Any
