"""Schemas for drift detection (comparison of a reference dataset with a current one)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .findings import Severity


class DriftFinding(BaseModel):
    column: str
    kind: str                       # numeric | categorical
    test: str                       # ks_2samp | chi2_contingency
    statistic_name: str             # ks_d | total_variation_distance
    statistic: float
    p_value: float
    severity: Severity
    evidence: dict[str, Any] = Field(default_factory=dict)
    recommendation: str = ""


class DriftReport(BaseModel):
    n_reference_rows: int
    n_current_rows: int
    columns_compared: list[str] = Field(default_factory=list)
    columns_skipped: dict[str, str] = Field(default_factory=dict)   # column -> reason
    findings: list[DriftFinding] = Field(default_factory=list)
    verified: bool = False          # every statistic was recomputed by separate code and matched
    verification_errors: list[str] = Field(default_factory=list)
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    alpha_used: float = 0.0         # per-column significance level after the Bonferroni split
    target: Optional[str] = None
