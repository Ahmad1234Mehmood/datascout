from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from .findings import ColumnProfile, PreparationStep, QualityFinding, ValidationResult


class TraceEvent(BaseModel):
    node: str
    seconds: float
    status: str  # ok | error
    note: str = ""


class QualityReport(BaseModel):
    dataset_name: str
    task_description: str
    target: Optional[str] = None
    llm_provider: str

    n_rows_before: int
    n_cols_before: int
    n_rows_after: int
    n_cols_after: int

    profiles: list[ColumnProfile] = Field(default_factory=list)
    findings: list[QualityFinding] = Field(default_factory=list)
    preparation_steps: list[PreparationStep] = Field(default_factory=list)
    validations: list[ValidationResult] = Field(default_factory=list)
    residual_findings: list[QualityFinding] = Field(default_factory=list)

    quality_score_before: float
    quality_score_after: float
    retries: int = 0
    needs_human_review: bool = False
    unresolved: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: str = ""
    trace: list[TraceEvent] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
