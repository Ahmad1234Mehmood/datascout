"""Pydantic schemas: every agent output is validated against these."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class IssueType(str, Enum):
    missing_values = "missing_values"
    outliers = "outliers"
    leakage = "leakage"
    imbalance = "imbalance"
    duplicates = "duplicates"
    multicollinearity = "multicollinearity"
    constant_column = "constant_column"


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class ActionType(str, Enum):
    drop_column = "drop_column"
    impute_median = "impute_median"
    impute_mode = "impute_mode"
    drop_rows_missing = "drop_rows_missing"
    clip_outliers = "clip_outliers"
    remove_outlier_rows = "remove_outlier_rows"
    drop_duplicates = "drop_duplicates"


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    semantic_type: str  # numeric | categorical | datetime | text | id | boolean | constant | empty
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skew: Optional[float] = None
    top_values: dict[str, int] = Field(default_factory=dict)


class QualityFinding(BaseModel):
    column: Optional[str] = None  # None for dataset-level findings; "a~b" for column pairs
    issue_type: IssueType
    severity: Severity
    evidence: dict[str, Any]
    recommendation: str = ""

    @property
    def key(self) -> str:
        return f"{self.issue_type.value}:{self.column}"


# ---- LLM-facing plan schemas (what the model is allowed to decide) ----

class CheckCall(BaseModel):
    tool: str
    column: Optional[str] = None


class CheckPlan(BaseModel):
    calls: list[CheckCall]


class PrepAction(BaseModel):
    action: ActionType
    column: Optional[str] = None
    finding_key: str = ""   # "<issue_type>:<column>"; derived by the executor when blank
    rationale: str = ""


class PrepPlan(BaseModel):
    actions: list[PrepAction]


class Judgement(BaseModel):
    appropriate: bool
    reason: str = ""


# ---- execution records ----

class PreparationStep(BaseModel):
    finding_key: str
    action: ActionType
    column: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    before_stats: dict[str, Any] = Field(default_factory=dict)
    after_stats: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ValidationResult(BaseModel):
    subject: str            # "finding:<key>" | "step:<key>" | "invariant:<name>" | "grounding:<key>"
    passed: bool
    claimed: Any = None
    actual: Any = None
    detail: str = ""
