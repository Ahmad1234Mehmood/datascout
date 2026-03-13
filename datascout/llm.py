"""LLM abstraction: rule-based HeuristicClient and ChatModelClient (LangChain) behind one interface."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, Protocol

import difflib
from pydantic import BaseModel, ValidationError

from .config import Settings, get_settings
from .models.findings import (ActionType, CheckCall, CheckPlan, ColumnProfile, IssueType, Judgement, PrepAction,
                              PrepPlan, PreparationStep, QualityFinding)
from .recommendations import template_recommendation
from .tools.quality import CHECKS

log = logging.getLogger("datascout.llm")
PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


class LLMCallError(RuntimeError):
    pass


# ---- extra structured-output schemas (batched, to cut the number of LLM calls on CPU) ----

class Rec(BaseModel):
    key: str
    text: str


class Recs(BaseModel):
    items: list[Rec]


class JudgeItem(BaseModel):
    step_id: int
    appropriate: bool
    reason: str = ""


class JudgeItems(BaseModel):
    items: list[JudgeItem]


class LLMClient(Protocol):
    name: str

    def plan_checks(self, profiles: list[ColumnProfile], task: str, target: Optional[str]) -> CheckPlan: ...
    def plan_preparation(self, findings: list[QualityFinding], profiles: list[ColumnProfile], task: str,
                         target: Optional[str], levels: dict[str, int], feedback: list[str]) -> PrepPlan: ...
    def recommend_many(self, findings: list[QualityFinding], task: str) -> dict[str, str]: ...
    def judge_steps(self, findings: list[QualityFinding], steps: list[PreparationStep], task: str) -> dict[int, Judgement]: ...
    def pop_events(self) -> list[str]: ...


# =============================== rule-based client ===============================

class HeuristicClient:
    name = "heuristic"

    def __init__(self) -> None:
        self.events: list[str] = []

    def pop_events(self) -> list[str]:
        ev, self.events = self.events, []
        return ev

    # -- quality agent: which checks to run
    def plan_checks(self, profiles, task, target) -> CheckPlan:
        calls: list[CheckCall] = []
        for p in profiles:
            calls.append(CheckCall(tool="check_missing", column=p.name))
            if p.semantic_type == "constant":
                calls.append(CheckCall(tool="check_constant", column=p.name))
            elif p.semantic_type == "numeric" and p.name != target:
                calls.append(CheckCall(tool="detect_outliers", column=p.name))
        calls += baseline_checks(target)
        return CheckPlan(calls=calls)

    # -- preparation agent: which fixes to apply
    def plan_preparation(self, findings, profiles, task, target, levels, feedback) -> PrepPlan:
        sem = {p.name: p.semantic_type for p in profiles}
        miss = {p.name: p.missing_count for p in profiles}
        actions: list[PrepAction] = []
        dropped: set[str] = set()

        def drop(col: str, f: QualityFinding, why: str) -> None:
            if col != target and col not in dropped:
                dropped.add(col)
                actions.append(PrepAction(action=ActionType.drop_column, column=col, finding_key=f.key, rationale=why))

        for f in findings:
            if f.issue_type == IssueType.leakage:
                drop(f.column, f, "feature encodes the target")
            elif f.issue_type == IssueType.constant_column:
                drop(f.column, f, "single value, no information")
        pairs = sorted((f for f in findings if f.issue_type == IssueType.multicollinearity),
                       key=lambda f: -abs(f.evidence["correlation"]))
        for f in pairs:
            a, b = f.evidence["column_a"], f.evidence["column_b"]
            if a in dropped or b in dropped:
                continue
            drop(a if miss.get(a, 0) > miss.get(b, 0) else b, f, "redundant with its collinear partner")

        for f in findings:
            lvl = levels.get(f.key, 0)
            col = f.column
            if f.issue_type == IssueType.missing_values and col not in dropped:
                pct = f.evidence["missing_pct"]
                if col == target:
                    actions.append(PrepAction(action=ActionType.drop_rows_missing, column=col, finding_key=f.key,
                                              rationale="target must not be imputed"))
                elif lvl == 0:
                    if pct >= 60:
                        drop(col, f, "mostly missing")
                    elif sem.get(col) == "numeric":
                        actions.append(PrepAction(action=ActionType.impute_median, column=col, finding_key=f.key,
                                                  rationale="robust to skew"))
                    else:
                        actions.append(PrepAction(action=ActionType.impute_mode, column=col, finding_key=f.key,
                                                  rationale="categorical column"))
                else:  # fallback strategy after a failed attempt
                    if pct <= 20:
                        actions.append(PrepAction(action=ActionType.drop_rows_missing, column=col, finding_key=f.key,
                                                  rationale="fallback: drop the few incomplete rows"))
                    else:
                        drop(col, f, "fallback: too incomplete to repair")
            elif f.issue_type == IssueType.outliers and col not in dropped:
                act = ActionType.clip_outliers if lvl == 0 else ActionType.remove_outlier_rows
                actions.append(PrepAction(action=act, column=col, finding_key=f.key,
                                          rationale="bound extreme values" if lvl == 0 else "fallback: remove rows"))
            elif f.issue_type == IssueType.duplicates:
                actions.append(PrepAction(action=ActionType.drop_duplicates, finding_key=f.key,
                                          rationale="exact duplicate rows"))
        return PrepPlan(actions=actions)

    def recommend_many(self, findings, task) -> dict[str, str]:
        return {f.key: template_recommendation(f) for f in findings}

    # -- validator agent (judge): is the chosen fix appropriate?
    def judge_steps(self, findings, steps, task) -> dict[int, Judgement]:
        by_key = {f.key: f for f in findings}
        out: dict[int, Judgement] = {}
        for i, s in enumerate(steps):
            f = by_key.get(s.finding_key)
            ok, why = True, "appropriate"
            if f is not None and f.issue_type == IssueType.missing_values:
                pct = f.evidence["missing_pct"]
                if s.action == ActionType.drop_column and pct < 20:
                    ok, why = False, f"only {pct}% missing: dropping the column wastes information"
            if f is not None and f.issue_type == IssueType.outliers and s.action == ActionType.remove_outlier_rows:
                if f.evidence["outlier_pct"] > 10:
                    ok, why = False, "removing this many rows is destructive"
            out[i] = Judgement(appropriate=ok, reason=why)
        return out


def baseline_checks(target: Optional[str]) -> list[CheckCall]:
    """Dataset-level safety checks that always run, whatever the LLM plans."""
    calls = [CheckCall(tool="check_duplicates"), CheckCall(tool="check_correlation")]
    if target:
        calls += [CheckCall(tool="check_imbalance", column=target), CheckCall(tool="check_leakage", column=target)]
    return calls


# =============================== LLM-backed client ===============================

class ChatModelClient:
    def __init__(self, model, name: str, settings: Settings, callbacks: list | None = None, anthropic_cache: bool = False):
        self.model, self.name, self.s = model, name, settings
        self.callbacks = callbacks or []
        self.anthropic_cache = anthropic_cache
        self.fallback = HeuristicClient()
        self.events: list[str] = []

    def pop_events(self) -> list[str]:
        ev, self.events = self.events + self.fallback.pop_events(), []
        return ev

    # ---- core: structured call with retry/backoff and Pydantic validation ----
    def _system(self, text: str):
        if self.anthropic_cache:   # only pays off when the prefix exceeds the provider's minimum cacheable size
            return ("system", [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}])
        return ("system", text)    # stable prefix -> Ollama reuses its KV cache while the model stays loaded

    def _call(self, schema: type[BaseModel], system: str, user: str):
        last: Exception | None = None
        prompt = user
        for attempt in range(self.s.llm_max_retries):
            try:
                runnable = self.model.with_structured_output(schema)
                cfg = {"callbacks": self.callbacks} if self.callbacks else {}
                res = runnable.invoke([self._system(system), ("human", prompt)], config=cfg)
                return res if isinstance(res, schema) else schema.model_validate(res)
            except ValidationError as e:
                last = e
                prompt = user + f"\n\nYour previous reply was invalid ({str(e)[:300]}). Reply with JSON matching the schema exactly."
            except Exception as e:  # timeouts, connection errors, parser errors
                last = e
            if attempt < self.s.llm_max_retries - 1:
                time.sleep(min(2 ** attempt, 8))
        raise LLMCallError(f"{type(last).__name__}: {last}")

    # ---- quality agent ----
    def plan_checks(self, profiles, task, target) -> CheckPlan:
        system = load_prompt("quality_system")
        names = {p.name for p in profiles}
        calls: list[CheckCall] = []
        batch = self.s.columns_per_batch                       # context management: <= N profiles per call
        for i in range(0, len(profiles), batch):
            chunk = profiles[i:i + batch]
            user = ("MODE=plan_checks\n" f"TASK: {task}\nTARGET: {target}\n"
                    f"PROFILES: {json.dumps([_compact(p) for p in chunk])}")
            try:
                plan = self._call(CheckPlan, system, user)
                calls += [c for c in (_fix_call(c, names) for c in plan.calls) if c is not None]
            except LLMCallError as e:
                self.events.append(f"plan_checks batch {i // batch}: LLM failed ({e}); used rule-based plan")
                calls += [c for c in self.fallback.plan_checks(chunk, task, target).calls if c.column]
        calls += baseline_checks(target)
        return CheckPlan(calls=calls)

    def recommend_many(self, findings, task) -> dict[str, str]:
        if not findings:
            return {}
        system = load_prompt("quality_system")
        user = ("MODE=recommendations\n" f"TASK: {task}\n"
                f"FINDINGS: {json.dumps([{'key': f.key, 'issue': f.issue_type.value, 'severity': f.severity.value, 'evidence': f.evidence} for f in findings])}")
        try:
            recs = self._call(Recs, system, user)
            got = {r.key: r.text for r in recs.items}
        except LLMCallError as e:
            self.events.append(f"recommendations: LLM failed ({e}); used templates")
            got = {}
        return {f.key: got.get(f.key) or template_recommendation(f) for f in findings}

    # ---- preparation agent ----
    def plan_preparation(self, findings, profiles, task, target, levels, feedback) -> PrepPlan:
        if levels and max(levels.values(), default=0) >= 2:      # last-resort: deterministic fallback
            self.events.append("preparation: retry budget nearly spent, used rule-based plan")
            return self.fallback.plan_preparation(findings, profiles, task, target, {k: max(v, 1) for k, v in levels.items()}, feedback)
        system = load_prompt("preparer_system")
        sem = {p.name: p.semantic_type for p in profiles}
        user = (f"TASK: {task}\nTARGET: {target}\nCOLUMN_TYPES: {json.dumps(sem)}\n"
                f"FINDINGS: {json.dumps([{'key': f.key, 'issue': f.issue_type.value, 'column': f.column, 'evidence': f.evidence} for f in findings])}\n"
                f"FEEDBACK: {json.dumps(feedback)}")
        try:
            plan = self._call(PrepPlan, system, user)
            cols = {p.name for p in profiles}
            ok = [a for a in plan.actions if a.action == ActionType.drop_duplicates or (a.column in cols)]
            if len(ok) < len(plan.actions):
                self.events.append(f"preparation: discarded {len(plan.actions) - len(ok)} action(s) naming unknown columns")
            return PrepPlan(actions=ok)
        except LLMCallError as e:
            self.events.append(f"preparation: LLM failed ({e}); used rule-based plan")
            return self.fallback.plan_preparation(findings, profiles, task, target, levels, feedback)

    # ---- validator agent (LLM judge) ----
    def judge_steps(self, findings, steps, task) -> dict[int, Judgement]:
        if not steps:
            return {}
        system = load_prompt("validator_system")
        by_key = {f.key: f for f in findings}
        user = (f"TASK: {task}\nSTEPS: " + json.dumps([
            {"step_id": i, "action": s.action.value, "column": s.column,
             "finding": {"issue": by_key[s.finding_key].issue_type.value, "evidence": by_key[s.finding_key].evidence}
             if s.finding_key in by_key else None} for i, s in enumerate(steps)]))
        try:
            res = self._call(JudgeItems, system, user)
            return {it.step_id: Judgement(appropriate=it.appropriate, reason=it.reason) for it in res.items}
        except LLMCallError as e:
            self.events.append(f"judge: LLM failed ({e}); used rule-based judge")
            return self.fallback.judge_steps(findings, steps, task)


def _compact(p: ColumnProfile) -> dict:
    d = p.model_dump(exclude_none=True)
    d.pop("top_values", None) if p.semantic_type not in ("categorical", "boolean") else None
    return d


def _fix_call(c: CheckCall, names: set[str]) -> CheckCall | None:
    """Tool-call guard: unknown tools are dropped; misspelled columns are corrected to the closest real one."""
    if c.tool not in CHECKS:
        return None
    scope = CHECKS[c.tool][1]
    if scope != "column":
        return None                      # dataset-level checks are added by baseline_checks()
    if c.column in names:
        return c
    close = difflib.get_close_matches(c.column or "", list(names), n=1, cutoff=0.8)
    return CheckCall(tool=c.tool, column=close[0]) if close else None


# =============================== factory ===============================

def get_llm_client(settings: Settings | None = None) -> LLMClient:
    s = settings or get_settings()
    provider = s.llm_provider.lower()
    if provider == "heuristic":
        return HeuristicClient()

    callbacks: list = []
    if s.langfuse_enabled:
        try:
            from langfuse.langchain import CallbackHandler
            callbacks.append(CallbackHandler())
        except Exception as e:  # tracing must never break the pipeline
            log.warning("Langfuse callback unavailable: %s", e)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        model = ChatOllama(model=s.llm_model, base_url=s.ollama_base_url, temperature=0,
                           keep_alive="30m", num_ctx=8192, client_kwargs={"timeout": s.llm_timeout_s})
        return ChatModelClient(model, f"ollama:{s.llm_model}", s, callbacks)
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise RuntimeError("pip install langchain-openai to use DATASCOUT_LLM_PROVIDER=openai") from e
        kwargs = {"base_url": s.openai_base_url} if s.openai_base_url else {}
        return ChatModelClient(ChatOpenAI(model=s.llm_model, temperature=0, timeout=s.llm_timeout_s, **kwargs),
                               f"openai:{s.llm_model}", s, callbacks)
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise RuntimeError("pip install langchain-anthropic to use DATASCOUT_LLM_PROVIDER=anthropic") from e
        return ChatModelClient(ChatAnthropic(model=s.llm_model, temperature=0, timeout=s.llm_timeout_s),
                               f"anthropic:{s.llm_model}", s, callbacks, anthropic_cache=True)
    raise ValueError(f"Unknown DATASCOUT_LLM_PROVIDER '{s.llm_provider}'")
