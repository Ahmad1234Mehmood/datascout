import numpy as np
import pandas as pd
import pytest

from datascout.agents.graph import analyze_dataframe
from datascout.llm import ChatModelClient, HeuristicClient, LLMCallError
from datascout.config import get_settings
from datascout.models.findings import CheckCall, CheckPlan, Judgement, PrepAction, ActionType, PrepPlan
from datascout.tools import preparation


def messy(seed=1, n=600):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({"customer_id": np.arange(n), "age": r.normal(40, 10, n), "income": r.normal(50, 8, n),
                       "plan": r.choice(["basic", "pro", "team"], n), "churn": r.integers(0, 2, n)})
    df.loc[:89, "age"] = np.nan                                    # 15% missing, numeric
    df.loc[100:119, "income"] = 900.0                              # outliers
    df["leak"] = df["churn"] * 5 + r.normal(scale=0.01, size=n)    # leakage
    df["income2"] = df["income"] * 2 + r.normal(scale=0.01, size=n)  # collinear (after outliers still ~1)
    df = pd.concat([df, df.iloc[200:240]], ignore_index=True)      # duplicates
    return df


def test_end_to_end_finds_fixes_and_verifies():
    report, clean = analyze_dataframe(messy(), "churn prediction", "churn")
    kinds = {(f.issue_type.value, f.column) for f in report.findings}
    assert ("leakage", "leak") in kinds
    assert ("missing_values", "age") in kinds
    assert ("outliers", "income") in kinds
    assert ("duplicates", None) in kinds
    assert "leak" not in clean.columns and clean["age"].isna().sum() == 0 and clean.duplicated().sum() == 0
    assert "churn" in clean.columns
    assert not report.needs_human_review and report.retries == 0
    assert all(v.passed for v in report.validations if v.subject.startswith(("finding:", "step:", "invariant:")))
    assert report.quality_score_after > report.quality_score_before
    assert [t.node for t in report.trace][:3] == ["profiler", "quality", "preparer"]


def test_clean_data_skips_preparation():
    r = np.random.default_rng(3)
    df = pd.DataFrame({"a": r.normal(size=500), "b": r.normal(size=500), "g": r.choice(list("xy"), 500),
                       "y": r.integers(0, 2, 500)})
    report, clean = analyze_dataframe(df, "demo", "y")
    assert report.findings == [] and report.preparation_steps == []
    assert "preparer" not in [t.node for t in report.trace]           # conditional edge skipped it
    assert clean.equals(df)


def test_retry_then_success(monkeypatch):
    """First imputation silently does nothing -> validator rejects -> preparer retries with fallback."""
    real = preparation.apply_action
    calls = {"n": 0}

    def flaky(df, action, target, key):
        if action.action == ActionType.impute_median:
            calls["n"] += 1
            if calls["n"] == 1:                      # sabotage only the first attempt
                out, step = real(df, action, target, key)
                return df.copy(), step              # pretend to impute, change nothing
        return real(df, action, target, key)

    monkeypatch.setattr(preparation, "apply_action", flaky)
    report, clean = analyze_dataframe(messy(), "churn", "churn")
    assert report.retries == 1 and not report.needs_human_review
    assert clean["age"].isna().sum() == 0


def test_retry_budget_exhausted_flags_review(monkeypatch):
    real = preparation.apply_action

    def always_broken(df, action, target, key):
        out, step = real(df, action, target, key)
        if action.action in (ActionType.impute_median, ActionType.drop_rows_missing, ActionType.drop_column) and action.column == "age":
            return df.copy(), step
        return out, step

    monkeypatch.setattr(preparation, "apply_action", always_broken)
    report, _ = analyze_dataframe(messy(), "churn", "churn")
    assert report.retries == 2 and report.needs_human_review
    assert any("age" in u for u in report.unresolved)


def test_node_crash_returns_partial_report(monkeypatch):
    import datascout.agents.graph as g
    monkeypatch.setattr(g, "preparer_node", lambda s, llm: (_ for _ in ()).throw(RuntimeError("boom")))
    report, _ = analyze_dataframe(messy(), "churn", "churn")
    assert report.needs_human_review and "boom" in report.summary
    assert report.findings                                        # work done before the crash is kept


def test_missing_target_is_reported_not_fatal():
    report, _ = analyze_dataframe(messy(), "x", "no_such_col")
    assert any("not found" in w for w in report.warnings)
    assert not any(f.issue_type.value == "leakage" for f in report.findings)


# ---------------- LLM client behaviour with a stub chat model ----------------

class StubModel:
    def __init__(self, outputs):
        self.outputs, self.i = outputs, 0

    def with_structured_output(self, schema):
        outer = self

        class R:
            def invoke(self, msgs, config=None):
                o = outer.outputs[min(outer.i, len(outer.outputs) - 1)]
                outer.i += 1
                if isinstance(o, Exception):
                    raise o
                return o
        return R()


def client(outputs, retries=2):
    s = get_settings()
    object.__setattr__(s, "llm_max_retries", retries)
    c = ChatModelClient(StubModel(outputs), "stub", s)
    return c


def test_llm_bad_tool_and_column_are_guarded(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    df = messy()
    from datascout.tools.profiling import profile_columns
    plan = CheckPlan(calls=[CheckCall(tool="rm_rf", column="age"), CheckCall(tool="check_missing", column="agee"),
                            CheckCall(tool="check_missing", column="nope")])
    out = client([plan]).plan_checks(profile_columns(df), "t", "churn")
    cols = [(c.tool, c.column) for c in out.calls]
    assert ("check_missing", "age") in cols            # typo corrected
    assert not any(t == "rm_rf" for t, _ in cols) and ("check_missing", "nope") not in cols
    assert any(t == "check_duplicates" for t, _ in cols)   # baseline checks always run


def test_llm_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    llm = client([TimeoutError("ollama down")])
    report, clean = analyze_dataframe(messy(), "churn", "churn", llm=llm)
    assert {f.issue_type.value for f in report.findings} >= {"leakage", "missing_values", "outliers"}
    assert any("LLM failed" in w for w in report.warnings)
    assert "leak" not in clean.columns


def test_hallucinated_numbers_in_llm_advice_are_repaired(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    from datascout.llm import Recs, Rec
    df = messy()
    llm = client([TimeoutError("x")])
    real = llm.recommend_many
    llm.recommend_many = lambda findings, task: {f.key: "Roughly 77% of this is bad, fix it." for f in findings}
    report, _ = analyze_dataframe(df, "churn", "churn", llm=llm)
    assert all("77" not in f.recommendation for f in report.findings)
    assert any(v.subject.startswith("grounding:") and not v.passed for v in report.validations)
    assert not report.needs_human_review                         # repaired, not blocking


def test_llm_invalid_plan_action_is_retried_via_coverage(monkeypatch):
    """LLM proposes dropping a column that doesn't exist and forgets the outliers: coverage check forces a retry."""
    monkeypatch.setattr("time.sleep", lambda *_: None)

    class WeakLLM(HeuristicClient):
        def plan_preparation(self, findings, profiles, task, target, levels, feedback):
            if not feedback:
                return PrepPlan(actions=[PrepAction(action=ActionType.drop_column, column="ghost")])
            return super().plan_preparation(findings, profiles, task, target, {k: 0 for k in levels}, feedback)

    report, clean = analyze_dataframe(messy(), "churn", "churn", llm=WeakLLM())
    assert report.retries == 1 and not report.needs_human_review
    assert "leak" not in clean.columns


def test_model_that_skips_checks_is_covered_and_reported():
    """A model plan without detect_outliers must not hide the outliers."""
    class Lazy(HeuristicClient):
        name = "lazy"

        def plan_checks(self, profiles, task, target):
            return CheckPlan(calls=[CheckCall(tool="check_missing", column=p.name) for p in profiles])

    report, _ = analyze_dataframe(messy(), "churn", "churn", llm=Lazy())
    assert "outliers" in {f.issue_type.value for f in report.findings}
    assert any(w.startswith("coverage:") and "detect_outliers" in w for w in report.warnings)


def test_judge_only_veto_costs_one_retry_not_the_whole_budget():
    class Vetoer(HeuristicClient):
        name = "vetoer"

        def judge_steps(self, findings, steps, task):
            by_key = {f.key: f for f in findings}
            return {i: Judgement(appropriate=by_key[s.finding_key].issue_type.value != "multicollinearity",
                                 reason="dropping loses information")
                    for i, s in enumerate(steps) if s.finding_key in by_key}

    report, clean = analyze_dataframe(messy(), "churn", "churn", llm=Vetoer())
    assert report.retries == 1
    assert report.needs_human_review
    assert any("judge disagrees" in u for u in report.unresolved)
