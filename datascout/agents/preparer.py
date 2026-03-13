from __future__ import annotations

from ..llm import LLMClient
from ..tools.preparation import run_plan


def preparer_node(state: dict, llm: LLMClient) -> dict:
    """Plan fixes (LLM), then execute them from the original data so a retry never stacks on a bad attempt."""
    findings, target = state["findings"], state.get("target")
    levels = dict(state.get("levels", {}))
    plan = llm.plan_preparation(findings, state["profiles"], state["task"], target, levels, state.get("feedback", []))
    clean, steps, notes = run_plan(state["original_df"], plan.actions, target, findings)
    warnings = list(state.get("warnings", [])) + notes + llm.pop_events()
    return {"clean_df": clean, "steps": steps, "warnings": warnings,
            "_note": f"attempt {state.get('retry_count', 0) + 1}: {len(steps)} steps applied, {len(notes)} notes"}
