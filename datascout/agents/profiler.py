from __future__ import annotations

from ..tools.profiling import profile_columns


def profiler_node(state: dict) -> dict:
    """Deterministic tool node: the LLM later reasons over these summaries, never over raw rows."""
    df = state["original_df"]
    profiles = profile_columns(df)
    warnings = list(state.get("warnings", []))
    target = state.get("target")
    if target and target not in df.columns:
        warnings.append(f"target '{target}' not found in the dataset; target-based checks skipped")
        state_target = None
    else:
        state_target = target
        if not target:
            warnings.append("no target column given; leakage and class-imbalance checks skipped")
    return {"profiles": profiles, "target": state_target, "warnings": warnings,
            "_note": f"{len(profiles)} columns, {len(df)} rows"}
