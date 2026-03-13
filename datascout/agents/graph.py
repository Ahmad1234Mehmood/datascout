"""LangGraph pipeline: profiler -> quality -> preparer -> validator (retry/flag) -> reporter."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from langgraph.graph import END, START, StateGraph

from ..config import get_settings
from ..llm import LLMClient, get_llm_client
from ..models.report import QualityReport
from ..tools.profiling import load_dataframe
from ..tracing.langfuse_setup import flush_langfuse, traced
from .preparer import preparer_node
from .profiler import profiler_node
from .quality_checker import quality_node
from .reporter import reporter_node
from .state import AgentState
from .validator import validator_node


def _after(next_node: str):
    return lambda s: "reporter" if s.get("fatal") else next_node


def _after_quality(s: dict) -> str:
    if s.get("fatal") or not s.get("findings"):
        return "reporter"
    return "preparer"


def _after_validator(s: dict) -> str:
    if s.get("fatal"):
        return "reporter"
    return "preparer" if s.get("route") == "retry" else "reporter"


def build_graph(llm: LLMClient):
    g = StateGraph(AgentState)
    g.add_node("profiler", traced("profiler")(profiler_node))
    g.add_node("quality", traced("quality")(lambda s: quality_node(s, llm)))
    g.add_node("preparer", traced("preparer")(lambda s: preparer_node(s, llm)))
    g.add_node("validator", traced("validator")(lambda s: validator_node(s, llm)))
    g.add_node("reporter", traced("reporter")(lambda s: reporter_node(s, llm.name)))

    g.add_edge(START, "profiler")
    g.add_conditional_edges("profiler", _after("quality"), {"quality": "quality", "reporter": "reporter"})
    g.add_conditional_edges("quality", _after_quality, {"preparer": "preparer", "reporter": "reporter"})
    g.add_conditional_edges("preparer", _after("validator"), {"validator": "validator", "reporter": "reporter"})
    g.add_conditional_edges("validator", _after_validator, {"preparer": "preparer", "reporter": "reporter"})
    g.add_edge("reporter", END)
    return g.compile()


def analyze_dataframe(df: pd.DataFrame, task: str, target: str | None = None, name: str = "dataset",
                      llm: LLMClient | None = None) -> tuple[QualityReport, pd.DataFrame]:
    settings = get_settings()
    llm = llm or get_llm_client(settings)
    app = build_graph(llm)
    try:
        final = app.invoke({"dataset_name": name, "task": task, "target": target or None, "original_df": df,
                            "levels": {}, "feedback": [], "retry_count": 0, "max_retries": settings.max_prep_retries,
                            "warnings": list(df.attrs.get("load_notes", [])), "errors": [], "trace": []}, {"recursion_limit": 40})
    finally:
        flush_langfuse()
    report: QualityReport = final["report"]
    return report, final["clean_df"]


def analyze_file(path: str | Path, task: str, target: str | None = None, llm: LLMClient | None = None
                 ) -> tuple[QualityReport, pd.DataFrame]:
    df = load_dataframe(path)
    return analyze_dataframe(df, task, target, Path(path).name, llm)
