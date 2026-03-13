"""Smoke-test the configured LLM provider on examples/churn_demo.csv."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx                                              # noqa: E402

from datascout.agents.graph import analyze_file          # noqa: E402
from datascout.config import get_settings                # noqa: E402
from datascout.llm import get_llm_client                 # noqa: E402

DEMO = Path(__file__).resolve().parents[1] / "examples" / "churn_demo.csv"


def main() -> int:
    s = get_settings()
    print(f"provider={s.llm_provider}  model={s.llm_model}")
    if s.llm_provider == "heuristic":
        print("Provider is 'heuristic' (no LLM). Set DATASCOUT_LLM_PROVIDER in .env to test a real model.")
        return 0

    if s.llm_provider == "ollama":
        print(f"ollama url={s.ollama_base_url}")
        try:
            tags = httpx.get(f"{s.ollama_base_url.rstrip('/')}/api/tags", timeout=15).json()
            names = [m["name"] for m in tags.get("models", [])]
            print("models on server:", names or "none")
            if not any(n == s.llm_model or n.split(":")[0] == s.llm_model for n in names):
                print(f"WARNING: '{s.llm_model}' is not listed; run `ollama pull {s.llm_model}` on that server.")
        except Exception as e:
            print(f"FAIL: cannot reach the Ollama server: {type(e).__name__}: {e}")
            print("Check OLLAMA_BASE_URL (tunnel URL changes every Kaggle/Colab session) and that the notebook is still running.")
            return 1

    client = get_llm_client(s)
    t0 = time.perf_counter()
    report, _ = analyze_file(DEMO, "churn prediction", "churned", llm=client)
    dt = time.perf_counter() - t0

    # full report for debugging (ignored by git)
    out = Path(__file__).resolve().parents[1] / "check_llm_report.json"
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    fallbacks = [w for w in report.warnings if "LLM failed" in w or "rule-based" in w]
    print(f"\npipeline finished in {dt:.1f}s | findings={len(report.findings)} steps={len(report.preparation_steps)} "
          f"retries={report.retries} needs_review={report.needs_human_review}")
    print("summary:", report.summary)
    judged = [v for v in report.validations if v.subject.startswith("judge:")]
    print(f"LLM judge verdicts: {sum(v.passed for v in judged)}/{len(judged)} fixes judged appropriate")
    grounding_fail = [v.subject for v in report.validations if v.subject.startswith("grounding:") and not v.passed]
    print(f"LLM advice repaired by the number-grounding check: {len(grounding_fail)}")
    print("\n--- details (send this block when something looks wrong) ---")
    print("findings:", [f"{f.issue_type.value}:{f.column}" for f in report.findings])
    print("steps:   ", [f"{st.action.value}:{st.column}" for st in report.preparation_steps])
    print("residual findings after cleaning:", [f"{f.issue_type.value}:{f.column}" for f in report.residual_findings])
    print("unresolved:", report.unresolved)
    for v in report.validations:
        if not v.passed:
            print(f"  FAILED {v.subject} | claimed={v.claimed} actual={v.actual} | {v.detail}")
    for v in judged:
        print(f"  judge {'OK ' if v.passed else 'NO '} {v.subject} | {v.detail}")
    print("warnings:")
    for w in report.warnings:
        print("  -", w)
    print(f"(full report written to {out.name})")
    if fallbacks:
        print("\nFAIL: the rule-based fallback replaced the LLM at least once:")
        for w in fallbacks:
            print("  -", w)
        return 1
    print("\nOK: every LLM call was served by the model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())