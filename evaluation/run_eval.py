"""Run DataScout on the golden set, score detection and enforce the CI gate."""
from __future__ import annotations

import argparse
import json
import re
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datascout.agents.graph import analyze_file          # noqa: E402
from datascout.config import get_settings                # noqa: E402
from datascout.llm import get_llm_client                 # noqa: E402
from evaluation.metrics import aggregate, score_dataset  # noqa: E402

GOLD = Path(__file__).parent / "golden_sets"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--min-precision", type=float, default=0.90)
    ap.add_argument("--min-recall", type=float, default=0.90)
    ap.add_argument("--max-clean-false-positive-datasets", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--markdown-out", default=None)
    a = ap.parse_args()
    if a.provider:
        os.environ["DATASCOUT_LLM_PROVIDER"] = a.provider

    llm = get_llm_client(get_settings())
    per: dict[str, dict] = {}
    pipeline: dict[str, dict] = {}
    clean_fp: list[str] = []
    t_start = time.perf_counter()
    for csv in sorted((GOLD / "datasets").glob("*.csv")):
        ann = json.loads((GOLD / "annotations" / f"{csv.stem}.json").read_text())
        t0 = time.perf_counter()
        report, clean = analyze_file(csv, ann["task"], ann["target"], llm=llm)
        found = [{"issue_type": f.issue_type.value, "column": f.column} for f in report.findings]
        per[csv.stem] = score_dataset(ann["expected"], found)
        step_fail = sum(not v.passed for v in report.validations if v.subject.startswith(("step:", "invariant:", "finding:")))
        judged = [v for v in report.validations if v.subject.startswith("judge:")]
        pipeline[csv.stem] = {"seconds": round(time.perf_counter() - t0, 2), "retries": report.retries,
                              "needs_review": report.needs_human_review, "failed_checks": step_fail,
                              "score_before": report.quality_score_before, "score_after": report.quality_score_after,
                              # LLM-specific signals (all zero for the rule-based client)
                              "fallbacks": sum("LLM failed" in w or "rule-based" in w for w in report.warnings),
                              "checks_omitted_by_model": sum(int(m.group(1)) for w in report.warnings
                                                             if (m := re.match(r"coverage: .*?omitted (\d+)", w))),
                              "judge_total": len(judged), "judge_rejections": sum(not v.passed for v in judged),
                              "grounding_repairs": sum(v.subject.startswith("grounding:") and not v.passed
                                                       for v in report.validations),
                              "review_reasons": report.unresolved}
        if not ann["expected"] and found:
            clean_fp.append(csv.stem)

    agg = aggregate(per)
    n = len(per)
    result = {"provider": llm.name, "datasets": n, "aggregate": agg,
              "clean_datasets": sum(1 for s in per if not json.loads((GOLD / 'annotations' / f'{s}.json').read_text())["expected"]),
              "clean_datasets_with_false_positives": clean_fp,
              "pipeline_success_rate": round(sum(not p["needs_review"] and p["failed_checks"] == 0 for p in pipeline.values()) / n, 4),
              "retried_datasets": sum(p["retries"] > 0 for p in pipeline.values()),
              "llm_signals": {k: sum(p[k] for p in pipeline.values()) for k in
                              ("fallbacks", "checks_omitted_by_model", "judge_total", "judge_rejections", "grounding_repairs")},
              "total_seconds": round(time.perf_counter() - t_start, 1),
              "per_dataset": {k: {**per[k], **pipeline[k]} for k in per}}

    gates = {
        f"precision >= {a.min_precision}": agg["precision"] >= a.min_precision,
        f"recall >= {a.min_recall}": agg["recall"] >= a.min_recall,
        f"clean datasets with false positives <= {a.max_clean_false_positive_datasets}": len(clean_fp) <= a.max_clean_false_positive_datasets,
    }
    result["gates"] = gates

    md = [f"## DataScout evaluation ({llm.name})", "",
          f"| metric | value |", "|---|---|",
          f"| datasets | {n} ({result['clean_datasets']} clean hard-negatives) |",
          f"| precision | {agg['precision']} ({agg['tp']} TP, {agg['fp']} FP) |",
          f"| recall | {agg['recall']} ({agg['fn']} missed) |",
          f"| F1 | {agg['f1']} |",
          f"| pipeline success (no review flag, all checks pass) | {result['pipeline_success_rate']} |",
          f"| datasets needing a retry | {result['retried_datasets']} |",
          f"| steps judged / rejected by the judge | {result['llm_signals']['judge_total']} / {result['llm_signals']['judge_rejections']} |",
          f"| rule-based fallbacks (model unusable) | {result['llm_signals']['fallbacks']} |",
          f"| column checks the model omitted (added by the safety net) | {result['llm_signals']['checks_omitted_by_model']} |",
          f"| advice repaired by the number-grounding check | {result['llm_signals']['grounding_repairs']} |", "",
          "| gate | status |", "|---|---|"] + [f"| {g} | {'PASS' if ok else 'FAIL'} |" for g, ok in gates.items()]
    flagged = [(k, p) for k, p in pipeline.items() if p["needs_review"] or p["failed_checks"]]
    if flagged:
        md += ["", "### Pipeline not clean (flagged for review or failed checks)"]
        for k, p in flagged:
            md.append(f"- `{k}`: retries {p['retries']}, failed checks {p['failed_checks']}, reasons: {p['review_reasons']}")
    misses = [(k, v) for k, v in per.items() if v["fp"] or v["fn"]]
    if misses:
        md += ["", "### Errors"]
        for k, v in misses:
            md.append(f"- `{k}`: missed {v['fn']} | false positives {v['fp']}")
    text = "\n".join(md)
    print(text)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(result, indent=2, default=str))
    if a.markdown_out:
        Path(a.markdown_out).write_text(text)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    sys.exit(main())