from __future__ import annotations

from collections import defaultdict


def norm_col(issue_type: str, column: str | None) -> str | None:
    if column is None:
        return None
    if issue_type == "multicollinearity":
        return "~".join(sorted(column.split("~")))      # pair order is irrelevant
    return column


def to_set(items) -> set[tuple]:
    return {(i["issue_type"], norm_col(i["issue_type"], i["column"])) for i in items}


def score_dataset(expected: list[dict], found: list[dict]) -> dict:
    exp, got = to_set(expected), to_set(found)
    return {"tp": sorted(exp & got, key=str), "fp": sorted(got - exp, key=str), "fn": sorted(exp - got, key=str)}


def aggregate(per_dataset: dict[str, dict]) -> dict:
    tp = sum(len(d["tp"]) for d in per_dataset.values())
    fp = sum(len(d["fp"]) for d in per_dataset.values())
    fn = sum(len(d["fn"]) for d in per_dataset.values())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    by_type = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    for d in per_dataset.values():
        for k in ("tp", "fn", "fp"):
            for issue, _ in d[k]:
                by_type[issue][k] += 1
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "by_issue_type": {t: {**v, "recall": round(v["tp"] / (v["tp"] + v["fn"]), 3) if v["tp"] + v["fn"] else None}
                              for t, v in sorted(by_type.items())}}
