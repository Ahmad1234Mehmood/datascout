## DataScout evaluation (ollama:qwen2.5:7b)

| metric | value |
|---|---|
| datasets | 26 (8 clean hard-negatives) |
| precision | 1.0 (28 TP, 0 FP) |
| recall | 1.0 (0 missed) |
| F1 | 1.0 |
| pipeline success (no review flag, all checks pass) | 0.9615 |
| datasets needing a retry | 1 |
| steps judged / rejected by the judge | 25 / 1 |
| rule-based fallbacks (model unusable) | 0 |
| column checks the model omitted (added by the safety net) | 191 |
| advice repaired by the number-grounding check | 2 |

| gate | status |
|---|---|
| precision >= 0.9 | PASS |
| recall >= 0.9 | PASS |
| clean datasets with false positives <= 0 | PASS |

### Pipeline not clean (flagged for review or failed checks)
- `missing_heavy_categorical`: retries 1, failed checks 0, reasons: ['missing_values:segment: judge disagrees after a retry, verified fix kept (Dropping a column that is only slightly incomplete (under 20% missing) loses information and is inappropriate.)']