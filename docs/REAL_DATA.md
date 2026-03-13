# Real-data test: Telco Customer Churn

Dataset: IBM sample data "Telco Customer Churn" (Kaggle, `WA_Fn-UseC_-Telco-Customer-Churn.csv`, 7043 rows, 21 columns).
Not included in this repo (license: "Data files (c) Original Authors"). Download it from Kaggle and run:

```bash
python scripts/analyze_csv.py path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv "churn prediction" Churn
```

Rule-based mode, current commit.

## What happened

1. **First run: a miss.** DataScout said "No data quality issues detected". It had not seen that `TotalCharges` is stored as
   text and holds 11 blank values (customers with `tenure = 0`). The golden set could not show this, because I wrote it.
2. **Fix.** Blank strings now count as missing, and a text column whose non-blank values all parse as numbers becomes numeric.
   Nothing is converted if even one value fails, and ID-like names or codes with leading zeros stay text.
   Every change is written into the report warnings. Two regression tests cover it.
3. **Second run.** `TotalCharges` is numeric with 11 missing values and the report warns about it.

## Result after the fix

| | |
|---|---|
| Findings | 0 (quality score 100) |
| Report warnings | 11 blank `TotalCharges` values treated as missing, column converted to numeric |

Telco is a fairly clean dataset, so zero findings is plausible, but I checked these points by hand with pandas:

- 0 exact duplicate rows (22 if `customerID` is ignored, 0.3%, below the 1% threshold).
- Churn is 26.5% (above the 20% imbalance threshold, so no finding).
- The 11 missing `TotalCharges` are 0.16%, below the 1% missing threshold. They stay in the cleaned file as missing values.

## What DataScout still does not cover here

- `customerID` is a unique ID and is left in the feature set. There is no check for ID columns.
- Duplicates are exact-row only, so rows that differ only by ID are not seen.
- Missing values below 1% are not fixed. They are only mentioned in the load warning.

## Summary

One real dataset, one real miss found and fixed, and a clear list of what is still not covered.
This is one dataset checked by one person, so it does not prove general accuracy.

## Drift on the same data

Reference = Telco, current = Telco.

| Test | Result |
|---|---|
| 200 random 50/50 splits of Telco | 0 false drift flags |
| `MonthlyCharges` x 1.25 and 30% of contracts switched to month-to-month | both flagged (`MonthlyCharges` KS D 0.26, `Contract` TVD 0.14), statistics re-verified |
| Customers with tenure ≤ 12 months vs. longer | 15 of 20 columns flagged. This is a real difference between two groups, not a data problem |

The last row shows what drift means: the two groups really differ. The tool reports it, and it is up to the human to
decide whether it matters.
