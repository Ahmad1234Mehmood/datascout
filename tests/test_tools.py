import numpy as np
import pandas as pd
import pytest

from datascout.models.findings import ActionType, IssueType, PrepAction
from datascout.tools import quality as q
from datascout.tools.preparation import apply_action, run_plan
from datascout.tools.profiling import DataLoadError, detect_semantic_type, load_dataframe, profile_columns
from datascout.tools.validation import (check_numeric_grounding, verify_finding, verify_steps)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def base(rng, n=400):
    return pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                         "cat": rng.choice(["x", "y", "z"], n), "y": rng.integers(0, 2, n)})


def test_semantic_types(rng):
    df = base(rng)
    df["customer_id"] = np.arange(len(df))
    df["const"] = 7
    assert detect_semantic_type(df["a"]) == "numeric"
    assert detect_semantic_type(df["cat"]) == "categorical"
    assert detect_semantic_type(df["customer_id"]) == "id"
    assert detect_semantic_type(df["const"]) == "constant"
    p = {x.name: x for x in profile_columns(df)}
    assert p["a"].mean is not None and p["cat"].top_values


def test_missing_flagged_and_verified(rng):
    df = base(rng)
    df.loc[:79, "a"] = np.nan            # 20%
    f = q.check_missing(df, "a")[0]
    assert f.evidence["missing_pct"] == 20.0 and f.severity.value == "high"
    assert verify_finding(df, f).passed
    assert q.check_missing(base(rng), "a") == []
    df2 = base(rng); df2.loc[:1, "a"] = np.nan   # 0.5% -> below flag threshold
    assert q.check_missing(df2, "a") == []


def test_outliers_and_skew_negative(rng):
    df = base(rng)
    df.loc[:9, "a"] = 50.0
    f = q.detect_outliers(df, "a")[0]
    assert f.evidence["outlier_count"] == 10 and verify_finding(df, f).passed
    skewed = pd.DataFrame({"v": rng.exponential(size=2000)})
    assert q.detect_outliers(skewed, "v") == []     # natural skew is not an outlier problem


def test_leakage_classification_regression_and_categorical(rng):
    df = base(rng)
    df["leak"] = df["y"] * 3 + rng.normal(scale=0.05, size=len(df))
    df["leak_cat"] = np.where(df["y"] == 1, "yes", "no")
    found = {f.column: f for f in q.check_leakage(df, target="y")}
    assert set(found) == {"leak", "leak_cat"}
    assert all(verify_finding(df, f).passed for f in found.values())
    reg = pd.DataFrame({"x": rng.normal(size=300)})
    reg["t"] = reg["x"] * 2 + rng.normal(scale=0.01, size=300)
    assert q.check_leakage(reg, target="t")[0].column == "x"


def test_imbalance_duplicates_collinearity_constant(rng):
    df = base(rng)
    df["y"] = 0
    df.loc[:9, "y"] = 1                  # exactly 2.5% minority
    f = q.check_imbalance(df, target="y")[0]
    assert f.severity.value == "critical" and verify_finding(df, f).passed
    d = pd.concat([base(rng), base(rng).iloc[:60]], ignore_index=True)
    d = pd.concat([d, d.iloc[:60]], ignore_index=True)
    fd = q.check_duplicates(d)[0]
    assert verify_finding(d, fd).passed
    c = base(rng); c["a2"] = c["a"] * 2 + rng.normal(scale=0.01, size=len(c))
    fc = q.check_correlation(c, target="y")[0]
    assert fc.column == "a~a2" and verify_finding(c, fc).passed
    k = base(rng); k["k"] = 1
    assert q.check_constant(k, "k")[0].issue_type == IssueType.constant_column


def test_prep_actions_and_verification(rng):
    df = base(rng)
    df.loc[:49, "a"] = np.nan
    df.loc[100:104, "b"] = 99.0
    findings = q.check_missing(df, "a") + q.detect_outliers(df, "b")
    plan = [PrepAction(action=ActionType.impute_median, column="a"),
            PrepAction(action=ActionType.clip_outliers, column="b")]
    clean, steps, notes = run_plan(df, plan, "y", findings)
    assert not notes and clean["a"].isna().sum() == 0 and clean["b"].max() < 99
    assert all(v.passed for v in verify_steps(df, clean, steps, "y"))
    # a broken fix is caught
    bad = verify_steps(df, df, steps, "y")
    assert not all(v.passed for v in bad)


def test_prep_guards(rng):
    df = base(rng)
    with pytest.raises(Exception):
        apply_action(df, PrepAction(action=ActionType.drop_column, column="y"), "y", "k")
    _, steps, notes = run_plan(df, [PrepAction(action=ActionType.impute_median, column="cat")], "y", [])
    assert not steps and "numeric" in notes[0]
    _, steps, notes = run_plan(df, [PrepAction(action=ActionType.drop_column, column="ghost")], "y", [])
    assert "no longer exists" in notes[0]


def test_numeric_grounding_catches_hallucinated_numbers(rng):
    df = base(rng); df.loc[:29, "a"] = np.nan       # 7.5%
    f = q.check_missing(df, "a")[0]
    assert check_numeric_grounding("About 7.5% (30 rows) are missing.", f).passed
    bad = check_numeric_grounding("Roughly 30% are missing.", f)
    assert not bad.passed and bad.claimed == ["30%"]      # 30 is the COUNT, not the percentage
    assert not check_numeric_grounding("Only 12 rows are missing.", f).passed


def test_load_errors(tmp_path):
    p = tmp_path / "bad.csv"; p.write_bytes(b"")
    with pytest.raises(DataLoadError):
        load_dataframe(p)
    q_ = tmp_path / "x.txt"; q_.write_text("hi")
    with pytest.raises(DataLoadError):
        load_dataframe(q_)


def test_outliers_cannot_mask_themselves_by_inflating_skew(rng):
    """Moment-skew from ~1% typos must not pick the log scale and hide them."""
    x = rng.integers(1, 72, 845).astype(float)
    x[:10] = 700.0
    df = pd.DataFrame({"tenure": x})
    f = q.detect_outliers(df, "tenure")
    assert f and f[0].evidence["method"] == "iqr" and f[0].evidence["outlier_count"] == 10
    assert verify_finding(df, f[0]).passed


def test_log_scale_still_chosen_for_genuinely_skewed_data(rng):
    df = pd.DataFrame({"v": rng.lognormal(10, 0.5, 600)})
    df.loc[:7, "v"] *= 60
    f = q.detect_outliers(df, "v")
    assert f and f[0].evidence["method"] == "iqr_log1p" and verify_finding(df, f[0]).passed


def test_semicolon_csv_is_loaded_as_table(tmp_path):
    from datascout.tools.profiling import load_dataframe
    p = tmp_path / "bank.csv"
    p.write_text("age;job;y\n30;admin;no\n41;tech;yes\n", encoding="utf-8")
    df = load_dataframe(p)
    assert list(df.columns) == ["age", "job", "y"] and len(df) == 2


def test_blank_text_becomes_missing_and_numeric_text_is_converted(tmp_path):
    from datascout.tools.profiling import load_dataframe
    p = tmp_path / "t.csv"
    p.write_text("id,total,code,zip,label\n1,10.5,a,01234,x\n2, ,b,05678,y\n3,7,c,09999,x\n", encoding="utf-8")
    df = load_dataframe(p)
    assert str(df["total"].dtype).startswith("float") and df["total"].isna().sum() == 1
    assert any("'total'" in n and "converted" in n for n in df.attrs["load_notes"])
    assert any("blank" in n for n in df.attrs["load_notes"])


def test_mixed_text_column_is_not_coerced(tmp_path):
    from datascout.tools.profiling import load_dataframe
    p = tmp_path / "m.csv"
    p.write_text("a,b\n1,x\n2,3\n3,4\n", encoding="utf-8")
    df = load_dataframe(p)
    assert not str(df["b"].dtype).startswith(("float", "int"))


def test_leading_zero_codes_stay_text():
    import pandas as pd
    from datascout.tools.profiling import normalize_text_columns
    df, notes = normalize_text_columns(pd.DataFrame({"code": ["01234", "05678", "09999"]}))
    assert df["code"].tolist() == ["01234", "05678", "09999"] and notes == []
