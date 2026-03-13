import io
import sys
import types

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from datascout.api import app as api
from datascout.models.drift import DriftFinding
from datascout.tools.drift import detect_drift, verify_drift
from datascout.tools.hf_loader import load_hf_dataset
from datascout.tools.profiling import DataLoadError


def frame(seed=0, n=600, shift=0.0, mix=0.5):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"amount": r.normal(100, 15, n) + shift,
                         "plan": np.where(r.random(n) < mix, "basic", "pro"),
                         "id": range(n)})


def test_same_distribution_is_not_drift():
    r = detect_drift(frame(0), frame(1))
    assert r.findings == [] and r.verified


def test_numeric_shift_is_found_and_verified():
    r = detect_drift(frame(0), frame(1, shift=15))
    f = {x.column: x for x in r.findings}
    assert "amount" in f and f["amount"].kind == "numeric" and f["amount"].statistic >= 0.3
    assert f["amount"].evidence["current_mean"] > f["amount"].evidence["reference_mean"] + 10
    assert r.verified and not r.verification_errors


def test_categorical_shift_is_found():
    r = detect_drift(frame(0, mix=0.5), frame(1, mix=0.8))
    f = {x.column: x for x in r.findings}
    assert "plan" in f and f["plan"].statistic_name == "total_variation_distance"


def test_tiny_shift_on_big_data_is_ignored():
    """Significant but irrelevant: the effect-size floor must stop it."""
    r = detect_drift(frame(0, n=20000), frame(1, n=20000, shift=0.5))
    assert not [f for f in r.findings if f.column == "amount"]


def test_verifier_catches_a_wrong_statistic():
    a, b = frame(0), frame(1, shift=15)
    f = detect_drift(a, b).findings[0]
    bad = DriftFinding(**{**f.model_dump(), "statistic": f.statistic + 0.05})
    ok, errors = verify_drift(a, b, [bad])
    assert not ok and errors


def test_too_few_rows_and_unmatched_columns():
    r = detect_drift(frame(0, n=10), frame(1, n=10))
    assert r.findings == [] and r.warnings
    b = frame(1).rename(columns={"plan": "tier"})
    r = detect_drift(frame(0), b)
    assert "plan" in r.columns_skipped and "tier" in r.columns_skipped


def test_drift_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASCOUT_DATA_DIR", str(tmp_path))
    c = TestClient(api.app)
    files = {"reference": ("a.csv", frame(0).to_csv(index=False).encode(), "text/csv"),
             "current": ("b.csv", frame(1, shift=15).to_csv(index=False).encode(), "text/csv")}
    r = c.post("/drift", files=files)
    assert r.status_code == 200 and r.json()["verified"] and r.json()["findings"][0]["column"] == "amount"
    bad = c.post("/drift", files={"reference": ("a.txt", b"x", "text/plain"), "current": files["current"]})
    assert bad.status_code == 415
    jobs = tmp_path / "jobs"
    assert not jobs.exists() or not any(jobs.iterdir())     # temp files are always removed


def test_mcp_new_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASCOUT_MCP_ROOT", str(tmp_path))
    d = frame(0)
    d["y"] = (np.arange(600) % 2)
    d["leak"] = d["y"] * 3.0
    d.to_csv(tmp_path / "a.csv", index=False)
    frame(1, shift=15).to_csv(tmp_path / "b.csv", index=False)
    from datascout.mcp_server import server as m
    assert m.compute_stats("a.csv", "amount")["rows"] == 600
    assert any(f["column"] == "leak" for f in m.check_leakage("a.csv", "y")["findings"])
    assert m.check_drift("a.csv", "b.csv")["findings"]
    with pytest.raises(ValueError):
        m.check_leakage("a.csv", "nope")
    with pytest.raises(ValueError):
        m.check_drift("../../etc/passwd", "b.csv")


# ------------------------------------------------------------------ Hugging Face (fake `datasets` module)

class _FakeDS:
    def __init__(self, df):
        self.df = df

    def to_pandas(self):
        return self.df


def _install_fake(monkeypatch, df, calls=None):
    def load_dataset(name, config=None, split=None):
        if calls is not None:
            calls.append((name, config, split))
        if name == "boom/boom":
            raise RuntimeError("404")
        return _FakeDS(df.copy())
    mod = types.ModuleType("datasets")
    mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", mod)


def test_hf_loader(monkeypatch):
    raw = pd.DataFrame({"a": [1, 2, 3], "b": ["1.5", " ", "2.5"], "nested": [[1], [2], [3]]})
    calls = []
    _install_fake(monkeypatch, raw, calls)
    df = load_hf_dataset("owner/name", "train", max_rows=100)
    assert calls == [("owner/name", None, "train[:100]")]
    assert "nested" not in df.columns and str(df["b"].dtype).startswith("float") and df["b"].isna().sum() == 1
    assert any("Hugging Face" in n for n in df.attrs["load_notes"])


@pytest.mark.parametrize("name", ["", "a b", "../x", "a/b/c", "x;rm"])
def test_hf_rejects_bad_names(name):
    with pytest.raises(DataLoadError):
        load_hf_dataset(name)


def test_hf_errors_are_clean(monkeypatch):
    _install_fake(monkeypatch, pd.DataFrame({"a": [1]}))
    with pytest.raises(DataLoadError, match="Could not load"):
        load_hf_dataset("boom/boom")
    monkeypatch.setitem(sys.modules, "datasets", None)      # simulates the package not being installed
    with pytest.raises(DataLoadError, match="pip install datasets"):
        load_hf_dataset("owner/name")


def test_analyze_hf_endpoint(tmp_path, monkeypatch):
    import time
    monkeypatch.setenv("DATASCOUT_DATA_DIR", str(tmp_path))
    r = np.random.default_rng(0)
    raw = pd.DataFrame({"x": r.normal(size=300), "y": r.integers(0, 2, 300)})
    raw["leak"] = raw["y"] * 2.0
    _install_fake(monkeypatch, raw)
    c = TestClient(api.app)
    resp = c.post("/analyze_hf", json={"dataset": "owner/name", "task": "demo", "target": "y"})
    assert resp.status_code == 202
    jid = resp.json()["job_id"]
    for _ in range(100):
        if c.get(f"/jobs/{jid}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    rep = c.get(f"/report/{jid}").json()
    assert any(f["issue_type"] == "leakage" for f in rep["findings"])
    assert any("Hugging Face" in w for w in rep["warnings"])
    assert c.post("/analyze_hf", json={"dataset": "owner/name", "task": "t", "target": "zzz"}).status_code == 422
    assert c.post("/analyze_hf", json={"dataset": "bad name", "task": "t"}).status_code == 422
