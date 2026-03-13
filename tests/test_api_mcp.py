import io
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from datascout.api import app as api


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASCOUT_DATA_DIR", str(tmp_path))
    return TestClient(api.app)


def csv_bytes(df):
    return df.to_csv(index=False).encode()


def sample():
    r = np.random.default_rng(0)
    df = pd.DataFrame({"x": r.normal(size=300), "y": r.integers(0, 2, 300)})
    df["leak"] = df["y"] * 2.0
    df.loc[:40, "x"] = np.nan
    return df


def wait(client, jid):
    for _ in range(100):
        s = client.get(f"/jobs/{jid}").json()
        if s["status"] in ("done", "failed"):
            return s
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_full_flow(client):
    r = client.post("/analyze", files={"file": ("d.csv", csv_bytes(sample()), "text/csv")},
                    data={"task": "demo", "target": "y"})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    assert wait(client, jid)["status"] == "done"
    rep = client.get(f"/report/{jid}").json()
    assert {f["issue_type"] for f in rep["findings"]} >= {"leakage", "missing_values"}
    clean = pd.read_csv(io.BytesIO(client.get(f"/download/{jid}").content))
    assert "leak" not in clean.columns and clean["x"].isna().sum() == 0


@pytest.mark.parametrize("name,payload,data,code", [
    ("a.txt", b"hi", {"task": "t"}, 415),
    ("a.csv", b"", {"task": "t"}, 422),
    ("a.csv", b"a,b\n1,2\n", {"task": "  "}, 422),
    ("a.csv", b"a,b\n1,2\n", {"task": "t", "target": "zzz"}, 422),
])
def test_bad_requests(client, name, payload, data, code):
    assert client.post("/analyze", files={"file": (name, payload)}, data=data).status_code == code


def test_job_lifecycle_errors(client):
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/report/nope").status_code == 404


def test_oversize_rejected(client, monkeypatch):
    monkeypatch.setenv("DATASCOUT_MAX_UPLOAD_MB", "0")
    r = client.post("/analyze", files={"file": ("a.csv", b"a,b\n1,2\n")}, data={"task": "t"})
    assert r.status_code == 413


def test_mcp_tools_and_path_safety(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASCOUT_MCP_ROOT", str(tmp_path))
    sample().to_csv(tmp_path / "d.csv", index=False)
    from datascout.mcp_server import server as m
    prof = m.profile_dataset("d.csv")
    assert {p["name"] for p in prof} == {"x", "y", "leak"}
    q = m.check_quality("d.csv", "y")
    assert any(f["issue_type"] == "leakage" for f in q["findings"])
    out = m.prepare_data("d.csv", "demo", "y", "clean.csv")
    assert (tmp_path / "clean.csv").exists() and out["report"]["findings"]
    with pytest.raises(ValueError):
        m.profile_dataset("/etc/passwd")
    with pytest.raises(ValueError):
        m.profile_dataset("../../etc/passwd")


def test_mcp_registered_tools():
    import asyncio
    from datascout.mcp_server import server as m
    tools = asyncio.run(m.mcp.list_tools())
    names = {t.name for t in (tools.tools if hasattr(tools, "tools") else tools)}
    assert names == {"profile_dataset", "check_quality", "check_leakage", "compute_stats", "check_drift", "prepare_data"}


def test_rejected_uploads_leave_nothing_on_disk(client, tmp_path):
    client.post("/analyze", files={"file": ("a.csv", b"a,b\n1,2\n")}, data={"task": "t", "target": "zzz"})
    client.post("/analyze", files={"file": ("a.csv", b"")}, data={"task": "t"})
    jobs = tmp_path / "jobs"
    assert not jobs.exists() or not any(jobs.iterdir())
