"""FastAPI service: analyze, drift, job status, report and download endpoints."""
from __future__ import annotations

import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..agents.graph import analyze_dataframe
from ..config import get_settings
from ..tools.drift import detect_drift
from ..tools.hf_loader import load_hf_dataset
from ..tools.profiling import DataLoadError, load_dataframe

log = logging.getLogger("datascout.api")
app = FastAPI(title="DataScout", version="0.1.0",
              description="Autonomous data quality & preparation agent")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:3000"],
                   allow_methods=["*"], allow_headers=["*"])

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
ALLOWED = {".csv", ".xlsx", ".xls"}


def _job_dir(job_id: str) -> Path:
    d = Path(get_settings().data_dir) / "jobs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _set(job_id: str, **kw) -> None:
    with _lock:
        _jobs[job_id].update(kw)


def _run(job_id: str, path: Path, task: str, target: Optional[str], df=None) -> None:
    _set(job_id, status="running")
    try:
        df = df if df is not None else load_dataframe(path)
        report, clean = analyze_dataframe(df, task, target, name=_jobs[job_id]["filename"])
        out = path.parent / "clean.csv"
        clean.to_csv(out, index=False)
        (path.parent / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        _set(job_id, status="done", summary=report.summary, needs_human_review=report.needs_human_review)
    except Exception as e:  # never leave a job stuck in "running"
        log.exception("job %s failed", job_id)
        _set(job_id, status="failed", error=f"{type(e).__name__}: {e}")


@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "llm_provider": s.llm_provider, "tracing": s.langfuse_enabled}


@app.post("/analyze", status_code=202)
async def analyze(background: BackgroundTasks, file: UploadFile = File(...), task: str = Form(...),
                  target: Optional[str] = Form(None)):
    s = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Upload .csv or .xlsx")
    task = task.strip()
    if not task or len(task) > 500:
        raise HTTPException(422, "task must be 1-500 characters, e.g. 'churn prediction'")

    job_id = uuid.uuid4().hex[:12]
    job_dir = _job_dir(job_id)
    path = job_dir / f"input{suffix}"
    limit, size = s.max_upload_mb * 1024 * 1024, 0
    try:
        with path.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File larger than {s.max_upload_mb} MB")
                fh.write(chunk)
        try:                               # fail fast on malformed files, before queueing work
            df = load_dataframe(path)
        except DataLoadError as e:
            raise HTTPException(422, str(e))
        target = (target or "").strip() or None
        if target and target not in df.columns:
            raise HTTPException(422, f"target '{target}' is not a column. Columns: {list(df.columns)[:50]}")
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)    # rejected uploads must not stay on disk
        raise

    with _lock:
        _jobs[job_id] = {"job_id": job_id, "status": "queued", "filename": file.filename, "task": task, "target": target}
    background.add_task(_run, job_id, path, task, target)
    return {"job_id": job_id, "status": "queued"}


def _get(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    return _get(job_id)


@app.get("/report/{job_id}")
def report(job_id: str):
    job = _get(job_id)
    if job["status"] == "failed":
        raise HTTPException(500, job.get("error", "job failed"))
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    return FileResponse(_job_dir(job_id) / "report.json", media_type="application/json")


@app.get("/download/{job_id}")
def download(job_id: str):
    job = _get(job_id)
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    return FileResponse(_job_dir(job_id) / "clean.csv", media_type="text/csv", filename=f"clean_{job['filename']}.csv")


class HFRequest(BaseModel):
    dataset: str
    task: str
    target: Optional[str] = None
    split: str = "train"
    config: Optional[str] = None


@app.post("/analyze_hf", status_code=202)
def analyze_hf(req: HFRequest, background: BackgroundTasks):
    task = req.task.strip()
    if not task or len(task) > 500:
        raise HTTPException(422, "task must be 1-500 characters, e.g. 'churn prediction'")
    try:
        df = load_hf_dataset(req.dataset, req.split, req.config)
    except DataLoadError as e:
        raise HTTPException(422, str(e))
    target = (req.target or "").strip() or None
    if target and target not in df.columns:
        raise HTTPException(422, f"target '{target}' is not a column. Columns: {list(df.columns)[:50]}")
    job_id = uuid.uuid4().hex[:12]
    path = _job_dir(job_id) / "input.csv"
    with _lock:
        _jobs[job_id] = {"job_id": job_id, "status": "queued", "filename": req.dataset, "task": task, "target": target}
    background.add_task(_run, job_id, path, task, target, df)
    return {"job_id": job_id, "status": "queued"}


async def _read_upload(file: UploadFile, dest: Path) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Upload .csv or .xlsx")
    limit, size = get_settings().max_upload_mb * 1024 * 1024, 0
    with dest.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"File larger than {get_settings().max_upload_mb} MB")
            fh.write(chunk)


@app.post("/drift")
async def drift(reference: UploadFile = File(...), current: UploadFile = File(...),
                target: Optional[str] = Form(None)):
    """Compare a reference dataset with a current one (for example train vs. new data)."""
    job_dir = _job_dir("drift_" + uuid.uuid4().hex[:10])
    try:
        paths = []
        for name, up in (("reference", reference), ("current", current)):
            dest = job_dir / f"{name}{Path(up.filename or '').suffix.lower()}"
            await _read_upload(up, dest)
            paths.append(dest)
        try:
            ref_df, cur_df = load_dataframe(paths[0]), load_dataframe(paths[1])
        except DataLoadError as e:
            raise HTTPException(422, str(e))
        return detect_drift(ref_df, cur_df, (target or "").strip() or None).model_dump(mode="json")
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
