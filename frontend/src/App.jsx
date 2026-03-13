import { useEffect, useRef, useState } from "react";
import { getHealth, getJob, getReport, startAnalysis } from "./api.js";
import UploadForm from "./components/UploadForm.jsx";
import Report from "./components/Report.jsx";

const POLL_MS = 800;

export default function App() {
  const [phase, setPhase] = useState("idle"); // idle | running | done | error
  const [jobId, setJobId] = useState(null);
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [health, setHealth] = useState(null);
  const timer = useRef(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: "unreachable" }));
    return () => clearInterval(timer.current);
  }, []);

  async function run(input) {
    setPhase("running");
    setError("");
    setReport(null);
    setElapsed(0);
    const started = Date.now();
    clearInterval(timer.current);
    timer.current = setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 500);
    try {
      const { job_id } = await startAnalysis(input);
      setJobId(job_id);
      for (;;) {
        const job = await getJob(job_id);
        if (job.status === "done") break;
        if (job.status === "failed") throw new Error(job.error || "Analysis failed");
        await new Promise((r) => setTimeout(r, POLL_MS));
      }
      setReport(await getReport(job_id));
      setPhase("done");
    } catch (e) {
      setError(e.message);
      setPhase("error");
    } finally {
      clearInterval(timer.current);
    }
  }

  function reset() {
    setPhase("idle");
    setReport(null);
    setError("");
  }

  return (
    <div className="page">
      <header className="top">
        <div>
          <h1>DataScout</h1>
          <p className="muted">Check a dataset before you train on it. Every finding and every fix is re-computed and verified.</p>
        </div>
        {health && (
          <span className={`chip ${health.status === "ok" ? "ok" : "bad"}`} title="Backend status">
            {health.status === "ok" ? `backend: ${health.llm_provider}` : "backend unreachable"}
          </span>
        )}
      </header>

      {(phase === "idle" || phase === "error" || phase === "running") && (
        <UploadForm onSubmit={run} busy={phase === "running"} />
      )}

      {phase === "running" && (
        <div className="card status" role="status">
          <span className="spinner" aria-hidden="true" />
          Agents working… {elapsed}s
          <span className="muted"> (a local LLM on CPU can take minutes)</span>
        </div>
      )}
      {phase === "error" && (
        <div className="card alert bad" role="alert">
          {error}
        </div>
      )}
      {phase === "done" && report && <Report report={report} jobId={jobId} onReset={reset} />}
    </div>
  );
}
