import { downloadUrl } from "../api.js";

const GROUPS = [
  ["finding", "Findings re-computed from the raw data"],
  ["step", "Fixes verified on the cleaned data"],
  ["invariant", "Safety invariants"],
  ["judge", "Judge: fix is appropriate"],
  ["grounding", "Advice numbers grounded in evidence"],
];

const fmt = (v) => (typeof v === "object" && v !== null ? JSON.stringify(v) : String(v));
const label = (s) => s.replaceAll("_", " ");

function Score({ before, after }) {
  return (
    <div className="score">
      <div>
        <span className="num">{before}</span>
        <span className="muted"> before</span>
      </div>
      <span aria-hidden="true">→</span>
      <div>
        <span className="num good">{after}</span>
        <span className="muted"> after</span>
      </div>
    </div>
  );
}

function Findings({ findings }) {
  if (!findings.length) return <p className="muted">No data quality issues detected.</p>;
  return (
    <ul className="list">
      {findings.map((f) => (
        <li key={`${f.issue_type}:${f.column}`} className="finding">
          <div className="head">
            <span className={`sev ${f.severity}`}>{f.severity}</span>
            <strong>{label(f.issue_type)}</strong>
            {f.column && <code>{f.column}</code>}
          </div>
          <p>{f.recommendation}</p>
          <dl className="evidence">
            {Object.entries(f.evidence).map(([k, v]) => (
              <div key={k}>
                <dt>{label(k)}</dt>
                <dd>{fmt(v)}</dd>
              </div>
            ))}
          </dl>
        </li>
      ))}
    </ul>
  );
}

function Steps({ steps }) {
  if (!steps.length) return <p className="muted">No fixes were needed.</p>;
  return (
    <div className="scroll">
      <table>
        <thead>
          <tr>
            <th>Action</th>
            <th>Column</th>
            <th>Parameters</th>
            <th>Before</th>
            <th>After</th>
          </tr>
        </thead>
        <tbody>
          {steps.map((s, i) => (
            <tr key={i}>
              <td>{label(s.action)}</td>
              <td>{s.column ? <code>{s.column}</code> : "all rows"}</td>
              <td className="small">{fmt(s.parameters)}</td>
              <td className="small">{fmt(s.before_stats)}</td>
              <td className="small">{fmt(s.after_stats)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Verification({ validations, provider }) {
  const judgeLabel = provider === "heuristic" ? "Rule-based judge: fix is appropriate" : "LLM judge: fix is appropriate";
  return (
    <div className="verify">
      {GROUPS.map(([prefix, baseTitle]) => {
        const title = prefix === "judge" ? judgeLabel : baseTitle;
        const items = validations.filter((v) => v.subject.startsWith(prefix + ":"));
        if (!items.length) return null;
        const failed = items.filter((v) => !v.passed);
        return (
          <details key={prefix} open={failed.length > 0}>
            <summary>
              <span className={`dot ${failed.length ? "bad" : "ok"}`} />
              {title}: {items.length - failed.length}/{items.length} passed
            </summary>
            <ul className="small">
              {(failed.length ? failed : items).map((v) => (
                <li key={v.subject}>
                  <code>{v.subject}</code> {v.passed ? "passed" : "failed"}
                  {v.detail ? ` – ${v.detail}` : ""}
                </li>
              ))}
            </ul>
          </details>
        );
      })}
    </div>
  );
}

function Trace({ trace }) {
  const max = Math.max(...trace.map((t) => t.seconds), 0.001);
  return (
    <ul className="trace">
      {trace.map((t, i) => (
        <li key={i}>
          <span className="node">{t.node}</span>
          <span className="bar">
            <span style={{ width: `${Math.max(3, (t.seconds / max) * 100)}%` }} className={t.status === "ok" ? "" : "bad"} />
          </span>
          <span className="small">{t.seconds}s</span>
          <span className="small muted">{t.note}</span>
        </li>
      ))}
    </ul>
  );
}

export default function Report({ report: r, jobId, onReset }) {
  return (
    <div className="report">
      <section className="card">
        <div className="between">
          <div>
            <h2>{r.dataset_name}</h2>
            <p className="muted">
              Task: {r.task_description}
              {r.target ? ` · target: ${r.target}` : ""} · mode: {r.llm_provider}
            </p>
          </div>
          <div className="actions">
            <a className="primary link" href={downloadUrl(jobId)}>
              Download cleaned CSV
            </a>
            <button onClick={onReset}>New analysis</button>
          </div>
        </div>
        {r.needs_human_review && (
          <div className="alert bad" role="alert">
            Flagged for human review.
            {r.unresolved.length > 0 && (
              <ul>
                {r.unresolved.map((u) => (
                  <li key={u}>{u}</li>
                ))}
              </ul>
            )}
          </div>
        )}
        <p>{r.summary}</p>
        <div className="stats">
          <Score before={r.quality_score_before} after={r.quality_score_after} />
          <div>
            <span className="num">
              {r.n_rows_before} → {r.n_rows_after}
            </span>
            <span className="muted"> rows</span>
          </div>
          <div>
            <span className="num">
              {r.n_cols_before} → {r.n_cols_after}
            </span>
            <span className="muted"> columns</span>
          </div>
          <div>
            <span className="num">{r.retries}</span>
            <span className="muted"> retries</span>
          </div>
        </div>
      </section>

      <section className="card">
        <h3>Findings ({r.findings.length})</h3>
        <Findings findings={r.findings} />
      </section>

      <section className="card">
        <h3>Fixes applied ({r.preparation_steps.length})</h3>
        <Steps steps={r.preparation_steps} />
      </section>

      <section className="card">
        <h3>Verification</h3>
        <Verification validations={r.validations} provider={r.llm_provider} />
        {r.residual_findings.length > 0 && (
          <p className="small muted">
            Still present after cleaning (advisory or not auto-fixable):{" "}
            {r.residual_findings.map((f) => `${label(f.issue_type)}${f.column ? ` (${f.column})` : ""}`).join(", ")}
          </p>
        )}
      </section>

      {r.warnings.length > 0 && (
        <section className="card">
          <h3>Warnings</h3>
          <ul className="small">
            {r.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </section>
      )}

      <section className="card">
        <h3>Agent trace</h3>
        <Trace trace={r.trace} />
      </section>
    </div>
  );
}
