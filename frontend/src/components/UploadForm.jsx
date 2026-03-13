import { useRef, useState } from "react";

// First line of a CSV -> column names (handles simple quoted fields). Used only to suggest target columns.
function parseHeader(text) {
  const line = text.split(/\r?\n/)[0] || "";
  const out = [];
  const re = /("([^"]|"")*"|[^,]*)(,|$)/g;
  let m;
  while ((m = re.exec(line)) && m[0] !== "") {
    out.push(m[1].replace(/^"|"$/g, "").replace(/""/g, '"').trim());
    if (m[3] === "") break;
  }
  return out.filter(Boolean);
}

export default function UploadForm({ onSubmit, busy }) {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [task, setTask] = useState("");
  const [target, setTarget] = useState("");
  const [drag, setDrag] = useState(false);
  const input = useRef(null);

  async function pick(f) {
    if (!f) return;
    setFile(f);
    setTarget("");
    if (f.name.toLowerCase().endsWith(".csv")) {
      const head = await f.slice(0, 65536).text();
      setColumns(parseHeader(head));
    } else {
      setColumns([]);
    }
  }

  const ready = file && task.trim() && !busy;

  return (
    <form
      className="card form"
      onSubmit={(e) => {
        e.preventDefault();
        if (ready) onSubmit({ file, task: task.trim(), target: target.trim() });
      }}
    >
      <div
        className={`drop ${drag ? "drag" : ""}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          pick(e.dataTransfer.files?.[0]);
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      >
        <input
          ref={input}
          type="file"
          accept=".csv,.xlsx,.xls"
          data-testid="file-input"
          hidden
          onChange={(e) => pick(e.target.files?.[0])}
        />
        {file ? (
          <>
            <strong>{file.name}</strong>
            <span className="muted"> · {(file.size / 1024).toFixed(0)} KB · click to change</span>
          </>
        ) : (
          <>
            <strong>Drop a CSV or XLSX here</strong>
            <span className="muted"> or click to browse (max 25 MB)</span>
          </>
        )}
      </div>

      <div className="row">
        <label>
          ML task
          <input
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="e.g. churn prediction"
            maxLength={500}
            required
          />
        </label>
        <label>
          Target column <span className="muted">(enables leakage and imbalance checks)</span>
          <input
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            list="columns"
            placeholder={columns.length ? "choose a column" : "optional"}
          />
          <datalist id="columns">
            {columns.map((c) => (
              <option key={c} value={c} />
            ))}
          </datalist>
        </label>
      </div>

      <button className="primary" type="submit" disabled={!ready}>
        {busy ? "Analysing…" : "Analyse dataset"}
      </button>
    </form>
  );
}
