const BASE = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

async function parse(res) {
  if (res.ok) return res.json();
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    /* keep status text */
  }
  throw new Error(detail);
}

export async function startAnalysis({ file, task, target }) {
  const form = new FormData();
  form.append("file", file);
  form.append("task", task);
  if (target) form.append("target", target);
  let res;
  try {
    res = await fetch(`${BASE}/analyze`, { method: "POST", body: form });
  } catch {
    throw new Error(`Cannot reach the backend at ${BASE}. Is it running?`);
  }
  return parse(res);
}

export const getJob = (id) => fetch(`${BASE}/jobs/${id}`).then(parse);
export const getReport = (id) => fetch(`${BASE}/report/${id}`).then(parse);
export const downloadUrl = (id) => `${BASE}/download/${id}`;
export const getHealth = () => fetch(`${BASE}/health`).then(parse);
