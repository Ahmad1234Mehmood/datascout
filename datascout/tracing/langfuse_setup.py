"""Per-node tracing and error containment, with optional Langfuse spans."""
from __future__ import annotations

import logging
import time
from typing import Callable

from ..config import get_settings
from ..models.report import TraceEvent

log = logging.getLogger("datascout.trace")


def _langfuse_wrap(name: str, fn: Callable) -> Callable:
    if not get_settings().langfuse_enabled:
        return fn
    try:
        from langfuse import observe
        return observe(name=name, capture_input=False, capture_output=False)(fn)
    except Exception as e:  # tracing must never break the pipeline
        log.warning("Langfuse unavailable, tracing disabled for %s: %s", name, e)
        return fn


def _annotate_span(name: str, note: str, status: str) -> None:
    if not get_settings().langfuse_enabled:
        return
    try:
        from langfuse import get_client
        get_client().update_current_span(metadata={"node": name, "status": status}, output=note or None)
    except Exception:
        pass


def traced(name: str):
    def deco(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        def inner(state: dict) -> dict:
            out = fn(state)
            _annotate_span(name, out.get("_note", ""), "ok")
            return out

        wrapped = _langfuse_wrap(name, inner)

        def run(state: dict) -> dict:
            t0 = time.perf_counter()
            try:
                out = dict(wrapped(state))
                status, note = "ok", str(out.pop("_note", ""))
            except Exception as e:
                log.exception("node %s failed", name)
                status, note = "error", f"{type(e).__name__}: {e}"
                out = {"fatal": True, "errors": list(state.get("errors", [])) + [f"{name}: {note}"]}
            ev = TraceEvent(node=name, seconds=round(time.perf_counter() - t0, 4), status=status, note=note)
            out["trace"] = list(state.get("trace", [])) + [ev.model_dump()]
            log.info("node=%s status=%s seconds=%.3f %s", name, status, ev.seconds, note)
            return out

        run.__name__ = f"traced_{name}"
        return run
    return deco


def flush_langfuse() -> None:
    if not get_settings().langfuse_enabled:
        return
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        pass
