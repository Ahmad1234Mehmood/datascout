import os

# Tests and the CI gate must be deterministic: never let a developer's local .env switch them to a real LLM.
os.environ["DATASCOUT_LLM_PROVIDER"] = "heuristic"
os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
os.environ.pop("LANGFUSE_SECRET_KEY", None)
