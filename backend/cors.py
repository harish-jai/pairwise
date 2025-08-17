import os, json
from fastapi.middleware.cors import CORSMiddleware

raw = os.getenv("_ALLOWED_ORIGINS") or os.getenv("ALLOWED_ORIGINS") or ""
# Accept either comma-separated or JSON array
try:
    parsed = json.loads(raw) if raw.strip().startswith("[") else None
except json.JSONDecodeError:
    parsed = None

origins = (
    [s.strip() for s in raw.split(",") if s.strip()]
    if not parsed else
    [str(s).strip() for s in parsed]
)
if not origins:
    origins = ["http://localhost:3000"]  # sensible default for local dev

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
