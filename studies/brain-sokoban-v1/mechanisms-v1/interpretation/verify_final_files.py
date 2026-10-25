"""Rehash every immutable run artifact against its original remote manifest."""
import hashlib
import json
from pathlib import Path
import time

root = Path(__file__).resolve().parents[1]
started = time.time()
state = {"started_unix": started, "status": "running", "jobs": 0, "files": 0, "bytes": 0, "mismatches": []}
target = root / "integrity/final-file-verification.json"
for manifest in sorted((root / "runs").glob("*/manifest.json")):
    payload = json.loads(manifest.read_text(encoding="utf8"))
    for relative, expected in payload["files"].items():
        file = manifest.parent / relative
        h = hashlib.sha256()
        if file.exists():
            with file.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    h.update(chunk)
            actual_bytes = file.stat().st_size
        else:
            actual_bytes = -1
        if actual_bytes != expected["bytes"] or h.hexdigest() != expected["sha256"]:
            state["mismatches"].append(str(file.relative_to(root)))
        state["files"] += 1
        state["bytes"] += max(0, actual_bytes)
    state["jobs"] += 1
    state["updated_unix"] = time.time()
    target.write_text(json.dumps(state, indent=2), encoding="utf8")
state["elapsed_seconds"] = time.time() - started
state["status"] = "verified" if state["jobs"] == 55 and not state["mismatches"] else "failed"
state["transport_verified"] = state["status"] == "verified"
target.write_text(json.dumps(state, indent=2), encoding="utf8")
print(json.dumps(state))
raise SystemExit(0 if state["transport_verified"] else 1)
