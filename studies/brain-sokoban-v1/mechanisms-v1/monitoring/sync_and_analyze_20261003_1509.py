"""Finish this monitoring cycle's data retrieval, then analyze saved data only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
SESSION = ROOT / "studies/brain-sokoban-v1/mechanisms-v1"
MONITOR = SESSION / "monitoring"
RECEIPT = MONITOR / "sync-analysis-20261003-1509.json"
state = {"pid": os.getpid(), "started_unix": time.time(), "stages": [], "mode": "retrieve_and_cpu_analysis_only"}


def save():
    temporary = RECEIPT.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, RECEIPT)


try:
    for action in ("sync", "analyze"):
        stage = {"action": action, "started_unix": time.time()}
        state["stages"].append(stage)
        state["status"] = action
        save()
        with (MONITOR / (action + "-20261003-1509.log")).open("w", encoding="utf-8") as log:
            child = subprocess.Popen(
                [sys.executable, "-B", "-X", "utf8", "-u", "workflow.py", "study", "mechanisms", action],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
            stage["pid"] = child.pid
            save()
            stage["returncode"] = child.wait()
        stage["ended_unix"] = time.time()
        save()
        if stage["returncode"]:
            raise RuntimeError(f"{action} failed with exit code {stage['returncode']}; evidence retained, no retry")
    state["status"] = "complete"
except BaseException as error:
    state["status"] = "failed"
    state["error"] = repr(error)
    raise
finally:
    state["updated_unix"] = time.time()
    save()
