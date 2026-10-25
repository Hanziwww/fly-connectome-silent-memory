"""Retrieve the final closed jobs after the existing CPU analysis finishes."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
SESSION = ROOT / "studies/brain-sokoban-v1/mechanisms-v1"
MONITOR = SESSION / "monitoring"
PRIOR = MONITOR / "sync-analysis-20261003-1509.json"
RECEIPT = MONITOR / "sync-analysis-final-20261003-1607.json"
state = {"pid": os.getpid(), "started_unix": time.time(), "stages": [],
         "status": "waiting_for_previous_analysis", "mode": "retrieve_and_cpu_analysis_only"}


def save():
    state["updated_unix"] = time.time()
    temporary = RECEIPT.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, RECEIPT)


def alive(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_uint32()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise OSError(ctypes.get_last_error(), "Cannot check predecessor process")
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


try:
    save()
    while True:
        prior = json.loads(PRIOR.read_text(encoding="utf-8"))
        if prior["status"] == "complete":
            break
        if prior["status"] == "failed" or not alive(prior["pid"]):
            raise RuntimeError("Previous analysis failed or disappeared; no automatic retry")
        state["previous_status"] = prior["status"]
        save()
        time.sleep(10)
    if not (SESSION.parent / "STOP").exists():
        raise RuntimeError("Original local training STOP is missing")
    for action in ("sync", "analyze"):
        stage = {"action": action, "started_unix": time.time()}
        state["stages"].append(stage)
        state["status"] = action
        save()
        with (MONITOR / (action + "-final-20261003-1607.log")).open("w", encoding="utf-8") as log:
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
        if action == "sync":
            plan = json.loads((SESSION / "plan.json").read_text(encoding="utf-8"))
            missing = []
            for job in plan["jobs"]:
                path = SESSION / "receipts" / ("runs_" + job["id"] + ".json")
                receipt = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                if receipt.get("transport_verified") is not True or receipt.get("mismatches"):
                    missing.append(job["id"])
            state["unverified_jobs"] = missing
            save()
            if missing:
                raise RuntimeError("Final retrieval has unverified jobs; do not call it complete")
    state["status"] = "complete"
except BaseException as error:
    state["status"] = "failed"
    state["error"] = repr(error)
    raise
finally:
    save()
