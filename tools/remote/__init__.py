"""Key-based SSH transport to the remote GPU host, and the lane/slot queue of a registered study (plan v2, Phase 0).

config       reads the untracked <project root>/remote.toml: host alias, remote root, python, GPU lanes and slots
ssh          OpenSSH subprocess with BatchMode=yes and StrictHostKeyChecking=yes, so no password prompt can occur
agent        stdlib-only script sent with every command; it does the remote half of each operation
sync         push (git archive <commit> plus listed data files) and pull (remote manifest -> tar -> local SHA-256)
verify_sync  hash check of a pulled study mirror against the remote (generalises tools/verify_remote_sync.py)
queue        lane/slot workers of one study: per-study lock, supervisor.json, events.jsonl, STOP

Command line, from the project root: python -m tools.remote --help (README.md has the setup). The retired paramiko
tools (tools/remote_study.py, tools/remote_supplement.py, tools/verify_remote_sync.py) are never imported.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class RemoteError(RuntimeError):
    """A remote operation failed or was refused."""


class ConfigError(ValueError):
    """remote.toml is missing or invalid."""
