"""Settings of the remote GPU host, read from the untracked <project root>/remote.toml.

The file names an OpenSSH Host alias whose key-based login is configured in ~/.ssh/config; it never holds
credentials. Keys such as password, passphrase, identity_file, user or port are refused: they belong in the
~/.ssh/config Host entry, which this code never reads. Template: tools/remote/remote.toml.example.
"""
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from . import ROOT, ConfigError

DEFAULT_PATH = ROOT / 'remote.toml'
EXAMPLE = 'tools/remote/remote.toml.example'
ALIAS = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]*')
REMOTE_PATH = re.compile(r'/[A-Za-z0-9_.+/-]*')
RELATIVE_PATH = re.compile(r'[A-Za-z0-9_.+-][A-Za-z0-9_.+/-]*')
COMMAND = re.compile(r'[A-Za-z0-9_.+-]+')
CREDENTIAL_WORDS = ('pass', 'secret', 'token', 'identity', 'private', 'credential')
SSH_CONFIG_KEYS = ('user', 'username', 'port', 'hostname', 'key', 'proxyjump', 'proxycommand')
TOP_KEYS = {'host', 'remote_root', 'python', 'agent_python', 'ssh', 'connect_timeout', 'exclude', 'gpu', 'disk'}
GPU_KEYS = {'lanes', 'slots_per_lane', 'memory_cap_gib', 'device_memory_gib', 'free_reserve_gib'}
DISK_KEYS = {'min_free_gib'}


@dataclass(frozen=True)
class RemoteConfig:
    host: str                      # Host alias in ~/.ssh/config
    remote_root: str               # absolute POSIX path of the project mirror on the remote host
    python: str                    # absolute remote interpreter for research.py (default <remote_root>/.venv-linux/bin/python)
    lanes: tuple                   # CUDA device indices the queue may use
    slots_per_lane: int            # concurrent worker processes per device
    memory_cap_gib: float          # allocator cap per worker slot (passed to workers as STUDYKIT_MEMORY_CAP_GIB)
    agent_python: str = 'python3'  # stdlib interpreter for transport operations (works before the venv exists)
    ssh: str = 'ssh'               # local OpenSSH client
    connect_timeout: int = 20
    device_memory_gib: float = None
    gpu_free_reserve_gib: float = 1.0
    min_free_disk_gib: float = 300.0
    exclude: tuple = ()            # git pathspecs left out of code pushes
    source: str = None             # file the settings came from

    def remote_path(self, *parts):
        """Absolute POSIX path under remote_root; every part must be relative and free of '..'."""
        rel = PurePosixPath(*parts) if parts else PurePosixPath('.')
        if rel.is_absolute() or '..' in rel.parts:
            raise ConfigError(f'remote path parts must be relative without "..": {parts!r}')
        return str(PurePosixPath(self.remote_root, rel))

    @property
    def slots(self):
        return [(lane, slot) for lane in self.lanes for slot in range(self.slots_per_lane)]

    def as_dict(self):
        value = asdict(self)
        value.update(lanes=list(self.lanes), exclude=list(self.exclude))
        return value


def _refuse_credentials(table, where='remote.toml'):
    for key, value in table.items():
        lower = key.lower().replace('-', '_')
        if any(word in lower for word in CREDENTIAL_WORDS) or lower.replace('_', '') in SSH_CONFIG_KEYS:
            raise ConfigError(f'{where} must not contain {key!r}: credentials and connection details (user, port, '
                              'key, proxy) belong in the ~/.ssh/config Host entry of the alias')
        if isinstance(value, dict):
            _refuse_credentials(value, f'{where} [{key}]')


def _number(value, name, minimum, strict=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f'{name} must be a number')
    if value < minimum or (strict and value == minimum):
        raise ConfigError(f'{name} must be {">" if strict else ">="} {minimum}')
    return float(value)


def _remote_absolute(value, name):
    if not isinstance(value, str) or not REMOTE_PATH.fullmatch(value):
        raise ConfigError(f'{name} must be an absolute POSIX path (letters, digits, . _ + - /), got {value!r}')
    path = PurePosixPath(value)
    if '..' in path.parts or str(path) != value.rstrip('/') or str(path) == '/':
        raise ConfigError(f'{name} must be a normalised absolute path below /, got {value!r}')
    return str(path)


def parse(data, source=None):
    """RemoteConfig from the parsed TOML table; raises ConfigError with the first problem found."""
    if not isinstance(data, dict):
        raise ConfigError('remote.toml must hold a table')
    _refuse_credentials(data)
    gpu, disk = data.get('gpu', {}), data.get('disk', {})
    for table, allowed, where in ((data, TOP_KEYS, 'remote.toml'), (gpu, GPU_KEYS, '[gpu]'), (disk, DISK_KEYS, '[disk]')):
        if not isinstance(table, dict):
            raise ConfigError(f'{where} must be a table')
        unknown = sorted(set(table) - allowed)
        if unknown:
            raise ConfigError(f'unknown keys in {where}: {unknown}')
    host = data.get('host')
    if not isinstance(host, str) or not ALIAS.fullmatch(host):
        raise ConfigError(f'host must be the Host alias from ~/.ssh/config (letters, digits, . _ -; no user@, no '
                          f'spaces, not starting with "-"), got {host!r}')
    root = _remote_absolute(data.get('remote_root'), 'remote_root')
    python = data.get('python', '.venv-linux/bin/python')
    if isinstance(python, str) and python.startswith('/'):
        python = _remote_absolute(python, 'python')
    elif isinstance(python, str) and RELATIVE_PATH.fullmatch(python) and '..' not in PurePosixPath(python).parts:
        python = str(PurePosixPath(root, python))
    else:
        raise ConfigError(f'python must be an absolute remote path or one relative to remote_root, got {python!r}')
    agent_python = data.get('agent_python', 'python3')
    if not isinstance(agent_python, str) or not (COMMAND.fullmatch(agent_python) or agent_python.startswith('/')):
        raise ConfigError(f'agent_python must be a command name or an absolute path, got {agent_python!r}')
    if agent_python.startswith('/'):
        agent_python = _remote_absolute(agent_python, 'agent_python')
    ssh = data.get('ssh', 'ssh')
    if not isinstance(ssh, str) or not ssh.strip():
        raise ConfigError('ssh must name the local OpenSSH client')
    timeout = data.get('connect_timeout', 20)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 300:
        raise ConfigError('connect_timeout must be an integer number of seconds from 1 to 300')
    exclude = data.get('exclude', [])
    if not isinstance(exclude, list) or not all(isinstance(e, str) and RELATIVE_PATH.fullmatch(e) and
                                                '..' not in PurePosixPath(e).parts for e in exclude):
        raise ConfigError('exclude must be a list of project-relative paths')
    lanes = gpu.get('lanes')
    if (not isinstance(lanes, list) or not lanes or len(set(lanes)) != len(lanes)
            or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in lanes)):
        raise ConfigError('[gpu] lanes must be a non-empty list of distinct CUDA device indices')
    slots = gpu.get('slots_per_lane', 1)
    if isinstance(slots, bool) or not isinstance(slots, int) or not 1 <= slots <= 16:
        raise ConfigError('[gpu] slots_per_lane must be an integer from 1 to 16')
    if 'memory_cap_gib' not in gpu:
        raise ConfigError('[gpu] memory_cap_gib is required (allocator cap per worker slot)')
    cap = _number(gpu['memory_cap_gib'], '[gpu] memory_cap_gib', 0)
    device = gpu.get('device_memory_gib')
    if device is not None:
        device = _number(device, '[gpu] device_memory_gib', 0)
        if slots * cap > device:
            raise ConfigError(f'[gpu] slots_per_lane x memory_cap_gib = {slots * cap:g} GiB exceeds '
                              f'device_memory_gib = {device:g}')
    reserve = _number(gpu.get('free_reserve_gib', 1.0), '[gpu] free_reserve_gib', 0, strict=False)
    min_free = _number(disk.get('min_free_gib', 300.0), '[disk] min_free_gib', 0, strict=False)
    return RemoteConfig(host=host, remote_root=root, python=python, lanes=tuple(lanes), slots_per_lane=slots,
                        memory_cap_gib=cap, agent_python=agent_python, ssh=ssh, connect_timeout=timeout,
                        device_memory_gib=device, gpu_free_reserve_gib=reserve, min_free_disk_gib=min_free,
                        exclude=tuple(exclude), source=None if source is None else str(source))


def load(path=None):
    """Read and validate remote.toml (default: the project root's)."""
    path = Path(path) if path else DEFAULT_PATH
    if not path.is_file():
        raise ConfigError(f'{path} is missing: copy {EXAMPLE} to remote.toml at the project root and fill in the '
                          'Host alias and the remote root (see tools/remote/README.md)')
    try:
        with path.open('rb') as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f'{path}: {error}') from None
    return parse(data, source=path)
