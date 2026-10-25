from pathlib import Path
import sys
import shlex
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from mechanism_base import *
from mechanism_transport import connect,execute,REMOTE,REMOTE_ROOT
c=connect()
try:
    program='import pathlib,hashlib,json;p=pathlib.Path('+repr(REMOTE+'/neurons')+');print(json.dumps({f.name:{"bytes":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in p.iterdir() if f.is_file()}))'
    expected=json.loads(execute(c,shlex.join([REMOTE_ROOT+'/.venv/bin/python','-c',program])))
    differences=[name for name,value in expected.items() if not (SESSION/'neurons'/name).exists() or digest(SESSION/'neurons'/name)!=value['sha256']]
    write(SESSION/'integrity/neuron-index.json',dict(transport_verified=not differences,mismatches=differences,files=expected,unix=time.time()))
    assert not differences,differences
    print({'shared_neuron_index_verified':True,'files':len(expected)})
finally:c.close()
