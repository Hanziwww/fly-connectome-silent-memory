"""O12, circuit level (GPU host): the Phase 2 credibility tests C4-C6 (credibility.py, unchanged) on the P05 brain under
each structural variant of variants.py.

    .venv-linux/bin/python -B -X utf8 studies/brain-pathology-v1/code/phase2_variants.py

Per variant: a fresh Rig on P05, the variant matrix swapped in (variants.apply checks the healthy rebuild first), then
credibility.c4 and credibility.c5_c6 exactly as in Phase 2. 'default' must reproduce runs/phase2-credibility/p05
(checked by score_variants.py). Outputs runs/phase2-variants/<variant>/{variant,c4,c5_c6}.json; a finished variant is
skipped (resumable).
"""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-pathology-v1'
OUT = STUDY / 'runs' / 'phase2-variants'


def main():
    sys.path.insert(0, str(STUDY / 'code'))
    import numpy as np
    import torch
    import credibility
    import variants
    OUT.mkdir(parents=True, exist_ok=True)
    for name in variants.VARIANTS:
        out = OUT / name
        if (out / 'c5_c6.json').exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        t = time.time()
        rig = credibility.Rig(credibility.P05, out / 'meta')
        rec = variants.apply(rig.enc, name)
        rec['seconds_build'] = time.time() - t
        (out / 'variant.json').write_text(json.dumps(rec), encoding='utf-8', newline='\n')
        r4 = credibility.c4(rig, out)
        print(f'{name}: C4 done in {time.time() - t:.0f} s; KC responding {np.round(r4["smell"]["kc_fraction_responding"], 3)}, '
              f'rate at 0 / -4 mV {r4["stable"][0]["mean_hz"]:.2f} / {r4["stable"][-1]["mean_hz"]:.2f} Hz', flush=True)
        credibility.c5_c6(rig, out, r4)
        print(f'{name}: C5/C6 done in {time.time() - t:.0f} s', flush=True)
        del rig
        gc.collect()                   # operators.apply leaves a reference cycle (enc.bind); free the GPU memory now
        torch.cuda.empty_cache()
    (OUT / 'done.json').write_text(json.dumps(dict(finished_unix=time.time(), variants=list(variants.VARIANTS))), encoding='utf-8')


if __name__ == '__main__':
    main()
