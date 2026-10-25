"""Run the final manuscript analysis (phases P0-P10) on CPU from saved records.

Usage (bundled runtime):
  .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script manuscript\\analysis\\run_all.py [--phase p1 p2 ...]
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from msa import guard  # noqa: E402

guard.install()

from msa import manifest, registry, stats  # noqa: E402

# P10 (figure-ready tables) runs before P9 so the manifests and the data dictionary include its tables.
PHASES = ['p0_inventory', 'p1_canonical', 'p2_acquisition', 'p3_brain_readout', 'p4_memory',
          'p5_interventions', 'p6_mechanisms', 'p6b_circuits', 'p7_methods', 'p11_followup', 'p12_plan_v2', 'p8_statistics', 'p10_figure_data', 'p9_qa']


def run(name):
    phase = name.split('_')[0]
    manifest.set_phase(phase)
    stats.set_phase(phase)
    module = importlib.import_module(f'panels.{name}')
    t = time.perf_counter()
    module.main()
    manifest.save_phase(phase)
    stats.save_phase(phase)
    registry.save_sets(phase)
    print(f'[{phase}] done in {time.perf_counter() - t:.1f} s', flush=True)


def clean():
    """Remove previously generated deliverables and intermediates (only the two writable roots)."""
    import shutil
    from msa import paths
    for root in paths.WRITABLE:
        if root.exists():
            shutil.rmtree(root)


def determinism():
    """Run the full pipeline twice in fresh processes and compare every output hash."""
    import os
    import subprocess
    import pandas as pd
    from msa import csvout, paths
    script = str(Path(__file__).resolve())
    manifests = []
    for i in (1, 2):
        r = subprocess.run([sys.executable, '-B', '-X', 'utf8', script, '--clean'], env=os.environ.copy())
        if r.returncode != 0:
            raise SystemExit(f'determinism pass {i} failed')
        manifests.append(pd.read_csv(paths.out('meta/output_manifest.csv')).set_index('file')['sha256'])
    a, b = manifests
    files = sorted(set(a.index) | set(b.index))
    rows = [dict(file=f, sha256_run1=a.get(f), sha256_run2=b.get(f), identical=a.get(f) == b.get(f)) for f in files]
    df = pd.DataFrame(rows)
    csvout.write_table(df, 'qa/determinism.csv', title='Determinism: two complete runs from a clean state', describe={
        'file': 'Deliverable', 'sha256_run1': 'SHA-256 after run 1', 'sha256_run2': 'SHA-256 after run 2', 'identical': 'Byte-identical'}, key=['file'])
    print(f'determinism: {int(df.identical.sum())}/{len(df)} files byte-identical', flush=True)
    if not df.identical.all():
        raise SystemExit('determinism check failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', nargs='*', help='phase prefixes to run, e.g. p1 p2 (default: all)')
    parser.add_argument('--clean', action='store_true', help='delete manuscript/results and analysis/_work first')
    parser.add_argument('--determinism', action='store_true', help='run everything twice from a clean state and compare output hashes')
    args = parser.parse_args()
    if args.determinism:
        determinism()
        return
    if args.clean:
        clean()
    selected = [p for p in PHASES if not args.phase or p.split('_')[0] in args.phase]
    for name in selected:
        run(name)


if __name__ == '__main__':
    main()
