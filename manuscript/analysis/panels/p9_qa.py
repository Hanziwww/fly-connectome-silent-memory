"""P9: history sets, input manifest, read-only audit, schema report, data dictionary and output manifest (written last)."""
import hashlib
import json
import os

import pandas as pd

from msa import csvout, manifest, paths, registry

SELF_DESCRIBED = ('meta/data_dictionary.csv', 'meta/output_manifest.csv', 'qa/determinism.csv')


def history_sets():
    rows = []
    for s in registry.all_sets():
        for i, m in enumerate(s['members']):
            rows.append(dict(history_set_id=s['history_set_id'], family=s['family'], unit=s['unit'], n=s['n'], member_index=i,
                             case=m.get('case'), cue=m.get('cue'), context=m.get('context'), visible=m.get('visible'), delay_ms=m.get('delay_ms'),
                             prefix_windows=m.get('prefix_windows'), split=m.get('split'), description=s['description']))
    df = pd.DataFrame(rows)
    for c in ('cue', 'context', 'delay_ms', 'prefix_windows'):
        df[c] = df[c].astype('Int64')
    bad = df.groupby('history_set_id').agg(n=('n', 'first'), k=('member_index', 'size'))
    if (bad.n != bad.k).any():
        raise RuntimeError('history set sizes inconsistent')
    csvout.write_table(df, 'meta/history_sets.csv', title='History sets defining every memory denominator (one row per member)', describe={
        'unit': 'history or cue_pair', 'n': 'Members in the set', 'member_index': 'Member index', 'description': 'Set definition'},
        key=['history_set_id', 'member_index'])
    return set(df.history_set_id)


def input_manifest():
    rows = manifest.merged()
    df = pd.DataFrame(rows)
    csvout.write_table(df, 'meta/input_manifest.csv', title='Every input file read (hashed once per NTFS file identity)', describe={
        'path': 'Project-relative path', 'file_id': 'Volume serial and file index (hard links share it)', 'bytes': 'File size',
        'sha256': 'SHA-256 of the content', 'n_links': 'Hard-link count', 'first_phase': 'Phase that first read the file'}, key=['path'])
    return df


def readonly_audit(inp):
    changed, missing = [], []
    hashes = {}
    for _, r in inp.iterrows():
        p = paths.ROOT / r['path']
        if not p.exists():
            missing.append(r['path'])
            continue
        if r['file_id'] not in hashes:
            h = hashlib.sha256()
            with open(p, 'rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b''):
                    h.update(chunk)
            hashes[r['file_id']] = h.hexdigest()
        if hashes[r['file_id']] != r['sha256']:
            changed.append(r['path'])
    rows = [dict(check='inputs_audited', value=len(inp)), dict(check='distinct_file_identities', value=len(hashes)),
            dict(check='inputs_changed_since_read', value=len(changed)), dict(check='inputs_missing', value=len(missing)),
            dict(check='inputs_inside_writable_roots', value=int(sum(any(str((paths.ROOT / p).resolve()).startswith(str(w.resolve()))
                                                                          for w in paths.WRITABLE) for p in inp['path'])))]
    for p in changed:
        rows.append(dict(check=f'changed:{p}', value=1))
    for p in missing:
        rows.append(dict(check=f'missing:{p}', value=1))
    csvout.write_table(pd.DataFrame(rows), 'qa/readonly_audit.csv', title='Read-only audit: every input re-hashed after the run', describe={
        'check': 'Audit item', 'value': 'Count'}, key=['check'], sort=[])
    if changed or missing:
        raise RuntimeError(f'read-only audit failed: {changed[:5]} {missing[:5]}')


def _dictionary_entries():
    entries = {}
    folder = paths.work('dictionary')
    for f in sorted(folder.glob('*.json')):
        e = json.loads(f.read_text(encoding='utf-8'))
        entries[e['file']] = e
    return entries


def schema_report(sets):
    entries = _dictionary_entries()
    rows = []
    used_sets = set()
    for p in sorted(paths.OUT.rglob('*.csv')):
        rel = p.relative_to(paths.OUT).as_posix()
        if rel in ('meta/output_manifest.csv', 'meta/data_dictionary.csv', 'qa/schema_report.csv', 'qa/determinism.csv'):
            continue
        df = pd.read_csv(p, low_memory=False, dtype=str, keep_default_na=False)
        e = entries.get(rel)
        documented = set(c['column'] for c in e['columns']) if e else set()
        undocumented = [c for c in df.columns if c not in documented]
        if 'history_set_id' in df.columns:
            used_sets |= {v for v in df['history_set_id'] if v}
        rows.append(dict(file=rel, rows=len(df), columns=len(df.columns), dictionary_entry=e is not None, undocumented_columns=len(undocumented),
                         rows_match_dictionary=(e is not None and e['rows'] == len(df)), utf8_no_bom=not p.read_bytes()[:3] == b'\xef\xbb\xbf',
                         lf_only=b'\r\n' not in p.read_bytes()))
    rep = pd.DataFrame(rows)
    unknown = sorted(used_sets - sets)
    rep['history_sets_resolved'] = not unknown
    csvout.write_table(rep, 'qa/schema_report.csv', title='Schema checks for every deliverable', describe={
        'file': 'Deliverable', 'rows': 'Data rows', 'columns': 'Columns', 'dictionary_entry': 'Has a data-dictionary entry',
        'undocumented_columns': 'Columns without a description', 'rows_match_dictionary': 'Row count equals the count recorded at write time',
        'utf8_no_bom': 'UTF-8 without byte-order mark', 'lf_only': 'LF line endings only', 'history_sets_resolved': 'Every history_set_id used anywhere exists in meta/history_sets.csv'},
        key=['file'])
    failed = rep[~rep.dictionary_entry | (rep.undocumented_columns > 0) | ~rep.rows_match_dictionary | ~rep.utf8_no_bom | ~rep.lf_only]
    if len(failed) or unknown:
        raise RuntimeError(f'schema report failed: {failed.file.tolist()[:5]} unknown sets {unknown[:5]}')


def data_dictionary():
    entries = _dictionary_entries()
    rows = []
    for rel, e in sorted(entries.items()):
        if rel in SELF_DESCRIBED:   # described below; an entry left by an earlier partial run is ignored
            continue
        for c in e['columns']:
            rows.append(dict(file=rel, title=e['title'], key=';'.join(e['key']), rows=e['rows'], column=c['column'], dtype=c['dtype'],
                             description=c['description']))
    for rel, title, cols in (('meta/data_dictionary.csv', 'Data dictionary', [('file', 'Deliverable'), ('title', 'Table title'), ('key', 'Key columns'),
                                                                               ('rows', 'Rows at write time'), ('column', 'Column'), ('dtype', 'Value type'),
                                                                               ('description', 'Description')]),
                             ('meta/output_manifest.csv', 'Output manifest', [('file', 'Deliverable'), ('rows', 'Data rows'), ('columns', 'Columns'),
                                                                              ('bytes', 'File size'), ('sha256', 'SHA-256')]),
                             ('qa/determinism.csv', 'Determinism: two complete runs from a clean state (written by run_all.py --determinism)',
                              [('file', 'Deliverable'), ('sha256_run1', 'SHA-256 after run 1'), ('sha256_run2', 'SHA-256 after run 2'),
                               ('identical', 'Byte-identical')])):
        for c, d in cols:
            rows.append(dict(file=rel, title=title, key='file' if rel.endswith('manifest.csv') else 'file;column', rows=None, column=c,
                             dtype='string' if c not in ('rows', 'columns', 'bytes') else 'integer', description=d))
    df = pd.DataFrame(rows)
    df['rows'] = df['rows'].astype('Int64')
    csvout.write_table(df, 'meta/data_dictionary.csv', title='Data dictionary', describe={
        'file': 'Deliverable', 'title': 'Table title', 'key': 'Key columns', 'rows': 'Rows at write time', 'column': 'Column', 'dtype': 'Value type',
        'description': 'Description'}, key=['file', 'column'])


def output_manifest():
    rows = []
    for p in sorted(paths.OUT.rglob('*.csv')):
        rel = p.relative_to(paths.OUT).as_posix()
        if rel in ('meta/output_manifest.csv', 'qa/determinism.csv'):
            continue
        data = p.read_bytes()
        rows.append(dict(file=rel, rows=data.count(b'\n') - 1, columns=data.split(b'\n', 1)[0].count(b',') + 1, bytes=len(data),
                         sha256=hashlib.sha256(data).hexdigest()))
    csvout.write_table(pd.DataFrame(rows), 'meta/output_manifest.csv', title='Output manifest', describe={
        'file': 'Deliverable', 'rows': 'Data rows', 'columns': 'Columns (header fields)', 'bytes': 'File size', 'sha256': 'SHA-256'}, key=['file'])


def main():
    sets = history_sets()
    inp = input_manifest()
    readonly_audit(inp)
    schema_report(sets)
    data_dictionary()
    output_manifest()
