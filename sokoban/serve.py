"""Inference viewer for the four-task pilot and the three B00 repair checkpoints."""
from workspace import ARCHIVE, RUNS
import argparse
import json
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import numpy as np
from bootstrap import HERE, ROOT
from curriculum import encode, legal, restore, solved, state_from, step, static_dead
from viewer_runtime import ViewerRuntime, decide, OUT
from viewer_catalog import REPAIR, INTERVENTION, PILOT_MODELS, REPAIR_MODELS, repair_bundle
from workspace import EVIDENCE


def public_result(result):
    # The page needs scores, never stored action sequences or training decisions.
    return dict(solved=result['solved'], total=result['total'], complete=result['complete'],
                records=[{key: row[key] for key in ('level_id', 'status', 'steps')} for row in result['records']])


class Player:
    def __init__(self):
        self.levels = {l['id']: l for l in json.loads((ARCHIVE/'campaign/pilot-levels.json').read_text())}
        self.verification = json.loads((OUT/'viewer-verification.json').read_text())
        assert self.verification['status'] == 'verified'
        _, self.repair_trials, repair_level = repair_bundle()
        self.repair_verification = json.loads((INTERVENTION/'verification.json').read_text())
        assert self.repair_verification['status'] == 'verified'
        self.levels[repair_level['id']] = repair_level
        self.runtime = None
        self.lock = threading.Lock()

    def catalog(self):
        report=self.verification
        models = {**PILOT_MODELS, **REPAIR_MODELS}
        results={k:dict(pilot=dict(test=public_result(v))) for k,v in report['results'].items()}
        for kind, result in self.repair_verification['results'].items():
            results.setdefault(kind, {})['repair'] = dict(test=public_result(result))
        training={}
        for i in range(4):
            path=OUT/f'stages/{i:03d}/training.json'
            training[f'Level {i+1}']=json.loads(path.read_text()) if path.exists() else []
        levels=[]
        for i, level in enumerate(self.levels.values()):
            repair = level['id'] == 'B00'
            item = {k:v for k,v in level.items() if k not in ('canonical_map', 'reference_actions')}
            if repair:
                item['descriptor'] = dict(moves=level['optimal_steps'], pushes=level['minimum_pushes'])
            levels.append(dict(item, suite='repair' if repair else 'pilot',
                               display_name='B00 · long walk' if repair else f'Level {i+1}'))
        repair_training = {}
        for kind, result in self.repair_trials.items():
            rows = [dict(generation=0, current_correct=result['baseline']['correct'], reference_states=13)]
            path = REPAIR / f"trials/D-s{result['seed']}/history.jsonl"
            rows += [dict(generation=r['generation'], current_correct=r['correct'], reference_states=13)
                     for r in map(json.loads, path.read_text().splitlines())]
            rows.append(dict(generation=result['training']['generations'], current_correct=13, reference_states=13))
            repair_training[models[kind]] = rows
        return dict(levels=levels,models=models,results=results,training=training,
            checkpoint_ids={**report['checkpoint_ids'], **self.repair_verification['checkpoint_ids']},
            max_steps=512,
            brain=json.loads((OUT/'brain.json').read_text()), default_suite='repair',
            suites=dict(pilot=dict(label='Four-level pilot', models=list(PILOT_MODELS), default_model='brain', reference_model='initial'),
                        repair=dict(label='B00 training repair', models=list(REPAIR_MODELS), default_model='repair10', reference_model='repair_source')),
            repair_training=repair_training, repair_interventions=self.repair_verification['summary'],
            repair_seconds={k:r['training']['training_seconds'] for k,r in self.repair_trials.items()})

    def validate(self, msg):
        source = self.levels[msg['level_id']]
        kind = msg.get('model', 'brain')
        allowed = REPAIR_MODELS if source['id'] == 'B00' else PILOT_MODELS
        if kind not in allowed:
            raise ValueError('Unknown checkpoint')
        obj = msg['state']
        if not isinstance(obj, list) or len(obj) != 2 or type(obj[0]) is not int or not isinstance(obj[1], list) or any(type(x) is not int for x in obj[1]):
            raise ValueError('Invalid state')
        m, s = restore(source), state_from(obj)
        p, b = s
        if len(b) != source['boxes'] or len(set(b)) != len(b) or p in b or any(x < 0 or x >= 100 or x in m['walls'] for x in (p, *b)):
            raise ValueError('Invalid placement')
        return source, m, s, kind

    def respond(self, msg, moving):
        source, m, s, kind = self.validate(msg)
        if moving and solved(m, s):
            return dict(terminal=True, state=s)
        if moving and static_dead(m, s):
            return dict(blocked=True, status='proven_corner_deadlock', state=s)
        with self.lock:
            if self.runtime is None:
                self.runtime = ViewerRuntime()
            runtime = self.runtime
            board, mask = encode(m, s), legal(m, s)
            feature, rates, trace = runtime.response(board, kind)
            reference_model = 'repair_source' if source['id'] == 'B00' else 'initial'
            reference_kind = 'before-repair' if source['id'] == 'B00' else 'before-learning'
            bf, br, bt = runtime.response(board, reference_model)
            reference = dict(rates=br.tolist(), trace=bt)
            result = dict(rates=rates.tolist(), trace=trace, source='recorded-simulation',
                          reference=reference, reference_kind=reference_kind, checkpoint_id=runtime.ids[kind],
                          reset_per_decision=True)
            if not mask.any():
                return dict(**result, blocked=True, state=s)
            decision = decide(runtime.head, feature, mask, runtime.ids[kind])
            before = decide(runtime.head, bf, mask, runtime.ids[reference_model])
            reference.update(action=before['action'], probabilities=before['final_probabilities'],
                             brain_logits=before['brain_logits'], checkpoint_id=before['checkpoint_id'])
            result.update(brain_logits=decision['brain_logits'], correction_logits=decision['correction_logits'],
                          final_probabilities=decision['final_probabilities'], probabilities=decision['final_probabilities'])
            if moving:
                after = step(m, s, decision['action'])
                result.update(action=decision['action'], state=after, terminal=solved(m, after))
            return result

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def reply(self, obj, status=200):
        body = json.dumps(obj, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/evidence/') and self.path.endswith('.json'):
            name = self.path[len('/evidence/'):-len('.json')]
            if name not in EVIDENCE:
                return self.reply(dict(error='Not found'), 404)
            return self.reply(json.loads(EVIDENCE[name].read_text(encoding='utf-8')))
        if self.path == '/README.md':
            body = (ROOT / 'README.md').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == '/api/catalog':
            return self.reply(self.server.player.catalog())
        return super().do_GET()

    def do_POST(self):
        if self.path not in ('/api/move', '/api/activity'):
            return self.reply(dict(error='Not found'), 404)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 2048:
                raise ValueError('Invalid request size')
            message = json.loads(self.rfile.read(size))
            return self.reply(self.server.player.respond(message, self.path == '/api/move'))
        except (ValueError, KeyError, IndexError, TypeError) as error:
            return self.reply(dict(error=str(error)), 400)
        except RuntimeError as error:
            return self.reply(dict(error=str(error)), 503)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8774)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.player = Player()
    if args.open:
        import webbrowser
        webbrowser.open(f'http://127.0.0.1:{args.port}/')
    print(f'FlyLab pilot and B00 repair: http://127.0.0.1:{args.port}/', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
