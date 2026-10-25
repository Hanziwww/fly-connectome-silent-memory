import time
from collections import Counter
import numpy as np
import torch
from common import BudgetExpired
from curriculum import encode, legal, restore, solved, state_from, static_dead, step

class Predictor:
    def __init__(self, head, *, encoder=None, checkpoint_id, cache=None, correction=True):
        self.head, self.encoder = head, encoder
        self.checkpoint_id = checkpoint_id
        self.cache = {} if cache is None else cache
        self.correction = correction

    def features(self, boards, check):
        keys = [b.astype("uint8").tobytes() for b in boards]
        missing = {key: b for key, b in zip(keys, boards) if key not in self.cache}
        if missing:
            features = self.encoder.encode(np.stack(list(missing.values())), check)
            self.cache.update(zip(missing, features))
        return np.stack([self.cache[k] for k in keys])

    def predict(self, boards, masks, check=lambda: None):
        check()
        x = self.features(boards, check) if self.encoder is not None else boards
        results = []
        with torch.no_grad():
            for start in range(0, len(x), 128):
                part = x[start:start+128]
                padded = np.zeros((128, *x.shape[1:]), dtype="float32")
                padded[:len(part)] = part
                outputs = self.head(torch.from_numpy(padded).cuda(), correction=self.correction) if self.encoder is not None else None
                q = outputs["final_logits"] if outputs is not None else self.head(torch.from_numpy(padded).cuda())
                q = q[:len(part)]
                valid = torch.from_numpy(masks[start:start+len(part)]).cuda().bool()
                # Blocked boards are removed by rollout; callers must supply a legal action.
                if not valid.any(1).all():
                    raise ValueError("Cannot choose from an empty action set")
                prob = q.masked_fill(~valid, -1e9).softmax(1).cpu().numpy()
                for row in range(len(part)):
                    results.append(dict(action=int(prob[row].argmax()), final_probabilities=prob[row].tolist(),
                        brain_logits=outputs["brain_logits"][row].cpu().tolist() if outputs is not None else None,
                        correction_logits=outputs["correction_logits"][row].cpu().tolist() if outputs is not None else None,
                        checkpoint_id=self.checkpoint_id))
        return results

def summarize(records):
    n = len(records)
    wins = sum(r["status"] == "solved" for r in records)
    completed = not any(r["status"] == "budget_interrupted" for r in records)
    p, z = wins/max(n, 1), 1.959963984540054
    denom = 1+z*z/max(n, 1)
    center = (p+z*z/(2*max(n, 1)))/denom
    half = z*np.sqrt(p*(1-p)/max(n, 1)+z*z/(4*max(n, 1)**2))/denom
    return dict(solved=wins, total=n, complete=completed, success_rate=p if completed else None,
                wilson95=[float(center-half), float(center+half)] if completed else None,
                statuses=dict(Counter(r["status"] for r in records)), records=records)

def rollout(predictor, levels, check=lambda: None, max_steps=80):
    started = time.monotonic()
    maps = [restore(level) for level in levels]
    states = [state_from(level["start"]) for level in levels]
    seen = [{state} for state in states]
    status = ["solved" if solved(m, s) else "running" for m, s in zip(maps, states)]
    histories = [[] for _ in levels]
    try:
        for _ in range(max_steps):
            check()
            active = []
            for i, s in enumerate(status):
                if s != "running":
                    continue
                if static_dead(maps[i], states[i]):
                    status[i] = "proven_corner_deadlock"
                elif not legal(maps[i], states[i]).any():
                    status[i] = "blocked"
                else:
                    active.append(i)
            if not active:
                break
            decisions = predictor.predict(np.stack([encode(maps[i], states[i]) for i in active]),
                                          np.stack([legal(maps[i], states[i]) for i in active]), check)
            for i, decision in zip(active, decisions):
                histories[i].append(decision)
                states[i] = step(maps[i], states[i], decision["action"])
                if solved(maps[i], states[i]):
                    status[i] = "solved"
                elif states[i] in seen[i]:
                    status[i] = "repeated_state"
                else:
                    seen[i].add(states[i])
    except BudgetExpired:
        status = ["budget_interrupted" if s == "running" else s for s in status]
    records = [dict(level_id=l["id"], split=l["split"], status="step_timeout" if status[i] == "running" else status[i],
                    steps=len(histories[i]), actions=[h["action"] for h in histories[i]], decisions=histories[i])
               for i, l in enumerate(levels)]
    return dict(**summarize(records), seconds=time.monotonic()-started,
                train=summarize([r for r in records if r["split"] == "train"]),
                validation=summarize([r for r in records if r["split"] == "validation"]))
