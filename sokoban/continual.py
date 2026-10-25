"""Sequential training helpers; no demonstration lookup in inference."""
from workspace import ARCHIVE, RUNS
import json
import numpy as np
import torch
from bootstrap import HERE
from common import file_hash
from curriculum import encode, legal, restore, state_from, step, solved
from evaluation import Predictor, rollout


def load_campaign(directory=ARCHIVE/"campaign"):
    receipt = json.loads((directory/"pilot-selection.json").read_text())
    for name, key in (("pilot-levels.json", "pilot_level_list_sha256"),
                      ("pilot-demonstrations.npz", "pilot_demonstrations_sha256")):
        if file_hash(directory/name) != receipt[key]:
            raise ValueError(f"Fixed campaign file changed: {name}")
    levels = json.loads((directory/"pilot-levels.json").read_text())
    with np.load(directory/"pilot-demonstrations.npz", allow_pickle=False) as saved:
        data = {k: saved[k] for k in saved.files}
    return levels, data, receipt


def subset(data, indices):
    return {k: data[k][indices] for k in ("boards", "labels", "legal")}


def concatenate(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in ("boards", "labels", "legal")}


def episode_data(level, record):
    if record["status"] != "solved":
        raise ValueError("Only successful autonomous trajectories enter retention bank")
    m, s = restore(level), state_from(level["start"])
    boards, labels, masks = [], [], []
    for action in record["actions"]:
        board, mask = encode(m, s).astype("uint8"), legal(m, s)
        if not mask[action]:
            raise ValueError("Illegal saved autonomous action")
        label = np.zeros(4, dtype=bool)
        label[action] = True
        boards.append(board)
        labels.append(label)
        masks.append(mask)
        s = step(m, s, action)
    if not solved(m, s) or not boards:
        raise ValueError("Saved trajectory does not solve its map")
    return dict(boards=np.stack(boards), labels=np.stack(labels), legal=np.stack(masks))


def evaluate(encoder, head, levels, tag, check=lambda: None, measured=None):
    # A new predictor per internal state; optional data were recomputed for THIS core.
    cache = {}
    for boards, features in measured or []:
        cache.update((b.astype("uint8").tobytes(), f) for b, f in zip(boards, features))
    predictor = Predictor(head, encoder=encoder, checkpoint_id=tag, correction=False, cache=cache)
    result = rollout(predictor, levels, check, max_steps=512)
    # These are trained tasks; do not attach a population confidence interval.
    return {k: result[k] for k in ("complete", "solved", "total", "success_rate", "records", "seconds")}


def retention_delta(before, after):
    a = {r["level_id"] for r in before if r["status"] == "solved"}
    b = {r["level_id"] for r in after if r["status"] == "solved"}
    return dict(new_successes=sorted(b-a), lost_successes=sorted(a-b), current_successes=sorted(b))


def choose_replay(bank, levels, rng, count=16):
    keys = sorted(bank)
    if not keys:
        return None
    # Balance by map, not by path length; cycle once before repeating a map.
    chosen = []
    while len(chosen) < count:
        chosen.extend(rng.permutation(keys).tolist())
    rows = []
    for key in chosen[:count]:
        data = episode_data(levels[int(key[1:])], bank[key])
        index = int(rng.integers(len(data["boards"])))
        rows.append(subset(data, [index]))
    return concatenate(rows)


def core_proposal(saved, direction, amount):
    import math
    result = {}
    for key, value in saved.items():
        bound = (math.log(.8), math.log(1.25)) if key.endswith("log_gain") else (-1., 1.)
        result[key] = (value + amount*direction[key]).clamp(*bound)
    return result


def measured(encoder, head, data, check=lambda: None):
    features = encoder.encode(data["boards"], check)
    with torch.no_grad():
        q = head(torch.as_tensor(features, device="cuda"), correction=False)["brain_logits"]
        valid = torch.as_tensor(data["legal"], device="cuda").bool()
        label = torch.as_tensor(data["labels"], device="cuda").float()
        q = q.masked_fill(~valid, -1e9)
        losses = -(label/label.sum(1, keepdim=True)*q.log_softmax(1)).sum(1).cpu().numpy()
        actions = q.argmax(1).cpu().numpy()
    correct = data["labels"][np.arange(len(actions)), actions]
    return dict(loss=float(losses.mean()), correct=int(correct.sum()),
                actions=actions, features=features, per_state_loss=losses)
