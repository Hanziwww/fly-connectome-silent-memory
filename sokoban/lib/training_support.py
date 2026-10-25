import time
import numpy as np
import torch
from common import tensor_hash
from heads import ActivityHead, action_loss

def batch_tensors(data, indices):
    return [torch.as_tensor(data[name][indices], device="cuda") for name in ("boards", "labels", "legal", "remaining")]

def resource_probe(encoder, data):
    """No optimizer step and no model selection; discard all probe gradients."""
    batch = encoder.batch
    board, labels, valid, _ = batch_tensors(data, np.arange(batch) % len(data["boards"]))
    head = ActivityHead().cuda()
    head.freeze_proposal()
    digest = tensor_hash(head.proposal.state_dict())
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    features = encoder.forward(board.float())
    loss = action_loss(head(features)["brain_logits"], labels, valid)
    loss.backward()
    torch.cuda.synchronize()
    gradients = {name: dict(nonzero=int((p.grad != 0).sum()), norm=float(p.grad.norm()),
                             finite=bool(torch.isfinite(p.grad).all())) for name, p in encoder.core.named_parameters()}
    if not all(g["finite"] and g["nonzero"] > 0 for g in gradients.values()):
        raise AssertionError("Action loss must reach all three internal parameter families")
    assert digest == tensor_hash(head.proposal.state_dict())
    return dict(batch=batch, seconds=time.monotonic()-start, gradients=gradients,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                task_only_gradient=True, optimizer_steps=0, readout_unchanged=True)

def check_zero_response(encoder, boards):
    with torch.no_grad():
        x = torch.as_tensor(boards, device="cuda").float()
        original = encoder.forward(x, reference=True, raw=True).clone()
        state = {name: getattr(encoder.brain, name).clone() for name in ("v", "g", "spikes", "adapt")}
        zero = encoder.forward(x, raw=True)
        states_equal = {name: bool(torch.equal(value, getattr(encoder.brain, name))) for name, value in state.items()}
        equal = bool(torch.equal(original, zero)) and all(states_equal.values())
        result = dict(passed=equal, max_abs_raw_difference=float((original-zero).abs().max()),
                      full_neuron_states_equal=states_equal, neurons_compared=encoder.c.n,
                      reference="Existing stable full LIF with unmodified effective weights and float64 accumulation")
        if not equal:
            raise AssertionError(f"Zero-modulation parity failed: {result}")
        return result



def gradients(encoder, head, data, check):
    accum = [torch.zeros_like(p) for p in encoder.core.parameters()]
    n = len(data["boards"])
    for offset in range(0, n, encoder.batch):
        check()
        count = min(encoder.batch, n-offset)
        padded = np.zeros((encoder.batch, 4, 10, 10), dtype="float32")
        padded[:count] = data["boards"][offset:offset+count]
        features = encoder.forward(torch.from_numpy(padded).cuda())[:count]
        q = head(features, correction=False)["brain_logits"]
        labels, valid = [torch.as_tensor(data[k][offset:offset+count], device="cuda") for k in ("labels", "legal")]
        (action_loss(q, labels, valid)*count/n).backward()
        for total, parameter in zip(accum, encoder.core.parameters()):
            total.add_(parameter.grad)
        encoder.release_graph()
    norm = torch.sqrt(sum(g.square().sum() for g in accum))
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite core gradient")
    return [g/max(1., float(norm)) for g in accum], float(norm)
