"""Bound evaluation batches without dropping or shortening memory histories."""
import memory

original_assessment = memory.batched_assessment


def bounded_assessment(policy, task, delays, prefixes, max_steps=512, intervention=None):
    probes, rollouts = [], []
    # At most eight independent histories; all original cue/delay/prefix pairs remain.
    for delay in delays:
        for start in range(0, len(prefixes), 4):
            p, r = original_assessment(policy, task, [delay], prefixes[start:start + 4],
                                       max_steps=max_steps, intervention=intervention)
            probes.extend(p)
            rollouts.extend(r)
    return probes, rollouts


def install():
    memory.batched_assessment = bounded_assessment


def smoke(config, out):
    """Compare full-history and checkpointed gradients, then one longest-history update."""
    import numpy as np
    import torch
    from memory_tasks import frames, prepare_memory
    from registry import write

    policy = memory.MemoryPolicy(config, out)
    task = prepare_memory()[0]
    sequence = frames(task, 0, 0, 1, False)
    observations = []
    for checkpointed in (False, True):
        policy.reset()
        boards = [torch.tensor(f['board'], device='cuda', requires_grad=(i == 0))[None]
                  for i, f in enumerate(sequence)]
        for board, frame in zip(boards, sequence):
            if checkpointed:
                features = policy.encoder.checkpointed_advance(board, frame['ms'])
            else:
                features = policy.encoder.advance(board, frame['ms'])
        q = policy.head(features)['final_logits']
        objective = q[0, 3] - q[0, 2]
        gradients = torch.autograd.grad(objective, [boards[0], *policy.encoder.core.parameters()])
        assert all(torch.isfinite(g).all() for g in gradients)
        observations.append((q.detach().cpu(), [g.detach().cpu() for g in gradients]))
        policy.encoder.release_graph()
        del q, objective, features, gradients, boards
    torch.testing.assert_close(observations[0][0], observations[1][0], rtol=1e-5, atol=1e-5)
    errors = []
    for direct, recomputed in zip(observations[0][1], observations[1][1]):
        torch.testing.assert_close(direct, recomputed, rtol=5e-3, atol=1e-4)
        errors.append(float((direct - recomputed).abs().max()))
    optimizer = torch.optim.Adam([
        dict(params=policy.head.parameters(), lr=.0003),
        dict(params=policy.encoder.core.parameters(), lr=.0001)])
    # Iteration eight selects trained delay 160 ms and prefix four in the locked design.
    gradient = memory.training_step(policy, task, optimizer, 8)
    assert gradient['delay_ms'] == 160 and gradient['prefix_windows'] == 4
    assert np.isfinite(gradient['loss'] + gradient['gradient_norm'])
    result = dict(status='passed', full_history_gradient_max_errors=errors,
                  cue_input_gradient_norm=float(observations[0][1][0].norm()),
                  longest_training_history=gradient,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  temporary_parameters_discarded=True)
    write(out / 'result.json', result)
    return 0
