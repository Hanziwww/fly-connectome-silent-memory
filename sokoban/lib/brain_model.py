"""Full connectome, hard LIF spikes, and type-shared internal plasticity.

No board representation is exposed to an action head. The upstream simulator
is reused without modifying its dynamics or pruning any of its connections.
"""
from __future__ import annotations

import hashlib
import io
import math
import zipfile
import numpy as np
import torch
from torch import nn

from common import ROOT, ASSETS, save_json


def make_groups(neurons, outputs):
    types = neurons.type.fillna("").astype(str).str.strip().to_numpy()
    classes = neurons.superclass.fillna("unknown").astype(str).to_numpy()
    keys = np.where(types != "", "type:" + types, "superclass:" + classes)
    names, index = np.unique(keys, return_inverse=True)
    body = neurons.bodyId.to_numpy()
    order = np.lexsort((body[outputs], types[outputs], classes[outputs]))
    observed = np.asarray(outputs)[order].reshape(-1, 16)
    return names.astype(str), index.astype("int64"), observed


def verify_saved_mapping(path, expected):
    """Verify numeric mappings and labels without unpickling old object arrays.

    The retained checkpoints saved group names as object arrays. For those locally created
    files, compare the serialized member with newly serialized expected labels;
    no bytes from the saved pickle are executed. New files use Unicode arrays.
    """
    legacy_labels = False
    with np.load(path, allow_pickle=False) as saved:
        for key, value in expected.items():
            try:
                recorded = saved[key]
            except ValueError as error:
                if key != "group_names" or "Object arrays cannot be loaded" not in str(error):
                    raise
                buffer = io.BytesIO()
                np.save(buffer, np.asarray(value, dtype=object), allow_pickle=True)
                with zipfile.ZipFile(path) as archive:
                    if archive.read("group_names.npy") != buffer.getvalue():
                        raise ValueError("Legacy group labels do not match the current annotations") from None
                legacy_labels = True
            else:
                np.testing.assert_array_equal(recorded, value)
    return dict(all_verified=True, legacy_labels_verified_by_bytes=legacy_labels, unpickled_saved_data=False)


class Core(nn.Module):
    def __init__(self, groups, device="cpu"):
        super().__init__()
        self.pre_log_gain = nn.Parameter(torch.zeros(groups, device=device))
        self.post_log_gain = nn.Parameter(torch.zeros(groups, device=device))
        self.threshold_mv = nn.Parameter(torch.zeros(groups, device=device))

    @torch.no_grad()
    def project(self):
        for parameter in (self.pre_log_gain, self.post_log_gain):
            parameter.clamp_(math.log(.8), math.log(1.25))
        self.threshold_mv.clamp_(-1., 1.)

    @torch.no_grad()
    def zero(self):
        for parameter in self.parameters():
            parameter.zero_()

    def penalty(self):
        return sum(parameter.square().mean() for parameter in self.parameters())

    def bound_audit(self):
        values = [p.detach() for p in self.parameters()]
        return dict(pre_gain_range=[float(values[0].exp().min()), float(values[0].exp().max())],
                    post_gain_range=[float(values[1].exp().min()), float(values[1].exp().max())],
                    threshold_shift_range_mv=[float(values[2].min()), float(values[2].max())],
                    in_bounds=all(bool(torch.isfinite(v).all()) for v in values)
                    and all(bool(((v >= math.log(.8)-1e-7) & (v <= math.log(1.25)+1e-7)).all()) for v in values[:2])
                    and bool((values[2].abs() <= 1).all()),
                    changed_parameters={n: int((p.detach() != 0).sum()) for n, p in self.named_parameters()})


class SharedMatrix:
    """Sparse W is constant; only dense pre/post factors request gradients.

    This never creates an N by N dense matrix or a gradient for sparse weights.
    Positive factors preserve every effective sign, endpoint and original zero.
    """
    def __init__(self, base):
        self.base = base.double()
        self.pre = self.post = None

    def bind(self, core, neuron_group):
        self.pre = core.pre_log_gain.exp()[neuron_group].double()
        self.post = core.post_log_gain.exp()[neuron_group].double()

    def __matmul__(self, x):
        return ((self.base @ (x.double() * self.pre[:, None])) * self.post[:, None]).float()


class StableReference:
    """The zero-plasticity float64 sum used by the reference simulator."""
    def __init__(self, base):
        self.base = base

    def __matmul__(self, x):
        return (self.base @ x.double()).float()


class FullBrain:
    def __init__(self, directory, batch=32, persist_mapping=True):
        from flybrain.brain import Brain, LIFParams
        from flybrain.connectome import load

        self.batch = batch
        self.c = load(cache_dir=ROOT / "flybrain/cache")
        mapping = np.load(ASSETS / "mapping.npz")
        self.inputs = mapping["inputs"]
        self.outputs = mapping["outputs"]
        self.visual = mapping["visual"]
        np.testing.assert_array_equal(self.c.neurons.bodyId.to_numpy()[self.outputs], mapping["output_body_ids"])
        assert self.c.n == 167106 and self.c.W.nnz == 25578600
        assert self.inputs.size == 3200 and len(self.outputs) == 4096
        assert not np.intersect1d(self.inputs, self.outputs).size
        names, groups, populations = make_groups(self.c.neurons, self.outputs)
        assert len(names) == 11770 and populations.shape == (256, 16)
        self.neuron_group = torch.as_tensor(groups, device="cuda")
        self.observed = torch.as_tensor(populations.ravel(), device="cuda")
        self.input_index = torch.as_tensor(self.inputs.ravel(), device="cuda")
        self.params = LIFParams(dt=1., event_driven=False, surrogate_grad=True)
        self.base_threshold = float(self.params.v_th)
        self.brain = Brain(self.c, self.params, device="cuda", batch=batch, seed=0,
                           cuda_kernels=False, metal_kernels=False)
        assert not self.brain._slow_active and self.brain.K == 1
        self.core = Core(len(names), "cuda")
        self.matrix = SharedMatrix(self.brain.W)
        self.brain.W = self.matrix
        self.reference = StableReference(self.matrix.base)
        w = self.brain._W_cpu
        self.base_digest = self.graph_digest()
        if persist_mapping:
            np.savez_compressed(directory / "neuron-mapping.npz", group_names=names,
                                neuron_group=groups, neuron_body_ids=self.c.neurons.bodyId.to_numpy(),
                                populations=populations, population_body_ids=self.c.neurons.bodyId.to_numpy()[populations],
                                inputs=self.inputs, original_outputs=self.outputs)
        else:
            self.mapping_verification = verify_saved_mapping(directory / "neuron-mapping.npz", dict(
                group_names=names, neuron_group=groups, populations=populations, inputs=self.inputs,
                neuron_body_ids=self.c.neurons.bodyId.to_numpy(),
                population_body_ids=self.c.neurons.bodyId.to_numpy()[populations], original_outputs=self.outputs))
        self.metadata = dict(neurons=self.c.n, stored_connectome_edges=self.c.W.nnz,
                             effective_stored_edges=w.nnz, effective_nonzero_edges=int(np.count_nonzero(w.data)),
                             groups=len(names), internal_parameters=3*len(names), input_neurons=3200,
                             observed_neurons=4096, populations=256, neurons_per_population=16,
                             feature_dim=512, simulation_ms=40, snapshots_ms=[14, 40],
                             sparse_accumulation="float64; float32 LIF states", base_graph_sha256=self.base_digest,
                             mapping="Known type, else superclass; observations sorted superclass/type/bodyId")
        if persist_mapping:
            save_json(directory / "brain.json", self.metadata)
        else:
            import json
            recorded = json.loads((directory/"brain.json").read_text(encoding="utf-8"))
            if recorded["base_graph_sha256"] != self.base_digest:
                raise ValueError("The saved model's effective connectome has changed")

    def graph_digest(self):
        w = self.brain._W_cpu
        h = hashlib.sha256()
        for value in (w.indptr, w.indices, w.data):
            h.update(value.tobytes())
        return h.hexdigest()

    def release_graph(self):
        for name in ("v", "g", "g_slow", "g_slow_cls", "refrac", "drive", "poisson_p", "rate",
                     "spikes", "adapt", "res", "spike_buf", "spike_counts"):
            setattr(self.brain, name, getattr(self.brain, name).detach())
        self.params.v_th = self.base_threshold
        self.matrix.pre = self.matrix.post = None
        for parameter in self.core.parameters():
            parameter.grad = None

    def resize(self, batch):
        """Resource fallback reduces batch size without reloading the graph."""
        if not 1 <= batch <= self.batch:
            raise ValueError("Batch fallback may only shrink")
        self.release_graph()
        b = self.brain
        for name in ("v", "g", "g_slow", "refrac", "drive", "poisson_p", "rate", "spikes",
                     "adapt", "res", "spike_counts"):
            setattr(b, name, getattr(b, name)[:batch].clone())
        b.g_slow_cls = b.g_slow_cls[:, :batch].clone()
        b.spike_buf = b.spike_buf[:, :batch].clone()
        b._syn_input = b._syn_input[:, :batch].clone()
        b.B = self.batch = batch

    def forward(self, boards, *, reference=False, raw=False):
        if boards.shape != (self.batch, 4, 10, 10):
            raise ValueError("Simulation uses its selected fixed batch, including zero padding")
        self.matrix.bind(self.core, self.neuron_group)
        self.brain.W = self.reference if reference else self.matrix
        self.params.v_th = (self.base_threshold if reference else
                            self.base_threshold + self.core.threshold_mv[self.neuron_group])
        self.brain.reset()
        self.brain.t = 0.
        self.brain.step_count = self.brain.buf_pos = 0
        self.brain.set_drive(self.input_index, boards.reshape(self.batch, 400).repeat_interleave(8, 1)*18)
        self.brain.run_ms(14)
        current = self.brain.g[:, self.observed]
        self.brain.run_ms(26)
        voltage = self.brain.v[:, self.observed] - self.params.v_rest
        if raw:
            return torch.cat((current, voltage), 1)
        return torch.cat((current.reshape(self.batch, 256, 16).mean(2),
                          voltage.reshape(self.batch, 256, 16).mean(2)), 1)

    def encode(self, boards, check=lambda: None):
        results = []
        with torch.no_grad():
            for start in range(0, len(boards), self.batch):
                check()
                chunk = boards[start:start+self.batch]
                padded = np.zeros((self.batch, 4, 10, 10), dtype="float32")
                padded[:len(chunk)] = chunk
                result = self.forward(torch.from_numpy(padded).cuda())
                results.append(result[:len(chunk)].cpu().numpy())
        return np.concatenate(results)

    def set_core(self, state=None):
        self.release_graph()
        if state is None:
            self.core.zero()
        else:
            self.core.load_state_dict(state, strict=True)
        assert self.core.bound_audit()["in_bounds"]
