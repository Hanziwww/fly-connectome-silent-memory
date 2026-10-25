"""Trace the actual reset-per-decision full-brain policy, with no solver or route input."""
import base64
import json
import math
from collections import OrderedDict
import numpy as np
import torch
from bootstrap import HERE
from common import ASSETS, configure, file_hash, tensor_hash
from brain_model import FullBrain
from heads import ActivityHead
from evaluation import Predictor
from viewer_catalog import OUT, repair_bundle

def record_trace(encoder, board):
    if not hasattr(encoder, '_trace_visual'):
        graph = json.loads((ASSETS / 'graph.json').read_text())
        assert [n['id'] for n in graph['nodes']] == encoder.c.neurons.bodyId.iloc[encoder.visual].tolist()
        edges = np.asarray(graph['edges'])
        pre, post = encoder.visual[edges[:, 0].astype(int)], encoder.visual[edges[:, 1].astype(int)]
        encoder._trace_visual = torch.as_tensor(encoder.visual, device='cuda')
        encoder._trace_edges = (pre, post, np.asarray(encoder.brain._W_cpu[post, pre]).ravel().astype('float64'))
    visual = encoder._trace_visual
    pre, post, base = encoder._trace_edges
    b, p = encoder.brain, encoder.params
    x = np.zeros((encoder.batch, 4, 10, 10), dtype='float32')
    x[0] = board
    with torch.no_grad():
        encoder.matrix.bind(encoder.core, encoder.neuron_group)
        b.W = encoder.matrix
        p.v_th = encoder.base_threshold + encoder.core.threshold_mv[encoder.neuron_group]
        b.reset()
        b.t = 0.
        b.step_count = b.buf_pos = 0
        b.set_drive(encoder.input_index, torch.from_numpy(x).cuda().reshape(encoder.batch, 400).repeat_interleave(8, 1)*18)
        voltages, spikes, releases, rates = [b.v[0, visual].clone()], [], [], []
        for tick in range(1, 41):
            b.step()
            voltages.append(b.v[0, visual].clone())
            spikes.append(b.spikes[0, visual].clone())
            releases.append(b.spike_buf[(b.buf_pos-1) % b.n_delay, 0, visual].clone())
            if tick == 14:
                current = b.g[0, encoder.observed].clone()
            if tick in (14, 40):
                rates.append(b.rate[0, visual].clone())
        voltage = b.v[0, encoder.observed]-p.v_rest
        feature = torch.cat((current.reshape(256, 16).mean(1), voltage.reshape(256, 16).mean(1))).cpu().numpy()
        spike_array, release_array = torch.stack(spikes).cpu().numpy(), torch.stack(releases).cpu().numpy()
        voltage_array = torch.stack(voltages).cpu().numpy().astype('<f2')
        rate_array = torch.stack(rates).cpu().numpy()
        weights = (base * encoder.matrix.pre[pre].cpu().numpy() * encoder.matrix.post[post].cpu().numpy()).astype('float32')
    events = [[]] + [[[int(i), float(release[i])] for i in np.flatnonzero(spike)]
                      for spike, release in zip(spike_array, release_array)]
    trace = dict(schema_version=1, dt_ms=1, steps=40, events=events,
        rate_decay=math.exp(-p.dt/p.rate_tau), rate_gain_hz=(1-math.exp(-p.dt/p.rate_tau))*1000/p.dt,
        synaptic_decay=math.exp(-p.dt/p.tau_syn), delay_steps=b.n_delay, edge_weights_mv=weights.tolist(),
        voltage_mv_f16=base64.b64encode(voltage_array.tobytes()).decode('ascii'), voltage_shape=list(voltage_array.shape),
        signal_definition='Effective learned weight times released presynaptic spike, with model delay and decay.',
        geometry='Measured anatomical routes; displayed subset of the complete simulated graph.',
        reset_per_decision=True, feature_dim=512)
    return feature, rate_array, trace

def decide(head, feature, mask, checkpoint_id):
    # Identical fixed 128-row head evaluation to the training evaluator.
    board = np.zeros((1, 4, 10, 10), dtype='uint8')
    cache = {board[0].tobytes(): feature}
    return Predictor(head, encoder=True, checkpoint_id=checkpoint_id, correction=False, cache=cache).predict(board, mask[None])[0]

class ViewerRuntime:
    def __init__(self, include_repair=True):
        configure(930204)
        free, total = torch.cuda.mem_get_info()
        if free < 5 * 2**30:
            raise RuntimeError('Need at least 5 GiB of free GPU memory for the viewer.')
        torch.cuda.set_per_process_memory_fraction(11 * 2**30 / total)
        self.paths = dict(brain=OUT/'model.pt', **{f'stage{i+1}': OUT/f'stage-{i:03d}.pt' for i in range(4)})
        if include_repair:
            self.paths.update(repair_bundle()[0])
        self.objects = {k: torch.load(p, map_location='cpu', weights_only=True) for k,p in self.paths.items()}
        base=self.objects['brain']
        self.objects['initial']=dict(base,core=base['inherited_core'])
        self.objects['zero']=dict(base,core={k:torch.zeros_like(v) for k,v in base['core'].items()})
        self.ids={k:file_hash(p) for k,p in self.paths.items()}
        self.ids.update(initial=self.ids['brain']+':inherited',zero=self.ids['brain']+':zero')
        self.head=ActivityHead().cuda();self.head.load_state_dict(base['head']);self.head.requires_grad_(False)
        frozen=tensor_hash(self.head.state_dict())
        assert all(tensor_hash(o['head'])==frozen for o in self.objects.values())
        self.encoder=FullBrain(OUT,batch=32,persist_mapping=False)
        self.kind = None
        self.cache = OrderedDict()

    def response(self, board, kind):
        key = (kind, board.astype('uint8').tobytes())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if self.kind != kind:
            self.encoder.set_core(self.objects[kind]['core'])
            self.kind = kind
        result = record_trace(self.encoder, board)
        result[2]['checkpoint_id'] = self.ids[kind]
        self.cache[key] = result
        while len(self.cache) > 128:
            self.cache.popitem(last=False)
        return result
