"""Cell-group anatomy: group membership, annotations and hop distances on the connectome graph."""
import numpy as np
import pandas as pd

from . import io, paths


def group_table():
    """One row per shared parameter group (11,770) with membership and majority annotations."""
    ann = io.parquet_load(paths.M1 / 'neurons' / 'annotations.parquet',
                          columns=['bodyId', 'type', 'superclass', 'nt', 'parameter_group', 'direct_input', 'observed_by_policy'])
    names = io.npz_load(paths.ARCHIVE / 'experiments' / 'campaign' / 'run' / 'neuron-mapping.npz')['group_names']
    ann['superclass'] = ann['superclass'].fillna('unannotated')
    ann['nt'] = ann['nt'].fillna('unknown')

    def majority(s):
        vc = s.value_counts()
        return vc.index[0]

    g = ann.groupby('parameter_group').agg(n_cells=('bodyId', 'size'), superclass=('superclass', majority), transmitter=('nt', majority),
                                           input_cells=('direct_input', 'sum'), observed_cells=('observed_by_policy', 'sum'))
    g = g.reindex(range(len(names)))
    g['group_name'] = names
    g['membership'] = np.select([(g.input_cells > 0) & (g.observed_cells > 0), g.input_cells > 0, g.observed_cells > 0],
                                ['input_and_observed', 'input', 'observed'], 'neither')
    return g.reset_index().rename(columns={'parameter_group': 'group'}), ann


def hop_distances(ann):
    """Shortest synaptic path length (pre -> post) from any input cell, and to any observed cell, per cell."""
    W = io.sparse_load(paths.CONNECTOME / 'W_post_pre.npz').tocsr()
    A = (W != 0).astype(np.float32)          # rows post, columns pre
    n = A.shape[0]
    index = pd.Series(np.arange(n), index=io.npz_load(paths.M1 / 'neurons' / 'index.npz')['body_ids'])
    order = index.loc[ann['bodyId'].to_numpy()].to_numpy()

    def bfs(sources, forward):
        dist = np.full(n, -1, dtype=np.int32)
        frontier = np.zeros(n, dtype=np.float32)
        frontier[sources] = 1
        dist[sources] = 0
        step = 0
        M = A if forward else A.T.tocsr()
        while frontier.any():
            step += 1
            reached = (M @ frontier) > 0
            new = reached & (dist < 0)
            dist[new] = step
            frontier = new.astype(np.float32)
        return dist

    idx = io.npz_load(paths.M1 / 'neurons' / 'index.npz')
    d_in = bfs(np.nonzero(idx['inputs'])[0], forward=True)      # post <- pre: downstream of inputs
    d_obs = bfs(np.nonzero(idx['observed'])[0], forward=False)  # upstream of observed cells
    return pd.DataFrame({'bodyId': ann['bodyId'].to_numpy(), 'hops_from_input': d_in[order], 'hops_to_observed': d_obs[order]})
