"""Numerical and timing probe for the exact sparse-propagation bottleneck."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'sokoban'))
import bootstrap
import time
import numpy as np
import torch
from common import configure
from backend import StudyBrain,SegmentedMatrix
from registry import STUDY,SOURCE,read,write
from game import encode,restore,state_from


def main():
    configure(10)
    encoder=StudyBrain(batch=16,precision='float32')
    saved=torch.load(SOURCE,map_location='cpu',weights_only=True);encoder.set_core(saved['core'])
    level=next(l for l in read(STUDY/'tasks/levels.json') if l['id']=='C0')
    x=torch.as_tensor(np.repeat(encode(restore(level),state_from(level['start']))[None],16,0),device='cuda')
    records=[]
    for deterministic,column in ((False,False),(True,False),(False,True),(True,True)):
        torch.use_deterministic_algorithms(deterministic)
        encoder.matrix.column_major=column
        try:
            values=[];times=[]
            for grad in (False,False,True,True):
                t=time.perf_counter()
                with torch.set_grad_enabled(grad):f=encoder.forward(x)
                values.append(f.detach().cpu().numpy());times.append(time.perf_counter()-t)
                encoder.release_graph()
            records.append(dict(deterministic=deterministic,column_major=column,seconds=times,
                repeated_inference=float(np.max(np.abs(values[0]-values[1]))),
                forward_vs_gradient=float(np.max(np.abs(values[0]-values[2]))),
                repeated_gradient=float(np.max(np.abs(values[2]-values[3]))),
                layout=str(encoder.matrix.base.layout)))
        except Exception as e:records.append(dict(deterministic=deterministic,error=repr(e)))
    write(STUDY/'speed-probe.json',records);print(records)
    encoder.matrix=SegmentedMatrix(encoder.matrix.base);encoder.brain.W=encoder.matrix
    values=[];times=[]
    for grad in (False,False,True,True):
        t=time.perf_counter()
        with torch.set_grad_enabled(grad):f=encoder.forward(x)
        values.append(f.detach().cpu().numpy());times.append(time.perf_counter()-t);encoder.release_graph()
    result=dict(seconds=times,repeated_inference=float(np.max(np.abs(values[0]-values[1]))),
        forward_vs_gradient=float(np.max(np.abs(values[0]-values[2]))),
        repeated_gradient=float(np.max(np.abs(values[2]-values[3]))))
    write(STUDY/'speed-probe-segmented.json',result);print(result)


if __name__=='__main__':main()
