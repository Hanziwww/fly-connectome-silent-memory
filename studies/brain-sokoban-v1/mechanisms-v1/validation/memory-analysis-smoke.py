import sys
sys.path.insert(0,'/root/autodl-tmp/brain-sokoban-v1-20261003/studies/brain-sokoban-v1/mechanisms-v1/code')
from mechanism_base import SESSION,write
from mechanism_analysis import memory_analysis
import torch
for name in ('tables','figures','atlas'):(SESSION/name).mkdir(exist_ok=True)
result=memory_analysis()
assert not torch.cuda.is_initialized(),'Offline analysis initialized CUDA'
write(SESSION/'memory-analysis-validation.json',dict(status='passed',cuda_initialized=False,
    geometry_rows=result['geometry_rows'],behavior_histories=result['behavior_histories'],models=len(result['dynamics'])))
print({'status':'passed','cuda_initialized':False,'geometry_rows':result['geometry_rows'],'models':len(result['dynamics'])})
