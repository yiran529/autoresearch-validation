"""Runtime-only finite checks for explicitly short diagnostics, not benchmark code."""
import json
import math
import os
from pathlib import Path
import torch

def record_loss(loss,step):
    value=float(loss.detach().float().item())
    path=Path(os.environ['GLVAL_OUTPUT'])/f'diagnostic-loss-rank-{os.environ["RANK"]}.jsonl'
    with path.open('a') as f:f.write(json.dumps({'step':step,'loss':value,'finite':math.isfinite(value)})+'\n')
    if not math.isfinite(value):raise RuntimeError(f'Nonfinite forward loss at update {step}')

def check_gradients(parameters,step):
    norms=torch._foreach_norm([p.grad for p in parameters if p.grad is not None])
    finite=bool(torch.isfinite(torch.stack(norms)).all().item())
    if not finite:raise RuntimeError(f'Nonfinite reduced gradient after backward at update {step}')

def finish(local):
    rank=int(os.environ['RANK'])
    loss=float(local['loss'].detach().float().item())
    result={'status':'diagnostic_completed','scope':'Short real-C4 stability check; not full pretraining/held-out evaluation',
            'completed_steps':local['update_step'],'target_steps':local['target_steps'],
            'scheduler_steps':local['args'].num_training_steps,'last_training_loss':loss,
            'last_loss_finite':math.isfinite(loss),'compression_start':local['args'].start_compress_iter,
            'rank':rank,'activation_checkpointing':local['args'].activation_checkpointing,
            'batch_per_device':local['args'].batch_size}
    (Path(os.environ['GLVAL_OUTPUT'])/f'diagnostic-result-rank-{rank}.json').write_text(json.dumps(result,indent=2)+'\n')

