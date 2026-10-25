"""Original frozen activity head; unused historical buffers preserved for checkpoint parity."""
import torch
from torch import nn

class ActivityHead(nn.Module):
    def __init__(self, mean=None, scale=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(512) if mean is None else torch.as_tensor(mean).float())
        self.register_buffer("scale", torch.ones(512) if scale is None else torch.as_tensor(scale).float().clamp_min(.1))
        self.register_buffer("logit_scale", torch.tensor(1.))
        self.proposal = nn.Linear(512, 4)
        self.residual = nn.Sequential(nn.Linear(512, 8), nn.Tanh(), nn.Linear(8, 4))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.progress = nn.Linear(512, 1)

    def forward(self, features, correction=True):
        if features.ndim != 2 or features.shape[1] != 512:
            raise ValueError("The head accepts only 512 activity features")
        z = ((features-self.mean)/self.scale).clamp(-10, 10)
        brain = self.proposal(z)/self.logit_scale
        adjustment = .25*torch.tanh(self.residual(z)) if correction else torch.zeros_like(brain)
        return {"brain_logits": brain, "correction_logits": adjustment, "final_logits": brain+adjustment,
                "progress": self.progress(z).squeeze(1)}

    def freeze_proposal(self):
        self.proposal.requires_grad_(False)

def action_loss(logits, labels, legal):
    """Uniform target over ALL verified optimal actions; shared legal mask."""
    logits = logits.masked_fill(~legal.bool(), -1e9)
    target = labels.float()/labels.sum(1, keepdim=True)
    return -(target*logits.log_softmax(1)).sum(1).mean()
