"""Trainable 512 -> 64 -> 4 readout with no board or recurrent-state input."""
import torch
from torch import nn


class ActivityMLP(nn.Module):
    architecture = 'activity_mlp_512_64_4_relu_v1'

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(512))
        self.register_buffer('scale', torch.ones(512))
        self.hidden = nn.Linear(512, 64)
        self.output = nn.Linear(64, 4)

    @classmethod
    def from_linear(cls, state):
        """Embed a linear map as ReLU(x)-ReLU(-x); remaining units can learn."""
        model = cls()
        with torch.no_grad():
            model.mean.copy_(state['mean'])
            model.scale.copy_(state['scale'])
            weight = state['proposal.weight'] / state['logit_scale']
            bias = state['proposal.bias'] / state['logit_scale']
            model.hidden.weight[:4].copy_(weight)
            model.hidden.weight[4:8].copy_(-weight)
            model.hidden.bias[:4].copy_(bias)
            model.hidden.bias[4:8].copy_(-bias)
            model.output.weight.zero_()
            model.output.bias.zero_()
            model.output.weight[:, :4].copy_(torch.eye(4))
            model.output.weight[:, 4:8].copy_(-torch.eye(4))
        return model

    def forward(self, features, correction=False):
        if features.ndim != 2 or features.shape[1] != 512:
            raise ValueError('ActivityMLP accepts only 512 brain activity features')
        z = ((features - self.mean) / self.scale).clamp(-10, 10)
        q = self.output(torch.relu(self.hidden(z)))
        # Existing evaluator field names are retained; these are joint-model logits.
        return dict(brain_logits=q, joint_logits=q, final_logits=q,
                    correction_logits=torch.zeros_like(q), progress=torch.zeros_like(q[:, 0]))


def padded_forward(head, features):
    if len(features) > 128:
        raise ValueError('This bounded experiment uses at most 128 training states')
    padded = torch.nn.functional.pad(features, (0, 0, 0, 128 - len(features)))
    return head(padded, correction=False)['final_logits'][:len(features)]
