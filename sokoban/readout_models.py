"""Activity-only readouts for a controlled frozen-connectome comparison."""
import torch
from torch import nn
from torch.nn import functional as F

from hybrid_model import ActivityMLP


def output(q):
    return dict(brain_logits=q, joint_logits=q, final_logits=q,
                correction_logits=torch.zeros_like(q), progress=torch.zeros_like(q[:, 0]))


class ActivityLinear(nn.Module):
    """Affine in the same fixed normalized/clipped features used by every arm."""
    architecture = 'activity_linear_512_4_v1'

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(512))
        self.register_buffer('scale', torch.ones(512))
        self.proposal = nn.Linear(512, 4)

    @classmethod
    def from_linear(cls, state):
        model = cls()
        with torch.no_grad():
            for name in ('mean', 'scale'):
                getattr(model, name).copy_(state[name])
            model.proposal.weight.copy_(state['proposal.weight'] / state['logit_scale'])
            model.proposal.bias.copy_(state['proposal.bias'] / state['logit_scale'])
        return model

    def forward(self, features, correction=False):
        if features.ndim != 2 or features.shape[1] != 512:
            raise ValueError('Readouts accept only 512 brain activity features')
        z = ((features - self.mean) / self.scale).clamp(-10, 10)
        return output(self.proposal(z))


class ActivityEI(nn.Module):
    """Nonnegative rates; each hidden unit has one fixed outgoing weight sign.

    This constraint applies only to the added hidden-to-action projections.
    Inputs are pooled, standardized neural measurements, not identified
    presynaptic neurons. This is not a whole-circuit Dale or spiking model.
    """
    architecture = 'activity_ei_512_64_4_relu_softplus_v1'

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(512))
        self.register_buffer('scale', torch.ones(512))
        self.register_buffer('signs', torch.cat((torch.ones(32), -torch.ones(32))))
        self.hidden = nn.Linear(512, 64)
        self.raw_magnitude = nn.Parameter(torch.zeros(4, 64))
        self.bias = nn.Parameter(torch.zeros(4))

    @classmethod
    def from_linear(cls, state):
        """Pair E/I units to match the old map without zero/dead softplus weights."""
        model = cls()
        with torch.no_grad():
            for name in ('mean', 'scale'):
                getattr(model, name).copy_(state[name])
            weight = state['proposal.weight'] / state['logit_scale']
            bias = state['proposal.bias'] / state['logit_scale']
            model.hidden.weight[:4].copy_(weight)
            model.hidden.weight[4:8].copy_(-weight)
            model.hidden.bias[:4].copy_(bias)
            model.hidden.bias[4:8].copy_(-bias)
            model.hidden.weight[32:].copy_(model.hidden.weight[:32])
            model.hidden.bias[32:].copy_(model.hidden.bias[:32])
            # Paired .01 contributions cancel. Four E increments represent
            # ReLU(q); four I increments represent -ReLU(-q).
            magnitude = torch.full((4, 64), .01)
            magnitude[:, :4].add_(torch.eye(4))
            magnitude[:, 36:40].add_(torch.eye(4))
            model.raw_magnitude.copy_(torch.log(torch.expm1(magnitude)))
        return model

    def effective_weight(self):
        return F.softplus(self.raw_magnitude) * self.signs[None, :]

    def forward(self, features, correction=False):
        if features.ndim != 2 or features.shape[1] != 512:
            raise ValueError('Readouts accept only 512 brain activity features')
        z = ((features - self.mean) / self.scale).clamp(-10, 10)
        rates = F.relu(self.hidden(z))
        return output(F.linear(rates, self.effective_weight(), self.bias))

    def sign_audit(self):
        with torch.no_grad():
            weight = self.effective_weight()
            expected = torch.cat((torch.ones(32), -torch.ones(32))).to(self.signs)
            return dict(fixed_signs_valid=bool(torch.equal(self.signs, expected)),
                        all_outgoing_signs_valid=bool((weight * expected[None, :] >= 0).all()),
                        excitatory_units=32, inhibitory_units=32,
                        minimum_magnitude=float(weight.abs().min()),
                        maximum_magnitude=float(weight.abs().max()))


MODELS = dict(linear=ActivityLinear, mlp=ActivityMLP, ei=ActivityEI)


def make_readout(kind, source_head=None):
    cls = MODELS[kind]
    return cls() if source_head is None else cls.from_linear(source_head)
