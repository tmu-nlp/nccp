import torch
from torch import nn
from utils.types import BaseWrapper, BaseType, true_type, false_type

def valid_codebook(name):
    if name.startswith('codebook'):
        if '|' in name:
            bar = name.index('|') + 1
            try:
                bar = float(name[bar:])
            except:
                return False
        return bar >= 0
    return False

logit_type = BaseType('affine', default_set = ('affine', 'linear', 'codebook'), validator = valid_codebook)

to_name = lambda x: x.__name__
rnn_module_type = BaseType(0, as_index = True, as_exception = True, default_set = BaseWrapper.from_gen((nn.LSTM, nn.GRU), to_name))
activation_type = BaseType(0, as_index = True, as_exception = True, default_set = BaseWrapper.from_gen((nn.ReLU, nn.ReLU6, nn.Softplus,# end == 0, nn.GELU
                                                                                   nn.LeakyReLU, nn.ELU, nn.CELU, nn.SELU, nn.RReLU, # end < 0
                                                                                   nn.Sigmoid, nn.LogSigmoid,
                                                                                   nn.Tanh, nn.Softsign, nn.Hardtanh, # -<0<+
                                                                                   nn.Tanhshrink, nn.Softshrink, nn.Hardshrink), to_name)) # -0+

act_fasttext = BaseType(None, as_index = True, as_exception = True, default_set = BaseWrapper.from_gen((nn.Tanh, nn.Softsign), to_name))

continuous_attention_hint = dict(unit       = false_type,
                                 state      = false_type,
                                 difference = true_type,
                                 boundary   = false_type,
                                 before     = false_type,
                                 after      = false_type)

discontinuous_attention_hint = continuous_attention_hint.copy()
discontinuous_attention_hint.pop('boundary')

fence_vote = BaseType(0, as_index = True, default_set = (None, 'state.dot', 'unit.dot', 'state.cat', 'unit.cat'))

hinge_bias = lambda x: x - 0.5

from models.self_att import SelfAttention
class SAL(nn.Module):
    def __init__(self, in_size, out_size, num_layers):
        super().__init__()
        self._sa = SelfAttention(in_size, 10, num_layers)
        self._fi = nn.Linear(in_size, out_size)
        self._a1 = nn.ReLU()
        self._a2 = nn.Tanh()

    def forward(self, base, seq_len):
        return self._a2(self._fi(self._a1(self._sa(base, seq_len))))

class LSA(nn.Module):
    def __init__(self, in_size, out_size, num_layers):
        super().__init__()
        self._fi = nn.Linear(in_size, out_size)
        self._sa = SelfAttention(out_size, 10, num_layers, norm_dims = 2)
        self._a1 = nn.ReLU()
        self._a2 = nn.Tanh()

    def forward(self, base, seq_len):
        return self._a2(self._sa(self._a1(self._fi(base)), seq_len))

orient_module = BaseType(0, as_index = True, as_exception = True, default_set = BaseWrapper.from_gen((nn.LSTM, nn.GRU, SAL, LSA), to_name))

finfo = torch.finfo(torch.get_default_dtype())
fmin = finfo.min
fmax = finfo.max