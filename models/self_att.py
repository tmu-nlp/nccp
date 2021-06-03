import torch
from torch import nn
from math import log, sqrt

def timing_signal_1d(seq_dim,
                     model_dim,
                     min_timescale = 1.0,
                     max_timescale = 1.0e4,
                     seq_start     = 0,
                     seq_len       = None):
    assert 0 < min_timescale < max_timescale
    if seq_len is None: # original forward
        num_timescales = model_dim // 2
        device = None
    else: # augmented backward
        num_timescales = model_dim // 4
        device = seq_len.device

    log_timescale_increment = log(max_timescale / min_timescale)
    log_timescale_increment /= max(num_timescales - 1, 1)
    inv_timescales = torch.arange(num_timescales, device = device) * -log_timescale_increment
    torch.exp_(inv_timescales)
    inv_timescales *= min_timescale
    inv_timescales.unsqueeze_(0)
    
    position = torch.arange(-seq_start, seq_dim - seq_start, device = device)
    # print(position)
    position.unsqueeze_(1)
    scaled_time = position * inv_timescales
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
    signal.unsqueeze_(0)
    
    if seq_len is not None:
        inv_position = torch.arange(seq_dim - 1, -1, -1, device = device)
        seq_pad_len = seq_dim - seq_len - seq_start
        if isinstance(seq_len, torch.Tensor):
            batch_size, = seq_len.shape
            signal = signal.repeat(batch_size, 1, 1)
            inv_position = inv_position[None, :] - seq_pad_len[:, None] # [batch, seq]
            # print(inv_position)
        else:
            inv_position -= seq_pad_len
            # print(inv_position)
            inv_position.unsqueeze_(0) # [1, seq]
        inv_position.unsqueeze_(2) # [1, seq, 1]

        inv_timescales.unsqueeze_(0) # [1, 1, timescale]
        inv_scaled_time = inv_position * inv_timescales # [batch, seq, timescale]
        signal = torch.cat([signal, torch.sin(inv_scaled_time), torch.cos(inv_scaled_time)], 2)

    return signal

# print((timing_signal_1d(4, 12, seq_start = 1, seq_len = 3) == timing_signal_1d(4, 12, seq_start = 1, seq_len = torch.arange(3) + 1)) * 1)
class NormalizationLayer(nn.Module):
    def __init__(self, op_dims):
        super().__init__()
        self._op_dims = op_dims
        self.scale = nn.Parameter(torch.Tensor(1))
        self.bias  = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / sqrt(5)
        nn.init.uniform_(self.scale, -bound, bound)
        nn.init.uniform_(self.bias,  -bound, bound)

    def forward(self, base):
        mean = base.mean(self._op_dims, keepdim = True)
        stdv = (base - mean) ** 2
        stdv = stdv.mean(self._op_dims, keepdim = True)
        stdv = stdv.sqrt()

        batch = (base - mean) / stdv
        return batch * self.scale + self.bias


class SelfAttentionLayer(nn.Module):
    def __init__(self,
                 model_dim,
                 num_head,
                 dropout,
                 norm_dims,
                 activation):
        super().__init__()
        assert model_dim % num_head == 0
        head_dim = model_dim // num_head
        self._qkv = nn.Linear(model_dim, 3 * model_dim)
        self._ffnn_1 = nn.Linear(model_dim, model_dim)
        self._ffnn_2 = nn.Linear(model_dim, model_dim)
        self._args = norm_dims, model_dim, num_head, head_dim, sqrt(head_dim)
        self._softmax = nn.Softmax(dim = 2)
        self._act = activation()
        self._n1 = NormalizationLayer(norm_dims)
        self._n2 = NormalizationLayer(norm_dims)

    def forward(self, base):
        norm_dims, split_len, num_head, head_dim, head_z = self._args

        qkv = self._qkv(base)
        q, k, v = qkv.split(split_len, dim = 2) # [batch, seq, num_head * head_dim]
        dim_start = 0
        hidden_heads = []
        for _ in range(num_head):
            dim_end = dim_start + head_dim
            qi = q[:, :, dim_start:dim_end] # [batch, seq, head_dim]
            ki = k[:, :, dim_start:dim_end].transpose(1, 2) # [batch, head_dim, seq] = [batch, seq, seq]
            vi = v[:, :, None, dim_start:dim_end] # [batch, seq, 1, head_dim]
            aij = torch.matmul(qi, ki) / head_z
            aij = self._softmax(aij).unsqueeze(3)
            avi = aij * vi
            avi = avi.sum(dim = 2)
            hidden_heads.append(avi)
            dim_start = dim_end
        hidden_heads = torch.cat(hidden_heads, 2)
        hidden = self._n1(self._ffnn_1(hidden_heads) + base) # to be n1
        return self._n2(self._act(self._ffnn_2(hidden)) + hidden) # n2

class SelfAttention(nn.Module):
    def __init__(self,
                 model_dim,
                 num_head,
                 num_layer,
                 dropout = None,
                 norm_dims = (1, 2),
                 seq_start = 1,
                 activation = nn.ReLU):
        super().__init__()
        args = model_dim, num_head, dropout, norm_dims, activation
        layers = []
        for lid in range(num_layer):
            layer = SelfAttentionLayer(*args)
            layers.append(layer)
            super().add_module(f'layer_{lid}', layer) # nn.ModuleList failed
        self._seq_start_layers = seq_start, layers

    def forward(self, base, seq_len = None):
        seq_start, layers = self._seq_start_layers
        if not layers:
            return base
            
        batch_size, seq_dim, input_dim = base.shape
        base = base + timing_signal_1d(seq_dim, input_dim, seq_start = seq_start, seq_len = seq_len)
        for layer in layers:
            base = layer(base)
        return base

class Singular(nn.Module):
    def __init__(self,
                 model_dim,
                 num_head,
                 num_layer,
                 dropout,
                 norm_dims = 1,
                 seq_start = 1,
                 activation = nn.ReLU):
        super().__init__()
        self._seq_start_layer = seq_start, num_layer
        self._singular = SelfAttentionLayer(model_dim,
                                            num_head,
                                            dropout,
                                            norm_dims,
                                            activation)

    def forward(self, base, seq_len = None):
        seq_start, num_layer = self._seq_start_layer
        batch_size, seq_dim, input_dim = base.shape
        base = base + timing_signal_1d(seq_dim, input_dim, seq_start = seq_start, seq_len = seq_len)
        for _ in range(num_layer):
            base = self._singular(base)
        return base

# base = torch.ones(2, 3, 32)
# layer = SelfAttention(32, 4, 5, None)
# print(layer)
# layer(base, None)