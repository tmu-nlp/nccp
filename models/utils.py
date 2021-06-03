import torch
class PCA:
    def __init__(self, emb, k = 9):
        emb_mean = torch.mean(emb, dim = 0)
        emb_shifted = emb - emb_mean
        emb_cov = torch.matmul(emb_shifted.T, emb_shifted)
        val, vec = torch.eig(emb_cov, True)
        _, idx = val[:, 0].topk(k) # ignore value & image part
        self._bases = vec[:, idx] # both tensorflow and torch use dim = 1

    def __call__(self, emb):
        m_ = (emb * emb).mean(-1, keepdim = True)
        pc = torch.matmul(emb, self._bases)
        return torch.cat([m_, pc], -1)

def mean_stdev(weights, dim = -1):
    wm = weights.mean(dim = dim, keepdim = True)
    sd = (weights - wm) ** 2
    sd = sd.sum(dim = dim, keepdim = True) / (weights.shape[dim] - 1)
    sd = torch.sqrt(sd)
    return torch.cat([wm, sd], dim = dim)

def fraction(cnt_n, cnt_d = None, dtype = torch.float32):
    if cnt_d is None:
        return cnt_n.sum().type(dtype) / cnt_n.nelement()
    return cnt_n.sum().type(dtype) / cnt_d.sum().type(dtype)

def hinge_score(hinge_logits, inplace):
    if inplace:
        hinge_scores = hinge_logits
        hinge_scores += 1
    else:
        hinge_scores = hinge_logits + 1
        
    hinge_scores /= 2
    hinge_scores[hinge_scores < 0] = 0
    hinge_scores[hinge_scores > 1] = 1

    if not inplace:
        return hinge_scores

import math
from torch.nn import Module, Parameter, init
class SimplerLinear(Module):
    __constants__ = ['bias', 'in_features']
    def __init__(self, in_features, weight = True, bias = True):
        assert weight or bias
        super(SimplerLinear, self).__init__()
        self.in_features = in_features
        if weight:
            self.weight = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        if self.weight is not None:
            init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.weight is not None:
            input = input * self.weight
        if self.bias is not None:
            input = input + self.bias
        return input

    def extra_repr(self):
        return 'in_features={}, bias={}'.format(
            self.in_features, self.bias is not None
        )

class Bias(Module):
    __constants__ = ['bias', 'in_features']
    def __init__(self, in_features):
        super(Bias, self).__init__()
        self.in_features = in_features
        self.bias = Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        init.uniform_(self.bias, -bound, bound)

    def forward(self):
        return self.bias

    def extra_repr(self):
        return 'in_features={}'.format(self.in_features)

class GaussianCodebook(Module):
    __constants__ = ['bias', 'codebook' 'io_features']
    def __init__(self, in_dim, num_codes, prefix_dims = 0, coeff = 0):
        super(GaussianCodebook, self).__init__()

        self.io_features = in_dim, num_codes
        self._coeff = coeff
        size = [1 for _ in range(prefix_dims)]
        size += [in_dim, num_codes]
        self.codebook = Parameter(torch.Tensor(*size))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.codebook, a = math.sqrt(5))

    def forward(self, x):
        diff = (x.unsqueeze(dim = 3) - self.codebook) ** 2
        return diff.mean(dim = -2) # [?...?, c]

    def repulsion(self, coeff = 0):
        if coeff <= 0:
            if self._coeff <= 0:
                return 0
            coeff = self._coeff
        in_dim, num_codes = self.io_features
        code = self.codebook.view(1, in_dim, num_codes)
        _code = code.transpose(0, 2)
        ccd = (code - _code) ** 2 # [c, h, c]
        ccd = ccd.mean(dim = 1) # [c, c]
        ccg = - ccd * coeff
        ccg.exp_() # repulsion
        return ccg.sum() # minimize

    def distance(self, logtis, oracle):
        # [b,s,c][b,s]
        oracle = oracle.unsqueeze(dim = 2)
        distance = logtis.gather(dim = 2, index = oracle)
        distance.squeeze_(dim = 2)
        return distance

    def extra_repr(self):
        in_features, out_features = self.io_features
        return 'in_dim={}, num_codes={}'.format(
            in_features, out_features
        )

from torch.nn import Linear, Softmax, Softmin
def get_logit_layer(logit_type):
    if logit_type in ('affine', 'linear'):
        Net = lambda i_size, o_size: Linear(i_size, o_size, bias = logit_type == 'affine')
        argmax = True
        score_fn = Softmax
    elif logit_type.startswith('codebook'):
        argmax = False
        if '|' in logit_type:
            bar = logit_type.index('|') + 1
            repulsion = float(logit_type[bar:])
        else:
            repulsion = 0
        score_fn = Softmin
        Net = lambda i_size, o_size: GaussianCodebook(i_size, o_size, coeff = repulsion)
    return Net, argmax, score_fn

def bos_mask(seq_len, offset):
    mask = torch.arange(seq_len, device = offset.device)[None]
    return mask < offset[:, None]

def eos_mask(seq_len, length):
    mask = torch.arange(seq_len, device = length.device)[None]
    return mask >= length[:, None]

def condense_splitter(right_layer, joint_layer, existence):
    batch_size, old_batch_len = existence.shape
    agree_orient = right_layer[:, :-1] & existence[:, :-1] # lhs2r
    agree_orient &= right_layer[:, 1:].logical_not() & existence[:, 1:] # rhs2l
    swapping_spot = agree_orient & joint_layer.logical_not()
    physical_joint = agree_orient & joint_layer
    bool_pads = torch.zeros(batch_size, 1, dtype = physical_joint.dtype, device = physical_joint.device)
    rhs_exist = torch.cat([bool_pads, physical_joint], dim = 1)
    lhs_exist = rhs_exist.logical_not()
    lhs_exist &= existence
    rhs_exist &= existence
    lhs_seq_idx = torch.cumsum(lhs_exist, dim = 1)
    rhs_seq_idx = lhs_seq_idx.roll(1, 1) 
    lhs_seq_idx *= lhs_exist
    rhs_seq_idx *= rhs_exist
    max_len = lhs_seq_idx.max() + 1
    lhs_helper = lhs_seq_idx, max_len, True
    rhs_helper = rhs_seq_idx, max_len, True
    return lhs_helper, rhs_helper, physical_joint, swapping_spot, bool_pads

def condense_helper(existence_or_start,
                    as_existence      = False,
                    offset            = None,
                    get_rid_of_last_k = 0):
    seq_idx = torch.cumsum(existence_or_start, 1)

    if offset is not None:
        if isinstance(offset, int):
            seq_idx += offset
        else:
            seq_idx += offset[:, None]

    # 0 used as a dump
    if get_rid_of_last_k:
        max_idx = seq_idx[:, -1:] - get_rid_of_last_k
        seq_idx[seq_idx > max_idx] = 0
        
    if as_existence:
        seq_idx *= existence_or_start
        
    max_len = seq_idx.max() + 1

    return seq_idx, max_len, as_existence

def condense_left(hidden, helper,
                  out_len    = None,
                  get_cumu   = False,
                  get_indice = False,
                  skip_dump0 = True):
    # from hidden redirected by seq_idx to zeros, throw away 1:
    hidden_shape = list(hidden.shape)
    seq_idx, max_len, as_existence = helper
    if out_len is None:
        hidden_shape[1] += 1
        truncate = max_len
    elif hidden_shape[1] < out_len:
        hidden_shape[1] = out_len + 1
        truncate = None
    else:
        hidden_shape[1] += 1
        truncate = out_len + 1
        
    if len(hidden_shape) == 3:
        seq_idx = seq_idx.unsqueeze(dim = -1)
        cumula_shape = hidden_shape[:-1] + [1]
    else:
        cumula_shape = hidden_shape

    base = torch.zeros(*hidden_shape, device = hidden.device, dtype = hidden.dtype)
    if as_existence:
        base = base.scatter(1, seq_idx.expand_as(hidden), hidden)
    else:
        base = base.scatter_add(1, seq_idx.expand_as(hidden), hidden)

    # 0 used as a dump
    base = base[:, skip_dump0:] if truncate is None else base[:, skip_dump0:truncate]

    if get_cumu:
        cumu = torch.zeros(*cumula_shape, device = hidden.device, dtype = seq_idx.dtype)
        cumu.scatter_add_(1, seq_idx, torch.ones_like(seq_idx))
        cumu = cumu[:, skip_dump0:] if truncate is None else cumu[:, skip_dump0:truncate]
        if as_existence:
            cumu = cumu > 0

    num = get_indice + get_cumu
    if num == 0:
        return base
    if num == 1:
        return base, cumu if get_cumu else seq_idx
    return base, cumu, seq_idx

def release_left(base, seq_idx):
    base_shape = base.shape
    if len(base_shape) == 3:
        batch_size, seq_len, model_dim = base_shape
        left = torch.zeros(batch_size, 1, model_dim, dtype = base.dtype, device = base.device)
        seq_idx = seq_idx.expand(batch_size, -1, model_dim)
    else:
        batch_size, seq_len = base_shape
        left = torch.zeros(batch_size, 1, dtype = base.dtype, device = base.device)
    base = torch.cat([left, base], dim = 1)
    return base.gather(1, seq_idx) # all non-dump index > 0

def shuffle_some_from(new_size, base, existence):
    buffer = base.masked_select(existence).reshape(-1, base.shape[-1])
    buffer_size = buffer.shape[0]
    indices = torch.randint(high = buffer_size, size = (new_size,), device = base.device)
    some = torch.index_select(buffer, 0, indices)

    batch_ids = torch.where(existence)[0]
    batch_ids = torch.index_select(batch_ids, 0, indices)

    continuous = indices[1:] - indices[:-1] == 1
    continuous &= batch_ids[1:] == batch_ids[:-1]

    return some, continuous

def blocky_softmax(sections,
                   logits,
                   domains = None,
                   values = None,
                   pad_first = True,
                   map_pad_weights_to_zero = True,
                   use_ones_as_values = False):
    # 0 in sections indicate a dump
    # [1 2 2 3 0] -> [0 1 2 3]
    idx = sections.unsqueeze(dim = 2)
    expanded_idx = idx.expand_as(logits)
    bs, sl, eb = logits.shape
    domain_sl = sections.max() + 1
    if domains is not None:
        padded = torch.zeros(bs, 1, eb, device = logits.device)
        padded = torch.cat([padded, domains] if pad_first else [domains, padded], dim = 1)
        padded = torch.gather(padded, 1, expanded_idx)
        logits = logits + padded # empty pad section will pollute logits
    exp_logits = logits.exp()
    zer_domains = torch.zeros(bs, domain_sl, eb, device = logits.device)
    exp_domains = zer_domains.scatter_add(1, expanded_idx, exp_logits)
    exp_domains = torch.gather(exp_domains, 1, expanded_idx)
    if map_pad_weights_to_zero: # polluted logits yield zero weights
        if pad_first:
            section_idx = idx != 0
        else:
            section_idx = idx != sections.max(dim = 1).values[:, None, None]
        exp_logits = section_idx * exp_logits
    weights = exp_logits / exp_domains
    if use_ones_as_values:
        from utils.shell_io import byte_style
        print(byte_style('[WARNING: use_ones_as_values only for debugging!!!]', '1'))
        print(byte_style('[+PROMPT: final_domains should have either 0 or 1]', '3'))
        values = torch.ones_like(weights)
    if values is None:
        return weights
    final_domains = zer_domains.scatter_add(1, expanded_idx, weights * values)
    final_domains = final_domains[:, 1:] if pad_first else final_domains[:, :-1]
    return weights, final_domains

def blocky_max(sections, values, has_batch_dim = True):
    m = sections.max() + 1
    if has_batch_dim:
        _, s = sections.shape
        s_range = torch.arange(s, device = sections.device)[None, :]
        m_range = torch.arange(m, device = sections.device)[None, :, None]
        mask = sections[:, None, :].repeat(1, m, 1) == m_range # [b, m, s]
        m_dim, s_dim = 1, 2
    else:
        s, = sections.shape
        s_range = torch.arange(s, device = sections.device)
        m_range = torch.arange(m, device = sections.device)[:, None]
        mask = sections[None, :].repeat(m, 1) == m_range
        m_dim, s_dim = 0, 1
    any_mask = mask.any(s_dim, keepdim = True)
    m_values = mask * values.unsqueeze(m_dim) # [b, m, s] * [b, 1, s]
    # _, indices = m_values.max(s_dim, keepdim = True)
    # print(indices)
    # print(m_values.argmax(s_dim, keepdim = True))# may have more than one...
    mask = s_range == m_values.argmax(s_dim, keepdim = True)
    mask &= any_mask
    return mask.any(m_dim)

def birnn_fwbw(embeddings, pad = None, existence = None):
    bs, sl, ed = embeddings.shape
    half_dim = ed >> 1
    if pad is None:
        pad = torch.ones(bs, 1, half_dim, dtype = embeddings.dtype, device = embeddings.device)
    else:
        pad = pad.repeat(bs, 1, 1) # not efficient as expand
    embeddings = embeddings.view(bs, sl, 2, half_dim)
    fw = embeddings[:, :, 0]
    bw = embeddings[:, :, 1]
    if existence is not None:
        within_seq = existence.unsqueeze(dim = 2)
        fw = torch.where(within_seq, fw, pad)
        bw = torch.where(within_seq, bw, pad)
    fw = torch.cat([pad, fw], dim = 1)
    bw = torch.cat([bw, pad], dim = 1)
    return fw, bw

def fencepost(fw, bw, splits):
    batch = torch.arange(fw.shape[0], device = fw.device)[:, None]
    fw = fw[batch, splits]
    fw_diff = fw[:, 1:] - fw[:, :-1]
    bw = bw[batch, splits]
    bw_diff = bw[:, :-1] - bw[:, 1:]
    return torch.cat([fw_diff, bw_diff], dim = 2)

def batch_insert(base, indice, subject = None):
    # base [b, s], obj [b, s-]
    batch_size, batch_len, emb_size = base.shape
    bs, extra_len = indice.shape
    assert batch_size == bs, 'batch_size should match'
    batch_range = torch.arange(batch_size, device = base.device)[:, None]
    extended_len = batch_len + extra_len

    inc_start = torch.zeros(batch_size, extended_len, dtype = indice.dtype, device = base.device)
    inc_start.scatter_add_(1, indice, torch.ones_like(indice)) # mark base (0) and subject (1) 224..
    # inc_start[batch_range, indice] += 1 does not work
    base_inc = inc_start.cumsum(dim = 1) # i-th insetion  0011222.. 
    base_inc = base_inc[:, :batch_len]
    base_inc += torch.arange(batch_len, device = base.device)[None] # base position 0134678..
    base_inc = base_inc.unsqueeze(dim = 2)

    new_base = torch.zeros(batch_size, extended_len, emb_size, dtype = base.dtype, device = base.device)
    new_base = new_base.scatter_add(1, base_inc.expand_as(base), base)
    if subject is None:
        return new_base
    
    subj_inc = torch.arange(extra_len, dtype = indice.dtype, device = base.device)[None]
    subj_inc = subj_inc + indice # 224 -> 236
    subj_inc.unsqueeze_(dim = 2)
    return new_base.scatter_add(1, subj_inc.expand_as(subject), subject)

def bool_start_end(continuous):
    batch_size, batch_len = continuous.shape
    seq_idx = torch.arange(batch_len, device = continuous.device)
    seq_idx = continuous * seq_idx
    max_idx = seq_idx.argmax(dim = 1)
    seq_idx[continuous.logical_not()] = batch_len
    min_idx = seq_idx.argmin(dim = 1)
    continuity = continuous.any(dim = 1)
    batch_dim = torch.arange(batch_size, device = continuous.device)
    return batch_dim[continuity], min_idx[continuity], max_idx[continuity]