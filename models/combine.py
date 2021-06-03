import torch
from torch import nn
from models.utils import SimplerLinear, condense_helper, condense_left, release_left
from models.types import activation_type
from utils.types import BaseType
from utils.str_ops import is_numeric
from collections import defaultdict
from sys import stderr

# 'CV2(0:100,200:300);Mul(100:200)'
def get_components(str_compound):
    all_slices = {}
    cmb_slices = {}
    for str_cmb in str_compound.split(';'):
        assert str_cmb.strip(), 'Unnecessary usage!'
        lbr = str_cmb.index('(')
        rbr = str_cmb.index(')')
        assert not str_cmb[rbr + 1:].strip(), 'Invalid syntax'
        assert lbr < rbr, 'Invalid parenthese syntax'
        cmb = str_cmb[:lbr].strip()
        slices = []
        s_starts = defaultdict(int)
        s_ends   = defaultdict(int)
        for str_slice in str_cmb[lbr + 1:rbr].split(','):
            s_start, s_end = str_slice.split(':')
            s_start = int(s_start)
            s_end   = int(s_end  )
            s_starts[s_start] += 1
            s_ends  [s_end  ] += 1
            assert s_start < s_end, 'Invalid slice: start ≥ end'
            slices.append((s_start, s_end))
            all_slices[s_start] = s_end
        cmb_slices[cmb] = slices
        assert not (s_starts.keys() & s_ends.keys()), 'Ambiguous inner continuity detected!'
        assert all(x == 1 for x in s_starts.values()), 'Douplicated starts!'
        assert all(x == 1 for x in s_ends  .values()), 'Douplicated ends!'
    s_end = 0
    assert len(cmb_slices) > 1, 'Unnecessary usage!'
    while all_slices:
        assert s_end in all_slices, f'Slices are not discontinuos after {s_end}!'
        s_end = all_slices.pop(s_end)
    cmb_dims = {}
    for cmb, slices in cmb_slices.items():
        cmb_dims[cmb] = sum(se - ss for ss, se in slices)
    return cmb_dims, cmb_slices

E_COMBINE = 'CV2 CV1 CS2 CS1 EV2 EV1 ES2 ES1 Add Mul Average NV NS BV BS'.split()
def valid_trans_compound(x): # CT.Tanh.3
    if ':' in x:
        try:
            _, cmb_slices = get_components(x)
            if any(':' in cmb for cmb in cmb_slices):
                return False
            return all(cmb in E_COMBINE or valid_trans_compound(cmb) for cmb in cmb_slices)
        except (AssertionError, Exception) as e:
            print(e, file = stderr)
            return False
    if x.startswith('CT') or x.startswith('BT'):
        segs = x.split('-')
        if len(segs) > 3:
            return False
        valid = activation_type.validate(segs[1])
        if len(segs) == 2:
            return valid
        return valid and is_numeric.fullmatch(segs[2])
    return False

combine_type = BaseType(0, as_index = True, default_set = E_COMBINE, validator = valid_trans_compound)
combine_static_type = BaseType(0, as_index = True,
                               default_set = [None] + E_COMBINE,
                               validator = lambda x: x is None or valid_trans_compound(x))

def get_combinator(type_id, in_size = None):
    if ':' in type_id:
        c_cmbs = {}
        c_dims, c_slices = get_components(type_id)
        assert in_size == sum(c_dims.values())
        for cmb, sub_dim in c_dims.items():
            c_cmbs[cmb] = get_combinator(cmb, sub_dim)
        cmb_slices_tuple = tuple((m, c_slices[cmb]) for cmb, m in c_cmbs.items())
        return CompoundCombinator(cmb_slices_tuple)

    types = {c.__name__:c for c in (Add, Mul, Average)}
    if type_id in types:
        return types[type_id]()
    return Interpolation(type_id, in_size)

class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rightwards_or_lhs, embeddings_or_rhs, existences_or_phy_jnt):
        if rightwards_or_lhs is not None and rightwards_or_lhs.shape == embeddings_or_rhs.shape:
            return self.disco_forward(rightwards_or_lhs, embeddings_or_rhs, existences_or_phy_jnt)
        else:
            return self.conti_forward(rightwards_or_lhs, embeddings_or_rhs, existences_or_phy_jnt)

    def disco_forward(self, lhs, rhs, phy_jnt):
        cmp_emb = self.compose(lhs, rhs, phy_jnt)
        return torch.where(phy_jnt, cmp_emb, lhs)
    
    def conti_forward(self, rightwards, embeddings, existences):
        if existences is None:
            assert rightwards is None
            lw_emb = embeddings[1:]
            rw_emb = embeddings[:-1]
            return self.compose(lw_emb, rw_emb, None)
        if rightwards is None:
            lw_emb = embeddings[:,  1:]
            rw_emb = embeddings[:, :-1]
            lw_ext = existences[:,  1:]
            rw_ext = existences[:, :-1]
        else:
            lw_ext = (existences & ~rightwards)[:,  1:]
            rw_ext = (existences &  rightwards)[:, :-1]
            lw_relay = lw_ext & ~rw_ext
            rw_relay = ~lw_ext & rw_ext

            right = rightwards.type(embeddings.dtype)
            # right.unsqueeze_(-1)
            lw_emb = embeddings[:,  1:] * (1 - right)[:,  1:]
            rw_emb = embeddings[:, :-1] *      right [:, :-1]

        new_jnt = lw_ext & rw_ext
        new_ext = lw_ext | rw_ext
        add_emb = lw_emb + rw_emb
        cmp_emb = self.compose(lw_emb, rw_emb, new_jnt)
        if cmp_emb is None:
            new_emb = add_emb
        else:
            new_emb = torch.where(new_jnt, cmp_emb, add_emb)

        if rightwards is None:
            return new_ext, new_emb
        return new_ext, new_jnt, lw_relay, rw_relay, new_emb

    def compose(self, lw_emb, rw_emb, is_jnt):
        return lw_emb + rw_emb # Default by add

class Mul(Add):
    def compose(self, lw_emb, rw_emb, is_jnt):
        return lw_emb * rw_emb

class Average(Add):
    def compose(self, lw_emb, rw_emb, is_jnt):
        return lw_emb * rw_emb / 2

# class Max(Add):
# class Cos(Add):

class CompoundCombinator(Add):
    def __init__(self, cmb_slices_tuple):
        super().__init__()
        # {((dim_slice, ...)): cmb}
        self._cmb_slices_tuple = cmb_slices_tuple
        self._cmb0 = cmb_slices_tuple[0][0]
        self._cmb1 = cmb_slices_tuple[1][0]
        if len(self._cmb_slices_tuple) > 2:
            self._cmb2 = cmb_slices_tuple[2][0]
        assert len(cmb_slices_tuple) < 4, 'Should not be too complex'

    def compose(self, lw_emb, rw_emb, is_jnt):
        all_embs = {}
        for cmb, dim_slices in self._cmb_slices_tuple:
            lw_slices = []
            rw_slices = []
            for dim_start, dim_end in dim_slices: # fan-in
                lw_slices.append(lw_emb[:, :, dim_start:dim_end])
                rw_slices.append(rw_emb[:, :, dim_start:dim_end])
            lw_slices = torch.cat(lw_slices, dim = 2) if len(lw_slices) > 1 else lw_slices.pop()
            rw_slices = torch.cat(rw_slices, dim = 2) if len(rw_slices) > 1 else rw_slices.pop()
            cw_emb = cmb.compose(lw_slices, rw_slices, is_jnt)
            seg_start = 0
            for dim_start, dim_end in dim_slices: # fan-out
                seg_end = seg_start + dim_end - dim_start
                all_embs[dim_start] = cw_emb[:, :, seg_start:seg_end] if len(dim_slices) > 1 else cw_emb
                seg_start = seg_end
        return torch.cat([all_embs.pop(s) for s in sorted(all_embs)], dim = 2)

class Interpolation(Add):
    def __init__(self, type_id, in_size, out_size = None, bias = True):
        super().__init__()
        use_condenser = False
        if out_size is None:
            out_size = in_size
        else:
            raise NotImplementedError()

        if type_id in E_COMBINE:
            activation = nn.Sigmoid()
        else:
            segs = type_id.split('-')
            activation = activation_type[segs[1]]()
            scale = float(segs[2]) if len(segs) == 3 else None
            print(type_id, activation, scale)
        self._activation = activation
        if type_id[0] == 'N':
            assert bias, f'invalid {type_id} without a bias parameter'
            if type_id == 'NV':
                itp_ = SimplerLinear(in_size, weight = False)
            elif type_id == 'NS':
                itp_ = SimplerLinear(1, weight = False)
            # extra_repr = type_id + ': ' + str(itp_)
            self._itp = itp_
            def _compose(lw, rw):
                itp = activation(itp_(1))
                return (1 - itp) * lw + itp * rw
        elif type_id[0] == 'C':
            if type_id =='CV2' or type_id.startswith('CT-'):
                # use_condenser = True # TESTED! 100sps SLOWER
                itp_l = nn.Linear(in_size, out_size, bias = False)
                itp_r = nn.Linear(in_size, out_size, bias = bias)
            elif type_id == 'CS2':
                itp_l = nn.Linear(in_size, 1, bias = False)
                itp_r = nn.Linear(in_size, 1, bias = bias)
            elif type_id == 'CV1':
                itp_l = SimplerLinear(in_size, bias = False)
                itp_r = SimplerLinear(in_size, bias = bias)
            elif type_id == 'CS1':
                itp_l = SimplerLinear(1, bias = False)
                itp_r = SimplerLinear(1, bias = bias)
            # extra_repr = f'{type_id}: {itp_l} & {itp_r}'
            self._itp_l = itp_l # pytorch direct register? yes
            self._itp_r = itp_r # even not uses
            if type_id == 'CT':
                def _compose(lw, rw):
                    tsf = itp_l(lw) + itp_r(rw) # Concatenate
                    if scale is not None:
                        tsf = activation(tsf) * scale
                    else:
                        tsf = activation(tsf)
                    return tsf
            else:
                def _compose(lw, rw):
                    # lw =  # Either Vector
                    # rw =  # Or Scalar to *
                    itp = activation(itp_l(lw) + itp_r(rw)) # Concatenate
                    return (1 - itp) * lw + itp * rw
        elif type_id[0] == 'E':
            if type_id =='EV2':
                itp_ = nn.Linear(in_size, out_size, bias = bias)
            elif type_id == 'ES2':
                itp_ = nn.Linear(in_size, 1, bias = bias)
            elif type_id == 'EV1':
                itp_ = SimplerLinear(in_size, bias = bias)
            elif type_id == 'ES1':
                itp_ = SimplerLinear(1, bias = bias)
            self._itp = itp_
            def _compose(lw, rw):
                itp = activation(itp_(lw) + itp_(rw))
                return (1 - itp) * lw + itp * rw
        elif type_id[0] == 'B':
            if type_id == 'BV' or type_id.startswith('BT-'):
                use_condenser = True
                itp_ = nn.Bilinear(in_size, in_size, out_size, bias = bias)
            elif type_id == 'BS':
                itp_ = nn.Bilinear(in_size, in_size, 1, bias = bias)
            # extra_repr = type_id + ': ' + str(itp_)
            self._itp = itp_
            if type_id == 'BT':
                def _compose(lw, rw):
                    if type_id != 'BS':
                        lw = lw.contiguous()
                        rw = rw.contiguous()
                    if scale is not None:
                        tsf = activation(itp_(lw, rw)) * scale
                    else:
                        tsf = activation(itp_(lw, rw))
                    return tsf
            else:
                def _compose(lw, rw):
                    if type_id != 'BS':
                        lw = lw.contiguous()
                        rw = rw.contiguous()
                    itp = itp_(lw, rw)
                    itp = activation(itp)
                    return (1 - itp) * lw + itp * rw

        # self._extra_repr = extra_repr
        self._use_condenser = use_condenser
        self._compose = _compose

    def compose(self, lw_emb, rw_emb, is_jnt):
        if self._use_condenser and isinstance(is_jnt, torch.Tensor) and is_jnt.any():
            helper = condense_helper(is_jnt.squeeze(dim = 2), as_existence = True)
            cds_lw, seq_idx = condense_left(lw_emb, helper, get_indice = True)
            cds_rw          = condense_left(rw_emb, helper)
            # print(f'helper {helper[1]} {cds_lw.shape}, {is_jnt.shape} {is_jnt.sum()}')
            cds_cmb = self._compose(cds_lw, cds_rw)
            return release_left(cds_cmb, seq_idx)
        return self._compose(lw_emb, rw_emb)

    def itp_rhs_bias(self):
        if hasattr(self, '_itp_r'):
            if hasattr(self._itp_r, 'bias'):
                return self._activation(self._itp_r.bias)
        elif hasattr(self._itp, 'bias'):
            return self._activation(self._itp.bias)
    # def extra_repr(self):
    #     return self._extra_repr