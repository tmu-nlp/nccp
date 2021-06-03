import torch
from torch import nn, Tensor
from utils.math_ops import s_index
from utils.types import BaseType, true_type, frac_4, frac_2, BaseWrapper
from utils.types import orient_dim, num_ori_layer, false_type

from models.combine import get_combinator, combine_type
stem_config = dict(orient_dim   = orient_dim,
                   combine_type = combine_type,
                   num_layers   = num_ori_layer,
                   rnn_drop_out = frac_2,
                   drop_out     = frac_4,
                   trainable_initials = false_type)

from models.utils import condense_helper, condense_left
from itertools import count
class Stem(nn.Module):
    def __init__(self,
                 model_dim,
                 orient_dim,
                 combine_type,
                 num_layers,
                 rnn_drop_out,
                 trainable_initials,
                 drop_out):
        super().__init__()
        hidden_size = orient_dim // 2
        self.orient_emb = nn.LSTM(model_dim, hidden_size,
                                  num_layers    = num_layers,
                                  bidirectional = True,
                                  batch_first   = True,
                                  dropout = rnn_drop_out if num_layers > 1 else 0)
        self._dp_layer = nn.Dropout(drop_out)
        self.orient = nn.Linear(orient_dim, 1)
        self.combine = get_combinator(combine_type, model_dim)
        if trainable_initials:
            c0 = torch.randn(num_layers * 2, 1, hidden_size)
            h0 = torch.randn(num_layers * 2, 1, hidden_size)
            self._c0 = nn.Parameter(c0, requires_grad = True)
            self._h0 = nn.Parameter(h0, requires_grad = True)
            self._h0_act = nn.Tanh()
            self._initial_size = hidden_size
        else:
            self.register_parameter('_h0', None)
            self.register_parameter('_c0', None)
            self._initial_size = None

    def blind_combine(self, unit_hidden, existence = None):
        return self.combine(None, unit_hidden, existence)

    def get_h0c0(self, batch_size):
        if self._initial_size:
            c0 = self._c0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._h0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._h0_act(h0)
            h0c0 = h0, c0
        else:
            h0c0 = None
        return h0c0

    def predict_orient(self, unit_hidden, h0c0):
        orient_hidden, _ = self.orient_emb(unit_hidden, h0c0)
        orient_hidden = self._dp_layer(orient_hidden)
        return self.orient(orient_hidden)

    def forward(self,
                existence,
                unit_hidden,
                height = 0,
                **kw_args):
        batch_size, seq_len, _ = existence.shape
        h0c0 = self.get_h0c0(batch_size)

        if height == 0:
            (layers_of_unit, layers_of_existence, layers_of_orient,
             trapezoid_info) = self.triangle_forward(existence, unit_hidden, batch_size, seq_len, h0c0, **kw_args)
        else:
            (layers_of_unit, layers_of_existence, layers_of_orient,
             trapezoid_info) = self.trapozoids_forward(height, existence, unit_hidden, batch_size, seq_len, h0c0, **kw_args)

        layers_of_unit.reverse()
        layers_of_orient.reverse()
        layers_of_existence.reverse()

        unit_hidden = torch.cat(layers_of_unit,   dim = 1)
        orient      = torch.cat(layers_of_orient, dim = 1)
        existence   = torch.cat(layers_of_existence, dim = 1)

        return unit_hidden, orient, existence, trapezoid_info

    def triangle_forward(self,
                         existence,
                         unit_hidden,
                         batch_size, seq_len, h0c0,
                         supervised_orient = None, **kw_args):
        layers_of_orient = []
        layers_of_unit   = []
        layers_of_existence = []
        num_layers = seq_len

        teacher_forcing = isinstance(supervised_orient, Tensor)
        modification = not teacher_forcing and isinstance(supervised_orient, tuple)
        if modification:
            offsets, lengths = supervised_orient
            batch_dim = torch.arange(batch_size, device = existence.device)
            ends = offsets + lengths - 1

        for length in range(num_layers, 0, -1):
            orient = self.predict_orient(unit_hidden, h0c0)
            layers_of_orient.append(orient)
            layers_of_unit  .append(unit_hidden)
            layers_of_existence.append(existence)
            if length == 1: break

            if teacher_forcing:
                start = s_index(length - 1)
                end   = s_index(length)
                right = supervised_orient[:, start:end, None]
            elif modification:
                right = orient > 0
                starts = torch.where(offsets < length, offsets, torch.zeros_like(offsets))
                _ends_ = ends - (num_layers - length)
                _ends_ = torch.where( starts < _ends_,  _ends_, torch.ones_like(_ends_) * (length - 1))
                right[batch_dim, starts] = True
                right[batch_dim, _ends_] = False
            else:
                right = orient > 0

            (existence, new_jnt, lw_relay, rw_relay,
             unit_hidden) = self.combine(right, unit_hidden, existence)
        return (layers_of_unit, layers_of_existence, layers_of_orient, None)


    def trapozoids_forward(self,
                           height,
                           existence,
                           unit_hidden,
                           batch_size, seq_len, h0c0,
                           supervised_orient = None, **kw_args):
        layers_of_orient = []
        layers_of_unit   = []
        layers_of_existence = []
        teacher_forcing = isinstance(supervised_orient, Tensor)
        if teacher_forcing:
            end = supervised_orient.shape[1]
        segment, seg_length = [], []

        for l_ in count():
            if not teacher_forcing:
                segment.append(seq_len)
                if l_ % height == 0:
                    seg_length.append(existence.sum(dim = 1)) #
                else:
                    seg_length.append(seg_length[-1] - 1)

            orient = self.predict_orient(unit_hidden, h0c0)
            layers_of_orient.append(orient)
            layers_of_unit  .append(unit_hidden)
            layers_of_existence.append(existence)
            if seq_len == 1: break

            if teacher_forcing:
                start = end - seq_len
                right = supervised_orient[:, start:end, None]
                end   = start
            else:
                right = orient > 0

            (existence, new_jnt, lw_relay, rw_relay,
             unit_hidden) = self.combine(right, unit_hidden, existence)

            if l_ % height == height - 1:
                # import pdb; pdb.set_trace()
                existence.squeeze_(dim = 2) # will soon be replaced
                helper = condense_helper(existence, as_existence = True)
                unit_hidden, existence = condense_left(unit_hidden, helper, get_cumu = True)
                seq_len = unit_hidden.shape[1]
            else:
                seq_len -= 1

        if not teacher_forcing:
            segment.reverse()
            seg_length.reverse()
            seg_length = torch.cat(seg_length, dim = 1)

        return (layers_of_unit, layers_of_existence, layers_of_orient, (segment, seg_length))


from models.utils import PCA
from utils.types import num_ctx_layer, frac_06, hidden_dim
from utils.param_ops import HParams, dict_print
from models.types import act_fasttext
input_config = dict(pre_trained = true_type, activation = act_fasttext, drop_out = frac_4)#, random_unk_prob = frac_06, random_unk_from_id = hidden_dim)

class InputLeaves(nn.Module):
    def __init__(self,
                 model_dim,
                 num_tokens,
                 initial_weight,
                 nil_as_pad,
                #  unk_id,
                 pre_trained,
                 activation,
                 drop_out):#,
                #  random_unk_from_id,
                #  random_unk_prob):
        super().__init__()

        if initial_weight is None: # tokenization without <nil>, <bos> & <eos> are included tuned with others
            fixed_dim = model_dim
            assert not pre_trained
        else: # parsing / sentiment analysis
            # 0: no special; 1: <unk>; 2: <bos> <eos> (w/o <nil>); 3: 1 and 2 (w/o <nil>)
            fixed_num, fixed_dim = initial_weight.shape
            num_special_tokens   = num_tokens - fixed_num
            assert num_special_tokens >= 0

        main_extra_bound = 0
        main_emb_layer = extra_emb_layer = None
        padding_kwarg = dict(padding_idx = 0 if nil_as_pad else None)
        if pre_trained:
            if num_special_tokens > 0: # unk | bos + eos | no <nil>
                main_extra_bound = initial_weight.shape[0]
                main_emb_layer  = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = False)
                extra_emb_layer = nn.Embedding(num_special_tokens, fixed_dim)
            else: # <nil> ... |
                main_emb_layer = nn.Embedding.from_pretrained(torch.as_tensor(initial_weight), freeze = False, **padding_kwarg)
        else: # nil ... unk | ... unk bos eos
            main_emb_layer = nn.Embedding(num_tokens, fixed_dim, **padding_kwarg)
        static_pca = fixed_dim == model_dim

        if activation is None:
            self._act_pre_trained = None
        else:
            self._act_pre_trained = activation()
        
        self._dp_layer = nn.Dropout(drop_out)
        self._main_extra_bound_pad = main_extra_bound, nil_as_pad
        self._input_dim = fixed_dim
        self._main_emb_layer  = main_emb_layer
        self._extra_emb_layer = extra_emb_layer
        self._pca_base = None, static_pca
        self._main_emb_tuned = True

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def has_static_pca(self):
        return self._pca_base[1]

    def flush_pc_if_emb_is_tuned(self):
        pca_base, static = self._pca_base
        assert static, 'has_no_static_pca'
        if self._main_emb_tuned or pca_base is None:
            self._pca_base = PCA(self._main_emb_layer.weight), True
            self._main_emb_tuned = False

    def pca(self, word_emb):
        pca_base, static = self._pca_base
        assert static, 'has_no_static_pca'
        return pca_base(word_emb)

    def forward(self, word_idx, tune_pre_trained):
        bound, nil_as_pad = self._main_extra_bound_pad
        if bound > 0: # [nil] vocab | UNK | BOS EOS
            fix_mask = word_idx < bound
            f0_idx = fix_mask * word_idx
            fb_idx =~fix_mask * (word_idx - bound)
            if tune_pre_trained:
                f0_emb = self._main_emb_layer(f0_idx)
            else:
                with torch.no_grad():
                    f0_emb = self._main_emb_layer(f0_idx)
            fb_emb = self._extra_emb_layer(fb_idx) # UNK BOS EOS must be tuned
            static_emb = torch.where(fix_mask.unsqueeze(-1), f0_emb, fb_emb)
        else:
            emb_layer = self._main_emb_layer or self._extra_emb_layer
            if tune_pre_trained:
                static_emb = emb_layer(word_idx)
            else:
                with torch.no_grad():
                    static_emb = emb_layer(word_idx)
        if nil_as_pad:
            bottom_existence = word_idx > 0
            bottom_existence.unsqueeze_(dim = 2)
            # static_emb = static_emb * bottom_existence: done by padding_idx = 0
        else:
            bottom_existence = torch.ones_like(word_idx, dtype = torch.bool) # obsolete (only for nccp)
            bottom_existence.unsqueeze_(dim = 2)
        self._main_emb_tuned = self.training and tune_pre_trained

        static_emb = self._dp_layer(static_emb)
        if self._act_pre_trained is not None:
            static_emb = self._act_pre_trained(static_emb)
        return static_emb, bottom_existence

state_usage = BaseType(None, as_index = False, as_exception = True,
                       validator   = lambda x: isinstance(x, int),
                       default_set = ('sum_layers', 'weight_layers'))

from models.types import rnn_module_type
contextual_config = dict(num_layers   = num_ctx_layer,
                         rnn_type     = rnn_module_type,
                         rnn_drop_out = frac_2,
                         use_state    = dict(from_cell = true_type, usage = state_usage))

class Contextual(nn.Module):
    def __init__(self,
                 input_dim,
                 model_dim,
                 hidden_dim,
                 num_layers,
                 rnn_type,
                 rnn_drop_out,
                 use_state):
        super().__init__()
        if num_layers:
            assert input_dim % 2 == 0
            assert model_dim % 2 == 0
            rnn_drop_out = rnn_drop_out if num_layers > 1 else 0
            self._contextual = rnn_type(input_dim,
                                        model_dim // 2,
                                        num_layers,
                                        bidirectional = True,
                                        batch_first = True,
                                        dropout = rnn_drop_out)
            state_none_num_sum_weight = use_state['usage']
            use_cell_as_state = use_state['from_cell']
            if state_none_num_sum_weight is None:
                self._state_config = None
            else:
                if use_cell_as_state:
                    assert rnn_type is nn.LSTM, 'GRU does not have a cell'
                if state_none_num_sum_weight == 'weight_layers':
                    self._layer_weights = nn.Parameter(torch.zeros(num_layers, 2, 1, 1))
                    self._layer_softmax = nn.Softmax(dim = 0)
                    self._state_to_top3 = nn.Linear(model_dim, 3 * hidden_dim)
                self._state_config = num_layers, use_cell_as_state, state_none_num_sum_weight, hidden_dim
        else:
            self._contextual = None

    @property
    def is_useless(self):
        return self._contextual is None

    def forward(self, static_emb):
        dynamic_emb, final_state = self._contextual(static_emb)

        if self._state_config is None:
            top_3 = None
        else:
            num_layers, use_cell_as_state, state_none_num_sum_weight, hidden_dim = self._state_config
            if isinstance(final_state, tuple):
                final_state = final_state[use_cell_as_state]
            batch_size, _, model_dim = dynamic_emb.shape
            final_state = final_state.view(num_layers, 2, batch_size, model_dim // 2)
            if isinstance(state_none_num_sum_weight, int): # some spec layer
                final_state = final_state[state_none_num_sum_weight]
            else: # sum dim = 0
                if state_none_num_sum_weight == 'weight_layers':
                    layer_weights = self._layer_softmax(self._layer_weights)
                    final_state = final_state * layer_weights
                final_state = final_state.sum(dim = 0)
            # final_state: [batch, model_dim]
            final_state = final_state.transpose(0, 1).reshape(batch_size, model_dim)
            if use_cell_as_state:
                final_state = torch.tanh(final_state)
            top_3 = self._state_to_top3(final_state).reshape(batch_size, 3, hidden_dim)

        return dynamic_emb, top_3

char_rnn_config = dict(embed_dim    = hidden_dim,
                       drop_out     = frac_4,
                       rnn_drop_out = frac_2,
                       module       = rnn_module_type,
                       num_layers   = num_ori_layer,
                       trainable_initials = false_type)
from models.utils import math, init, birnn_fwbw, fencepost, Bias
class PadRNN(nn.Module):
    def __init__(self,
                 num_chars,
                 attention_hint, # dims
                 linear_dim, # 01+ fence_vote, activation
                 embed_dim,
                 fence_dim,
                 drop_out,
                 num_layers,
                 module, # num_layers, rnn_drop_out
                 rnn_drop_out,
                 trainable_initials,
                 fence_vote = None,
                 activation = None,
                 char_space_idx = None):
        super().__init__()
        single_size = fence_dim // 2
        if num_layers:
            self._fence_emb = module(embed_dim, single_size,
                                     num_layers    = num_layers,
                                     bidirectional = True,
                                     batch_first   = True,
                                     dropout = rnn_drop_out if num_layers > 1 else 0)
        else:
            self._fence_emb = None
        self._tanh = nn.Tanh()
        bound = 1 / math.sqrt(single_size)
        if trainable_initials:
            c0 = torch.empty(num_layers * 2, 1, single_size)
            h0 = torch.empty(num_layers * 2, 1, single_size)
            self._c0 = nn.Parameter(c0, requires_grad = True)
            self._h0 = nn.Parameter(h0, requires_grad = True)
            init.uniform_(self._c0, -bound, bound)
            init.uniform_(self._h0, -bound, bound)
            self._initial_size = single_size
        else:
            self.register_parameter('_h0', None)
            self.register_parameter('_c0', None)
            self._initial_size = None

        if char_space_idx is None:
            self._pad = nn.Parameter(torch.empty(1, 1, single_size), requires_grad = True)
            init.uniform_(self._pad, -bound, bound)
        else:
            self._pad = char_space_idx
        self._stem_dp = nn.Dropout(drop_out)

        if num_chars: # forward is open
            self._char_emb = nn.Embedding(num_chars, embed_dim, padding_idx = 0)

        if attention_hint: # domain_and_subject is open
            if not isinstance(attention_hint, HParams): attention_hint = HParams(attention_hint)
            self._domain = nn.Linear(fence_dim, embed_dim, bias = False) if attention_hint.get('boundary') else None
            self._subject_unit  = nn.Linear(embed_dim, embed_dim, bias = False) if attention_hint.unit else None
            self._subject_state = nn.Linear(fence_dim, embed_dim, bias = False) if attention_hint.state else None
            single_size = fence_dim // 2
            if attention_hint.before:
                self._subject_fw_b = nn.Linear(single_size, embed_dim, bias = False)
                self._subject_bw_b = nn.Linear(single_size, embed_dim, bias = False)
            else:
                self._subject_fw_b = None
                self._subject_bw_b = None
            if attention_hint.after:
                self._subject_fw_a = nn.Linear(single_size, embed_dim, bias = False)
                self._subject_bw_a = nn.Linear(single_size, embed_dim, bias = False)
            else:
                self._subject_fw_a = None
                self._subject_bw_a = None
            if attention_hint.difference:
                self._subject_fw_d = nn.Linear(single_size, embed_dim, bias = False)
                self._subject_bw_d = nn.Linear(single_size, embed_dim, bias = False)
            else:
                self._subject_fw_d = None
                self._subject_bw_d = None
            self._subject_bias = Bias(embed_dim)

        if linear_dim:
            if fence_vote is None:
                self._fence_vote = None
                self._fence_l1 = nn.Linear(fence_dim, linear_dim)
                if linear_dim == 1:
                    self._fence_l2 = self._fence_act = lambda x: x
                else:
                    self._fence_act = activation()
                    self._fence_l2 = nn.Linear(linear_dim, 1)
            else:
                self._fence_act = activation()
                from_unit, method = fence_vote.split('.')
                from_unit = from_unit == 'unit'
                if method == 'dot':
                    self._fence_l1 = nn.Linear(fence_dim, linear_dim)
                    if from_unit:
                        self._fence_l2 = nn.Linear(model_dim, linear_dim)
                    else:
                        self._fence_l2 = nn.Linear(fence_dim, linear_dim)
                    method = self.predict_fence_2d_dot
                elif method == 'cat':
                    if from_unit:
                        self._fence_l1 = nn.Linear(fence_dim + model_dim, linear_dim)
                    else:
                        self._fence_l1 = nn.Linear(fence_dim << 1, linear_dim)
                    self._fence_l2 = nn.Linear(linear_dim, 1)
                    method = self.predict_fence_2d_cat
                else:
                    raise ValueError('Unknown method: ' + method)
                self._fence_vote = from_unit, method
        # fence_p: f->hidden [b, s+1, h]
        # fence_c: u->hidden [b, s, h]
        # pxc: v->vote [b, s+1, s]
        # fence: s->score [b, s+1] .sum() > 0

    def get_h0c0(self, batch_size):
        if self._initial_size:
            c0 = self._c0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._h0.expand(2, batch_size, self._initial_size).contiguous()
            h0 = self._tanh(h0)
            h0c0 = h0, c0
        else:
            h0c0 = None
        return h0c0

    def pad_fwbw_hidden(self, fence_hidden, existence):
        pad = self._stem_dp(self._pad)
        pad = self._tanh(pad)
        return birnn_fwbw(fence_hidden, pad, existence)

    def domain_and_subject(self, fw, bw, fence_idx, unit_emb, fence_hidden):
        if self._domain:
            dom_emb = self._domain(fencepost(fw, bw, fence_idx))
            dom_emb = self._stem_dp(dom_emb)
        else:
            dom_emb = None
        sub_emb = self._stem_dp(self._subject_bias())
        if self._subject_unit:  sub_emb = sub_emb + self._stem_dp(self._subject_unit(unit_emb))
        if self._subject_state: sub_emb = sub_emb + self._stem_dp(self._subject_state(fence_hidden))
        if self._subject_fw_a:  sub_emb = sub_emb + self._stem_dp(self._subject_fw_a(fw[:, 1:]))
        if self._subject_bw_a:  sub_emb = sub_emb + self._stem_dp(self._subject_bw_a(bw[:, :-1]))
        if self._subject_fw_b:  sub_emb = sub_emb + self._stem_dp(self._subject_fw_b(fw[:, :-1]))
        if self._subject_bw_b:  sub_emb = sub_emb + self._stem_dp(self._subject_bw_b(bw[:, 1:]))
        if self._subject_fw_d:  sub_emb = sub_emb + self._stem_dp(self._subject_fw_d(fw[:, 1:] - fw[:, :-1]))
        if self._subject_bw_d:  sub_emb = sub_emb + self._stem_dp(self._subject_bw_d(bw[:, :-1] - bw[:, 1:]))
        return dom_emb, sub_emb

    def forward(self, char_idx, fence = None, offset = None): # concat fence vectors
        batch_size, char_len = char_idx.shape
        char_emb = self._char_emb(char_idx)
        char_emb = self._stem_dp(char_emb)
        fence_hidden, _ = self._fence_emb(char_emb, self.get_h0c0(batch_size))
        if fence is None:
            helper = condense_helper(char_idx == self._pad, True, offset)
            fence_hidden = fence_hidden.view(batch_size, char_len, 2, -1)
            fw = fence_hidden[:, :, 0]
            bw = fence_hidden[:, :, 1]
        else:    
            existence = char_idx > 0
            fw, bw = birnn_fwbw(fence_hidden, self._tanh(self._pad), existence)
            helper = condense_helper(fence, True, offset)
        fw = condense_left(fw, helper)
        bw = condense_left(bw, helper)
        return torch.cat([fw[:, 1:] - fw[:, :-1], bw[:, :-1] - bw[:, 1:]], dim = 2)
        # select & concat: fw[:*-1] - fw[*1:] & bw...

    def predict_fence(self, fw, bw):
        fence = torch.cat([fw, bw], dim = 2)
        fence = self._fence_l1(fence)
        fence = self._stem_dp(fence)
        fence = self._fence_act(fence)
        return self._fence_l2(fence).squeeze(dim = 2)

    def predict_fence_2d_dot(self, fw, bw, hidden, seq_len): # TODO not act for unit
        fence = torch.cat([fw, bw], dim = 2)
        fence = self._fence_l1(fence)
        fence = self._fence_act(self._stem_dp(fence))
        unit = self._fence_l2(hidden)
        unit = self._fence_act(self._stem_dp(unit))
        vote = torch.bmm(fence, unit.transpose(1, 2)) # [b, s+1, s]
        third_dim = torch.arange(unit.shape[1], device = hidden.device)
        third_dim = third_dim[None, None] < seq_len[:, None, None]
        third_dim = torch.where(third_dim, vote, torch.zeros_like(vote))
        return third_dim, third_dim.sum(dim = 2)
    
    def predict_fence_2d_cat(self, fw, bw, hidden, seq_len):
        fence = torch.cat([fw, bw], dim = 2)
        _, fence_len, fence_dim = fence.shape
        batch_size, seg_len, hidden_dim = hidden.shape
        fence = fence[:, :, None].expand(batch_size, fence_len, seg_len, fence_dim)
        hidden = hidden[:, None].expand(batch_size, fence_len, seg_len, hidden_dim)
        vote = torch.cat([fence, hidden], dim = 3) # [b, s+1, s, e]
        vote = self._fence_l1(vote)
        vote = self._stem_dp(vote)
        vote = self._fence_act(vote)
        vote = self._fence_l2(vote).squeeze(dim = 3)
        third_dim = torch.arange(seg_len, device = hidden.device)
        third_dim = third_dim[None, None] < seq_len[:, None, None]
        third_dim = torch.where(third_dim, vote, torch.zeros_like(vote))
        return third_dim, third_dim.sum(dim = 2)