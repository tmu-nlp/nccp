from models.backend import torch, InputLeaves, Contextual, input_config, contextual_config
from models.backend import PadRNN, char_rnn_config, nn
from models.accp import BaseRnnTree, model_type
from utils.types import word_dim, true_type, false_type
from models.combine import get_combinator, combine_static_type

model_type = model_type.copy()
model_type['model_dim']        = word_dim
model_type['char_rnn']         = char_rnn_config
model_type['word_emb']         = input_config
model_type['use']              = dict(char_rnn = false_type, word_emb = true_type)
model_type['contextual_layer'] = contextual_config
model_type['combine_static']   = combine_static_type

class MultiRnnTree(BaseRnnTree):
    def __init__(self,
                 paddings,
                 model_dim,
                 use,
                 word_emb,
                 char_rnn,
                 contextual_layer,
                 combine_static,
                 num_chars       = None,
                 num_tokens      = None,
                 initial_weights = None,
                 **base_config):
        super().__init__(model_dim, **base_config)
        
        if use['word_emb']:
            self._word_emb = InputLeaves(model_dim, num_tokens, initial_weights, not paddings, **word_emb)
            input_dim = self._word_emb.input_dim
        else:
            self._word_emb = None
            input_dim = model_dim
        if use['char_rnn']:
            self._char_rnn = PadRNN(num_chars, None, None, fence_dim = model_dim, char_space_idx = 1, **char_rnn)
        else:
            self._char_rnn = None

        contextual_layer = Contextual(input_dim, model_dim, self.hidden_dim, **contextual_layer)
        diff = model_dim - input_dim
        self._combine_static = None
        self._bias_only = False
        if contextual_layer.is_useless:
            self._contextual_layer = None
            assert diff == 0, 'useless difference'
        else:
            self._contextual_layer = contextual_layer
            if combine_static:
                self._bias_only = combine_static in ('NS', 'NV')
                self._combine_static = get_combinator(combine_static, input_dim)
            assert diff >= 0, 'invalid difference'
        self._half_dim_diff = diff >> 1

    def get_static_pca(self):
        if self._word_emb and self._word_emb.has_static_pca:
            return self._word_emb.pca
        return None

    def update_static_pca(self):
        if self._word_emb and self._word_emb.has_static_pca:
            self._word_emb.flush_pc_if_emb_is_tuned()

    def forward(self, word_idx, tune_pre_trained,
                sub_idx = None, sub_fence = None, offset = None, **kw_args):
        batch_size,   batch_len  = word_idx.shape
        if self._word_emb:
            static, bottom_existence = self._word_emb(word_idx, tune_pre_trained)
            if self._char_rnn:
                static = static + self._char_rnn(sub_idx, sub_fence, offset) * bottom_existence
        else:
            bottom_existence = word_idx > 0
            bottom_existence.unsqueeze_(dim = 2)
            static = self._char_rnn(sub_idx, sub_fence, offset) * bottom_existence
        if self._contextual_layer is None:
            base_inputs = static
            top3_hidden = None
        else:
            dynamic, top3_hidden = self._contextual_layer(static)
            if self._half_dim_diff:
                zero_pads = torch.zeros(batch_size, batch_len, self._half_dim_diff, dtype = static.dtype, device = static.device)
                static = torch.cat([zero_pads, static, zero_pads], dim = 2)
            base_inputs  = dynamic * bottom_existence
            if self._combine_static is not None:
                base_inputs = self._combine_static.compose(static, base_inputs, None)
        base_returns = super().forward(base_inputs, bottom_existence.squeeze(dim = 2), **kw_args)
        top3_labels  = super().get_label(top3_hidden) if top3_hidden is not None else None
        return (batch_size, batch_len, static, top3_labels) + base_returns

    def tensorboard(self, recorder, global_step):
        if self._bias_only:
            ctx_ratio = self._combine_static.itp_rhs_bias().detach()
            if ctx_ratio is not None:
                params = dict(ContextualRatio = ctx_ratio.mean())
                if ctx_ratio.nelement() > 1:
                    params['RatioStdv'] = ctx_ratio.std()
                recorder.tensorboard(global_step, 'Parameters/%s', **params)

    @property
    def message(self):
        if self._bias_only:
            ctx_ratio = self._combine_static.itp_rhs_bias().detach()
            if ctx_ratio is not None:
                ctx_ratio *= 100
                msg = 'Contextual Rate:'
                msg += f' {ctx_ratio.mean():.2f}'
                if ctx_ratio.nelement() > 1:
                    msg += f'±{ctx_ratio.std():.2f}%'
                else:
                    msg += '%'
                return msg