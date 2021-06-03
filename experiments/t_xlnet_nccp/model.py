from models.nccp import BaseRnnTree, model_type, torch, nn, Tensor
from utils.types import word_dim, num_ctx_layer_0, frac_0, true_type
from models.plm import subword_proc, XLNetLeaves

from models.types import rnn_module_type, activation_type
xlnt_leaves_config = dict(contextual   = rnn_module_type,
                          num_layers   = num_ctx_layer_0,
                          drop_out     = frac_0,
                          rnn_drop_out = frac_0,
                          activation   = activation_type,
                          subword_proc = subword_proc,
                          sum_weighted_layers = true_type)
model_type = model_type.copy()
model_type['input_layer'] = xlnt_leaves_config
model_type['model_dim']   = word_dim

class ContinuousXLNetTree(BaseRnnTree):
    def __init__(self,
                 model_dim,
                 paddings,
                 input_layer,
                 **base_config):
        super().__init__(model_dim, **base_config)
        self._input_layer = XLNetLeaves(model_dim, paddings = paddings, **input_layer)

    def forward(self,
                word_idx,
                tune_pre_trained,
                offset, plm_idx, plm_start, 
                ingore_logits = False, **kw_args):
        batch_size, batch_len, base_inputs, bottom_existence = self._input_layer(word_idx, offset, plm_idx, plm_start, tune_pre_trained)
        base_returns = super().forward(base_inputs, bottom_existence, ingore_logits, **kw_args)
        return (batch_size, batch_len, base_inputs, None) + base_returns

    @property
    def message(self):
        if self._input_layer._layer_weights is not None:
            _layer_weights = self._input_layer._layer_weights.detach().reshape(-1)
            _layer_weights = self._input_layer._softmax(_layer_weights).cpu()
            min_val = _layer_weights.min()
            max_val = _layer_weights.max()
            _layer_weights = (_layer_weights - min_val) / (max_val - min_val)
            from utils.str_ops import height_ratio, str_ruler
            msg_0 = 'Layer weights: '
            msg_1 = ''.join(height_ratio(w) for w in _layer_weights.numpy())
            msg_2 = f'[{min_val:.2f}, {max_val:.2f}]'
            msg_3 = '\n               '
            msg_4 = str_ruler(len(msg_1))
            return msg_0 + msg_1 + msg_2 + msg_3 + msg_4