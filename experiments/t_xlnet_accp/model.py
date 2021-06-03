from models.accp import BaseRnnTree, model_type
from utils.types import word_dim, num_ctx_layer_0, frac_0
from models.plm import subword_proc, XLNetLeaves

from models.types import rnn_module_type, activation_type
xlnt_leaves_config = dict(contextual   = rnn_module_type,
                          num_layers   = num_ctx_layer_0,
                          drop_out     = frac_0,
                          rnn_drop_out = frac_0,
                          activation   = activation_type,
                          subword_proc = subword_proc)
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
                plm_idx, plm_start, 
                ingore_logits = False, **kw_args):
        batch_size, batch_len, base_inputs, bottom_existence = self._input_layer(word_idx, 0, plm_idx, plm_start, tune_pre_trained)
        base_returns = super().forward(base_inputs, bottom_existence.squeeze(2), ingore_logits, **kw_args)
        return (batch_size, batch_len, base_inputs, None) + base_returns