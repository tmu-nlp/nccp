import torch
from torch import nn, Tensor
from models.backend import Stem
from models.types import activation_type, logit_type

from utils.types import hidden_dim, frac_4

multi_class = dict(hidden_dim = hidden_dim,
                   activation = activation_type,
                   logit_type = logit_type,
                   drop_out   = frac_4)

from models.backend import stem_config
model_type = dict(orient_layer    = stem_config,
                  tag_label_layer = multi_class)
from models.utils import get_logit_layer
from models.loss import get_decision, get_decision_with_value, get_loss

class BaseRnnTree(nn.Module):
    def __init__(self,
                 model_dim,
                 num_tags,
                 num_labels,
                 orient_layer,
                 tag_label_layer,
                 **kw_args):
        super().__init__(**kw_args)

        self._stem_layer = Stem(model_dim, **orient_layer)

        hidden_dim = tag_label_layer['hidden_dim']
        if hidden_dim:
            self._shared_layer = nn.Linear(model_dim, hidden_dim)
            self._dp_layer = nn.Dropout(tag_label_layer['drop_out'])

            Net, argmax, score_act = get_logit_layer(tag_label_layer['logit_type'])
            self._tag_layer   = Net(hidden_dim, num_tags) if num_tags else None
            self._label_layer = Net(hidden_dim, num_labels) if num_labels else None
            self._logit_max = argmax
            if argmax:
                self._activation = tag_label_layer['activation']()
            self._score_fn = score_act(dim = 2)
        self._hidden_dim = hidden_dim
        self._model_dim = model_dim

    def forward(self,
                base_inputs,
                bottom_existence,
                ingore_logits = False,
                **kw_args):

        (layers_of_base, layers_of_orient, layers_of_existence,
         trapezoid_info) = self._stem_layer(bottom_existence,
                                            base_inputs, # dynamic can be none
                                            **kw_args)

        if self._hidden_dim:
            layers_of_hidden = self._shared_layer(layers_of_base)
            layers_of_hidden = self._dp_layer(layers_of_hidden)
            if self._logit_max:
                layers_of_hidden = self._activation(layers_of_hidden)

            if self._tag_layer is None or ingore_logits:
                tags = None
            else:
                _, batch_len, _ = base_inputs.shape
                tags = self._tag_layer(layers_of_hidden[:, -batch_len:])
            
            if self._label_layer is None or ingore_logits:
                labels = None
            else:
                labels = self._label_layer(layers_of_hidden)
        else:
            layers_of_hidden = tags = labels = None

        return layers_of_base, layers_of_hidden, layers_of_existence, layers_of_orient, tags, labels, trapezoid_info

    @property
    def model_dim(self):
        return self._model_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def get_label(self, hidden):
        return self._label_layer(hidden)

    def get_decision(self, logits):
        return get_decision(self._logit_max, logits)

    def get_decision_with_value(self, logits):
        return get_decision_with_value(self._score_fn, logits)

    def get_losses(self, batch, tag_logits, top3_label_logits, label_logits, height_mask, weight_mask):
        tag_loss   = get_loss(self._tag_layer,   self._logit_max, tag_logits,   batch, 'tag')
        label_loss = get_loss(self._label_layer, self._logit_max, label_logits, batch, True, height_mask, weight_mask, 'label')
        if top3_label_logits is not None:
            tag_loss += get_loss(self._label_layer, self._logit_max, top3_label_logits, batch, 'top3_label')
        return tag_loss, label_loss