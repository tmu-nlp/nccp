from torch import optim
from math import exp

class WarmOptimHelper:
    @classmethod
    def adam(cls, model, *args, **kwargs):
        opt = optim.Adam(model.parameters(), betas = (0.9, 0.98), weight_decay = 0.01, eps = 1e-6)
        return cls(opt, *args, **kwargs)

    def __init__(self, optimizer, base_lr = 0.001, wander_threshold = 0.15, damp_threshold = 0.4, damp = 1):
        assert base_lr > 0
        assert 0 < damp <= 1
        assert 0 < wander_threshold < damp_threshold <= 1
        self._opt_threshold_damp = optimizer, wander_threshold, damp_threshold, damp
        self._base_lr_last_wr = base_lr, 0
        
    def __call__(self, epoch, wander_ratio, lr_factor = 1):
        opt, wander_threshold, damp_threshold, damp = self._opt_threshold_damp
        base_lr, last_wr = self._base_lr_last_wr
        if wander_ratio < damp_threshold < last_wr:
            base_lr *= damp
        self._base_lr_last_wr = base_lr, wander_ratio

        if wander_ratio < wander_threshold:
            learning_rate = base_lr * (1 - exp(- epoch))
        else:
            linear_dec = (1 - (wander_ratio - wander_threshold) / (1 - wander_threshold + 1e-20))
            learning_rate = base_lr * linear_dec
        learning_rate *= lr_factor

        for opg in opt.param_groups:
            opg['lr'] = learning_rate
        # self._lr_discount_rate = 0.0001
        # for params in self._model.parameters():
        #     if len(params.shape) > 1:
        #         nn.init.xavier_uniform_(params)
        return learning_rate
            
    @property
    def optimizer(self):
        return self._opt_threshold_damp[0]