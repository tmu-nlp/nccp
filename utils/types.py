from copy import copy, deepcopy

class BaseWrapper:
    def __init__(self, item, to_str):
        self._item_to_str = item, to_str

    @property
    def name(self):
        item, to_str = self._item_to_str
        return to_str(item)

    @property
    def item(self):
        return self._item_to_str[0]

    def identical(self, x):
        return x == self.name

    @classmethod
    def from_gen(cls, gen, to_name):
        return tuple(cls(x, to_name) for x in gen)


class BaseType:
    def __init__(self, default_val, validator = None, default_set = None, as_exception = False, as_index = False):
        self._val_as_index = default_val, as_index, as_exception
        self._set = None if default_set is None else tuple(default_set)
        self._fallback = None
        if validator is None:
            if default_set is None: # uncountable value, float
                assert not as_index
                if as_exception:
                    cls = type(default_val)
                    self._valid = lambda x: isinstance(x, cls)
                else: # constant
                    self._valid = lambda x: x == default_val
            else:
                assert as_index
                assert isinstance(default_val, int) or default_val is None
                if as_exception: # (nn.LSTM, nn.GRU)
                    names_set = tuple(x.name for x in default_set)
                    if default_val is None:
                        names_set = (None,) + names_set
                    self._valid = lambda x: x in names_set
                else: # ('CV', 'NV')
                    self._valid = lambda x: x in default_set
        else:
            if default_set is None: # uncountable value | default_val can be an exception
                assert not as_index
                if as_exception: # [0, 1) or None
                    self._valid = lambda x: validator(x) or x == default_val
                else: # [0, 1] at 0.2
                    assert validator(default_val)
                    self._valid = validator
            elif as_index:
                assert not as_exception
                self._valid = lambda x: (x in default_set or validator(x))
            else:
                if as_exception: # []
                    self._valid = lambda x: (x in default_set or validator(x) or x == default_val)
                else:
                    self._valid = lambda x: (x in default_set or validator(x))
                    assert self._valid(default_val)

    @property
    def default(self):
        default_val, as_index, as_exception = self._val_as_index
        if as_index and default_val is not None:
            return (self._set[default_val].name if as_exception else self._set[default_val])
        return default_val

    def validate(self, val):
        # valid = self._valid(val)
        # if not valid and self._fallback is not None:
        #     return self._fallback.validate(val)
        # return valid
        return self._valid(val)

    # def set_fallback(self, btype):
    #     assert isinstance(btype, BaseType)
    #     self._fallback = btype

    def __getitem__(self, idx):
        default_val, as_index, as_exception = self._val_as_index
        if as_index and as_exception:
            for x in self._set:
                if x.identical(idx):
                    return x.item
            return default_val
        # elif self._fallback is not None:
        #     idx = self._fallback[idx]
        return idx

E_FT = (False, True)
E_LY = (1, 2, 4, 8)
E_MS = (2, 32, 64, 128)

frac_close     = lambda x: 0 <= x <= 1
frac_open_1    = lambda x: 0 <= x < 1
frac_open_0    = lambda x: 0 < x <= 1
valid_size      = lambda x: isinstance(x, int) and x > 0
valid_odd_size  = lambda x: valid_size(x) and x % 2 == 1
valid_even_size = lambda x: valid_size(x) and x % 2 == 0
valid_epoch     = lambda x: isinstance(x, int) and x >= 0
false_type     = BaseType(False, as_exception = True)
true_type      = BaseType(True,  as_exception = True)
frac_0         = BaseType(0.0, validator = frac_open_1)
frac_1         = BaseType(0.1, validator = frac_open_1)
frac_2         = BaseType(0.2, validator = frac_open_1)
frac_3         = BaseType(0.3, validator = frac_open_1)
frac_4         = BaseType(0.4, validator = frac_open_1)
frac_5         = BaseType(0.5, validator = frac_open_1)
frac_7         = BaseType(0.7, validator = frac_open_1)
frac_06        = BaseType(0.06, validator = frac_open_1)
rate_5         = BaseType(0.5, validator = frac_close)
distance_type  = BaseType(3.1, validator = lambda d: d > 0)
non0_5         = BaseType(0.5, validator = frac_open_0)
none_type      = BaseType(None)
num_ctx_layer = BaseType(6, validator = lambda x: isinstance(x, int) and 0 <= x <= 24)
num_ori_layer = BaseType(1, validator = lambda x: isinstance(x, int) and 1 <= x <= 4)
num_ctx_layer_0 = BaseType(0, validator = lambda x: isinstance(x, int) and 0 <= x <= 24)
vocab_size = BaseType(None, validator = lambda x: isinstance(x, int) and 2 < x, as_exception = True)
word_dim   = BaseType(300, validator = valid_even_size)
orient_dim = BaseType(64,  validator = valid_even_size)
chunk_dim  = BaseType(200,  validator = valid_even_size)
hidden_dim = BaseType(200, validator = valid_size)
half_hidden_dim = BaseType(100, validator = valid_size)
train_batch_size = BaseType(80, validator = valid_size)
train_bucket_len = BaseType(4, validator = valid_epoch)
tune_epoch_type  = BaseType(None, as_exception = True, validator = valid_epoch)
train_max_len    = BaseType(None, validator = valid_size, as_exception = True)
trapezoid_height = BaseType(-1, valid_size, (None, 0, 1), as_index = True)
combine_static = BaseType(0, as_index = True, default_set = (None, 'add', 'scalar_add', 'vector_add'))
fill_placeholder = '//FILL//THIS//'

from utils.str_ops import strange_to
def strange_validator(x):
    if x is None or x == '':
        return True
    try:
        strange_to(x)
    except:
        return False
    return True

str_num_array = BaseType('', validator = strange_validator)

NIL, PAD, UNK, BOS, EOS = '<nil>', '<pad>', '<unk>', '<bos>', '<eos>'
M_TRAIN = 'train'
M_DEVEL = 'devel'
M_TEST  = 'test'
M_INFER = 'infer'
E_MODE = (M_TRAIN, M_DEVEL, M_TEST, M_INFER)
E_ORIF4 = 'left', 'right', 'midin', 'midout'
O_LFT, O_RGT, O_MIN, O_MOT = E_ORIF4
O_HEAD = 'head'
O_M25, O_M50, O_M75 = (O_MIN + x for x in ('25', '50', '75'))
E_ORIF5 = O_LFT, O_M25, O_M50, O_M75, O_RGT
E_ORIF5_HEAD = O_LFT, O_M25, O_M50, O_M75, O_RGT, O_HEAD
E_CNF = O_LFT, O_RGT

import os
num_threads = (os.cpu_count() - 2) if os.cpu_count() > 2 else 1

frac_00 = BaseType(0.0, frac_close)
frac_85 = BaseType(0.85, frac_close)
frac_15 = BaseType(0.15, frac_close)
binarization = {O_LFT: frac_85,
                O_RGT: frac_15,
                O_MIN: frac_00,
                O_MOT: frac_00}
frac_close_0 = BaseType(0.0, frac_close)
frac_close_2 = BaseType(0.2, frac_close)
frac_close_1 = BaseType(1.0, frac_close)
binarization_5_head = {o: frac_close_0 if o == O_HEAD else frac_close_2 for o in E_ORIF5_HEAD}
S_ALL, S_EXH = 'all', 'except_head'
swapper = BaseType(0, default_set = (None, S_ALL, S_EXH), as_index = True)