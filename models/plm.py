from utils.types import BaseType
E_SUB = S_LFT, S_RGT, S_AVG, S_SGT = 'leftmost rightmost average selfgate'.split()
subword_proc = BaseType(0, as_index = True, default_set = E_SUB)

from models.utils import condense_helper, condense_left
from models.backend import torch, nn, init, math
class PreLeaves(nn.Module):
    has_static_pca = False # class property

    def __init__(self,
                 model_key_name,
                 model_dim,
                 contextual,
                 num_layers,
                 drop_out,
                 rnn_drop_out,
                 activation,
                 paddings,
                 subword_proc,
                 sum_weighted_layers):
        super().__init__() #  

        from transformers import AutoModel
        self._pre_model = AutoModel.from_pretrained(model_key_name)
        self._dp_layer = nn.Dropout(drop_out)
        self._activation = activation()
        
        if num_layers == 0:
            self._is_linear = True
            self._to_word_emb = nn.Linear(768, model_dim + (subword_proc == S_SGT))
        else:
            self._is_linear = False
            self._to_word_emb = contextual(768, model_dim // 2 + (subword_proc == S_SGT), 
                                           num_layers,
                                           batch_first = True,
                                           bidirectional = True,
                                           dropout = rnn_drop_out if num_layers > 1 else 0)
            if subword_proc == S_SGT:
                self._rnn_word_gate = nn.Linear(2, 1)
        if subword_proc == S_SGT:
            self._gate_act = nn.Sigmoid()

        if paddings:
            bos = torch.empty(1, 1, model_dim)
            eos = torch.empty(1, 1, model_dim)
            self._bos = nn.Parameter(bos, requires_grad = True)
            self._eos = nn.Parameter(eos, requires_grad = True)
            bound = 1 / math.sqrt(model_dim)
            init.uniform_(self._bos, -bound, bound)
            init.uniform_(self._eos, -bound, bound)

        if sum_weighted_layers:
            n_layer = self._pre_model.n_layer
            layer_weights = torch.empty(n_layer + 1, 1, 1, 1) # TODO: better to use layers[n:]?
            self._layer_weights = nn.Parameter(layer_weights, requires_grad = True)
            self._softmax = nn.Softmax(dim = 0)
            bound = 1 / math.sqrt(n_layer)
            init.uniform_(self._layer_weights, -bound, bound)
            self._pre_model.config.__dict__['output_hidden_states'] = True
        else:
            self._layer_weights = None

        self._paddings = paddings
        self._word_dim = model_dim
        self._subword_proc = subword_proc
        # self._pre_model.train()

    @property
    def embedding_dim(self):
        return self._word_dim

    def forward(self, word_idx, offset, plm_idx, plm_start, tune_pre_trained = False):
        batch_size, batch_len = word_idx.shape # just provide shape and nil info
        if isinstance(offset, int):
            temp = torch.ones(batch_size, dtype = word_idx.dtype, device = word_idx.device)
            temp *= offset
            offset = temp

        if tune_pre_trained:
            plm_outputs = self._pre_model(plm_idx, output_hidden_states = True)
            # xl_hidden = xl_hidden[:, :-2] # Bad idea: git rid of some [cls][sep]
        else:
            with torch.no_grad():
                plm_outputs = self._pre_model(plm_idx, output_hidden_states = True)
        
        if self._layer_weights is None:
            xl_hidden = plm_outputs.last_hidden_state
        else:
            layer_weights = self._softmax(self._layer_weights)
            xl_hidden = torch.stack(plm_outputs.hidden_states)
            xl_hidden = (xl_hidden * layer_weights).sum(dim = 0)

        xl_hidden = self._dp_layer(xl_hidden)

        def transform_dim(xl_hidden):
            word_hidden = self._to_word_emb(xl_hidden)
            if self._is_linear:
                if self._subword_proc == S_SGT:
                    word_gate = self._gate_act(word_hidden[:, :, 0, None])
                    word_hidden = self._activation(word_hidden[:, :, 1:])
                    word_hidden = word_gate * word_hidden
                else:
                    word_hidden = self._activation(word_hidden)
            else:
                word_hidden = word_hidden[0]
                if self._subword_proc == S_SGT:
                    lw_gate = word_hidden[:, :,  0, None]
                    rw_gate = word_hidden[:, :, -1, None]
                    word_gate = torch.cat([lw_gate, rw_gate], dim = 2)
                    word_gate = self._rnn_word_gate(word_gate)
                    word_gate = self._gate_act(word_gate)
                    word_hidden = word_gate * word_hidden[:, :, 1:-1]
            return word_hidden

        if self._subword_proc in (S_LFT, S_RGT):
            if self._subword_proc == S_LFT: # i dis #agree with it
                xl_pointer = plm_start      # 1 1   0      1    1
            else:                           # 1 0   1      1    1
                xl_pointer = torch.cat([plm_start[:, 1:], torch.ones_like(plm_start[:, :1])], dim = 1)
            helper = condense_helper(xl_pointer, as_existence = True, offset = offset, get_rid_of_last_k = 1)
            xl_hidden = condense_left(xl_hidden, helper, out_len = batch_len)
            xl_base = transform_dim(xl_hidden) # use left most sub-word to save precious time!
        else:
            word_hidden = transform_dim(xl_hidden)
            helper = condense_helper(plm_start, as_existence = False, offset = offset, get_rid_of_last_k = 1)
            if self._subword_proc == S_AVG:
                xl_base, xl_cumu = condense_left(word_hidden, helper, out_len = batch_len, get_cumu = True)
                xl_cumu[xl_cumu < 1] = 1 # prevent 0
                xl_base = xl_base / xl_cumu
            else:
                xl_base = condense_left(word_hidden, helper, out_len = batch_len)
        
        if self._paddings: # will overwrite [cls][sep]
            bos, eos = self._paddings['word']
            bos = (word_idx == bos)
            eos = (word_idx == eos)
            bos.unsqueeze_(dim = 2)
            eos.unsqueeze_(dim = 2)
            xl_base = torch.where(bos, self._bos.expand_as(xl_base), xl_base) # 不要让nn做太多离散的决定，人来做！
            xl_base = torch.where(eos, self._eos.expand_as(xl_base), xl_base)
            non_nil = torch.ones(batch_size, batch_len, 1, dtype = torch.bool, device = word_idx.device)
        else:
            non_nil = (word_idx > 0)
            non_nil.unsqueeze_(dim = 2)
            xl_base = xl_base * non_nil # in-place fails at FloatTensor
        return batch_size, batch_len, xl_base, non_nil # just dynamic

xlnet_model_key = 'xlnet-base-cased'
gbert_model_key = 'bert-base-german-cased'
class XLNetLeaves(PreLeaves):
    def __init__(self, *args, **kw_args):
        super().__init__(xlnet_model_key, *args, **kw_args)

class GBertLeaves(PreLeaves):
    def __init__(self, *args, **kw_args):
        super().__init__(gbert_model_key, *args, **kw_args)


_penn_to_xlnet = {'``': '"', "''": '"'}
from tqdm import tqdm
from unidecode import unidecode
from multiprocessing import Pool
from data.backend import TextHelper
class PreDatasetHelper(TextHelper):
    def __init__(self, text, device, *args):
        with Pool() as p:
            cache = p.map(self._append, text)
        super().__init__(cache, device)
        # self._cache = cache =  []
        # for penn_words in tqdm(text, desc = self.tknz_name):
        #     cache.append(self._append(penn_words))

    @classmethod
    def _append(cls, penn_words, check = True):
        text = cls._adapt_text_for_tokenizer(penn_words)
        xlnt_words  = cls.tokenizer.tokenize(text)
        word_idx    = cls.tokenizer.encode(text)
        # import pdb; pdb.set_trace()
        xlnt_starts = cls._start(penn_words, xlnt_words)
        if check:
            assert len(xlnt_words) == len(xlnt_starts), text + f" {' '.join(xlnt_words)}"
            if len(word_idx) - 2 != len(xlnt_words):
                import pdb; pdb.set_trace()
            if len(penn_words) != sum(xlnt_starts):
                import pdb; pdb.set_trace()
        return word_idx, xlnt_starts
        
    def get(self):
        plm_idx, plm_start = [], []
        start, end, pad_token_id = self.start_end
        for wi, ws, len_diff in self.gen_from_buffer():
            plm_idx  .append(wi + [pad_token_id] * len_diff)
            plm_start.append(start + ws + [True] + [False] * (len_diff + end)) # TODO check!
        plm_idx   = torch.tensor(plm_idx,   device = self._device)
        plm_start = torch.tensor(plm_start, device = self._device)
        return dict(plm_idx = plm_idx, plm_start = plm_start)

    @staticmethod
    def _adapt_text_for_tokenizer(penn_words):
        raise NotImplementedError('PreDatasetHelper._adapt_text_for_tokenizer')

    @staticmethod
    def _start(penn_words, xlnt_words):
        raise NotImplementedError('PreDatasetHelper._start')
    

class XLNetDatasetHelper(PreDatasetHelper):
    tokenizer = None
    tknz_name = 'XLNetTokenizer'

    @classmethod
    def __new__(cls, *args, **kwargs):
        # sent <sep> <cls> <pad>
        # 1234   0     0     0   # 0 was truncated
        if cls.tokenizer is None:
            from transformers import AutoTokenizer
            cls.tokenizer = t = AutoTokenizer.from_pretrained(xlnet_model_key)
            cls.start_end = [], 1, t.pad_token_id
        return object.__new__(cls)

    # @classmethod
    # def for_discontinuous(cls, *args, **kwargs):
    #     # sent <sep> <cls> <pad>
    #     # 2345   0     0     0   # 0 was truncated, 1 is <0>
    #     if cls.tokenizer is None:
    #         from transformers import XLNetTokenizer #, SPIECE_UNDERLINE
    #         cls.tokenizer = t = AutoTokenizer.from_pretrained(xlnet_model_key)
    #         cls.start_end = [False], 0, t.pad_token_id
    #     return cls(*args, **kwargs)

    @staticmethod
    def _adapt_text_for_tokenizer(penn_words):
        text = None
        for pw in penn_words:
            pw = _penn_to_xlnet.get(pw, pw)
            if text is None:
                text = pw
            elif pw in '.,();':
                text += pw
            else:
                text += ' ' + pw
        return text

    @staticmethod
    def _start(penn_words, xlnt_words):
        xlnt_starts = []
        xlnt_offset = 0
        for i_, pw in enumerate(penn_words):
            xlnt_starts.append(True)
            xlnt_word = xlnt_words[i_ + xlnt_offset]
            if xlnt_word[0] == '▁':
                xlnt_word = xlnt_word[1:]
            if pw == xlnt_word:
                continue
            while xlnt_word != pw:
                xlnt_offset += 1
                try:
                    piece = xlnt_words[i_ + xlnt_offset]
                except:
                    import pdb; pdb.set_trace()
                if piece == '"': # -`` in ptb
                    piece = '``' if '``' in pw else "''"
                xlnt_word += piece
                xlnt_starts.append(False)
        return xlnt_starts


class GBertDatasetHelper(PreDatasetHelper):
    # <cls> sent <sep> <pad>
    #   0   2345   0     0
    tokenizer = None
    tknz_name = 'GermanBertTokenizer'

    @classmethod
    def __new__(cls, *args, **kwargs):
        if cls.tokenizer is None:
            from transformers import AutoTokenizer
            cls.tokenizer = t = AutoTokenizer.from_pretrained(gbert_model_key)
            cls.start_end = [False], 0, t.pad_token_id
        return object.__new__(cls)

    @classmethod
    def _adapt_text_for_tokenizer(cls, penn_words):
        text = None
        unk_token_id = cls.tokenizer.unk_token_id
        encode_func  = cls.tokenizer.encode
        for wid, pw in enumerate(penn_words):
            pw = _penn_to_xlnet.get(pw, pw)
            if unk_token_id in encode_func(pw)[1:-1]:
                # _pw = pw
                pw = unidecode(pw)
                penn_words[wid] = pw
                # print(_pw, pw)
            if text is None:
                text = pw
            elif pw in '.,();':
                text += pw
            else:
                text += ' ' + pw
        return text

    @classmethod
    def _start(cls, penn_words, xlnt_words):
        xlnt_starts = []
        xlnt_offset = 0
        unk_token = cls.tokenizer.unk_token
        for i_, pw in enumerate(penn_words):
            xlnt_starts.append(True)
            xlnt_word = xlnt_words[i_ + xlnt_offset]
            if pw == xlnt_word or xlnt_word == '"' and pw in ('``', "''"):
                continue
            elif xlnt_word == unk_token:
                # print('Unknown word:', pw)
                continue
            while xlnt_word != pw:
                xlnt_offset += 1
                try:
                    piece = xlnt_words[i_ + xlnt_offset]
                except:
                    import pdb; pdb.set_trace()
                if piece.startswith('##'):
                    piece = piece[2:]
                xlnt_word += piece
                xlnt_starts.append(False)
        return xlnt_starts