from data.backend import LengthOrderedDataset, np, torch
from data.delta import s_index, DeltaX, xtype_to_logits, preproc_cnf
from data.penn_types import select_and_split_corpus, SourcePool
from tqdm import tqdm
from itertools import zip_longest, count
from multiprocessing import Process, Queue
from utils.file_io import DelayedKeyboardInterrupt
from time import sleep
# from data.delta import E_XDIM

fields = 'token', 'tag', 'ftag'
fieldx = 'label', 'xtype'
# FieldOrder = 'token', 'tag', 'label', 'xtype', 'ftag', 'length'

class PennTreeKeeper:
    def __init__(self, tree, v2is, trapezoid_height):
        self._tree = tree
        self._v2is = v2is
        self._w_p = None
        self._factored = {}
        self._trapezoid_height = trapezoid_height

    def update_factored(self, factored, words):
        self._factored.update(factored)
        tree = self._tree
        for i, word in enumerate(words):
            if word == '(':
                tree[tree.leaf_treeposition(i)] = '('
            elif word == ')':
                tree[tree.leaf_treeposition(i)] = ')'

    def __getitem__(self, factor):
        if factor in self._factored:
            return self._factored[factor]

        w2i, t2i, l2i, x2i = self._v2is
        dx, _ = DeltaX.from_penn(self._tree, factor, do_preproc = False) # [not here] watch for keyaki arg wordtrace for preproc_cnf
        if self._w_p is None:
            word, tag = dx.word_tag(w2i, t2i)
            word = np.asarray(word)
            tag  = np.asarray(tag)
            self._w_p = word, tag
        else:
            word, tag = self._w_p

        layers_of_labels = []
        layers_of_xtypes = []
        for labels, xtypes in dx.trapezoid_gen(self._trapezoid_height, l2i, x2i):
            labels = np.asarray(labels)
            xtypes = np.asarray(xtypes)
            layers_of_labels.append(labels)
            layers_of_xtypes.append(xtypes)

        factored = dict(token = word,
                        tag   = tag,
                        label = layers_of_labels,
                        xtype = layers_of_xtypes)
        self._factored[factor] = factored
        return factored

    def __str__(self):
        s = f'Keeper with ' + ', '.join(self._factored.keys()) + 'cached'
        return s

from unidecode import unidecode
class StanTreeKeeper:
    def __init__(self, line, v2is, trapezoid_height):
        self._line = line
        self._v2is = v2is
        self._factored = None
        self._trapezoid_height = trapezoid_height

    def update_factored(self, factored):
        self._factored = factored

    def get(self):
        if self._factored is None:

            w2i, p2i, x2i = self._v2is
            tree_str = self._line.replace(b'\\/', b'/').replace(b'\xc2\xa0', b'.').decode('utf-8')
            tree_str = unidecode(tree_str)
            tree = Tree.fromstring(tree_str)
            dx = DeltaX.from_stan(tree)
            self._words = words = tree.leaves()
            token = np.asarray([w2i(w) for w in words])

            layers_of_polars = []
            layers_of_xtypes = []
            for polars, xtypes in dx.trapezoid_gen(self._trapezoid_height, p2i, x2i):
                polars = np.asarray(polars)
                xtypes = np.asarray(xtypes)
                layers_of_polars.append(polars)
                layers_of_xtypes.append(xtypes)

            factored = dict(token = token,
                            polar = layers_of_polars,
                            xtype = layers_of_xtypes)
            self._factored = words, len(words), tree_str, factored
        return self._factored
        
# from data.multib import add_efficient_subs
class PennWorker(Process):
    def __init__(self, *args):
        Process.__init__(self)
        self._q_reader_fns_height_v2is_factors = args

    def run(self):
        (q, reader, fns, height, v2is, factors,
         word_trace) = self._q_reader_fns_height_v2is_factors

        for fn in fns:
            for tree in reader.parsed_sents(fn):
                try:
                    preproc_cnf(tree, word_trace = word_trace) # watch for ktb
                except:
                    print(tree)
                # _, tree = add_efficient_subs(tree)
                words = tree.leaves()
                length = len(words)
                keeper = PennTreeKeeper(tree, v2is, height)
                factored = {f: keeper[f] for f in factors}
                if '(' in words or ')' in words:
                    for i, word in enumerate(words):
                        if word == '(':
                            tree[tree.leaf_treeposition(i)] = '-LRB-'
                        elif word == ')':
                            tree[tree.leaf_treeposition(i)] = '-RRB-'
                results = words, length, str(tree), factored
                q.put(results)

class StanWorker(Process):
    def __init__(self, *args):
        Process.__init__(self)
        self._args = args

    def run(self):
        q, jobs, v2is, trapezoid_height = self._args
        for line in jobs:
            q.put(StanTreeKeeper(line, v2is, trapezoid_height).get())


def mp_workers(works, q, core_fn, num_threads):
    text = []
    lengths = []
    keepers = []
    with tqdm(desc = f'Receiving from {num_threads} threads ...') as qbar:
        try:
            while any(x.is_alive() for x in works):
                if q.empty():
                    sleep(0.00001)
                else:
                    words, length, keeper = core_fn(*q.get())
                    text.append(words)
                    lengths.append(length)
                    keepers.append(keeper)
                    qbar.update(1)
            qbar.desc = f'TreeKeepers'
        except KeyboardInterrupt as ex:
            with DelayedKeyboardInterrupt(ignore = True):
                for x in works:
                    x.kill()
            raise ex
    return text, lengths, keepers


from data.penn_types import Tree
from data.io import distribute_jobs
class TrapezoidDataset(LengthOrderedDataset):

    @classmethod
    def from_penn(cls,
                  reader,
                  get_fnames,
                  data_split,
                  trapezoid_height,
                  field_v2is,
                  paddings,
                  device,
                  factors,
                  word_trace,
                  min_len  = 0,
                  max_len  = None,
                  extra_text_helper = None,
                  num_threads = 0):

        _, w2i = field_v2is['token']
        _, t2i = field_v2is['tag']
        _, l2i = field_v2is['label']
        x2i = lambda x: xtype_to_logits(x, to_str = False)
        v2is = w2i, t2i, l2i, x2i
        
        fnames = get_fnames(data_split)
        if num_threads < 1:
            from utils.types import num_threads
        works = distribute_jobs(fnames, num_threads)
        q = Queue()
        for i in range(num_threads):
            w = PennWorker(q, reader, works[i], trapezoid_height, v2is, factors, word_trace)
            w.start()
            works[i] = w
        def core_fn(words, length, tree_str, factored):
            keeper = PennTreeKeeper(Tree.fromstring(tree_str), v2is, trapezoid_height)
            keeper.update_factored(factored, words)
            return words, length, keeper
        text, lengths, keepers = mp_workers(works, q, core_fn, num_threads)
        return cls('token tag label xtype', keepers, lengths, text,
                    trapezoid_height,
                    field_v2is,
                    paddings,
                    device,
                    factors,
                    min_len,
                    max_len,
                    extra_text_helper)

    @classmethod
    def from_stan(cls,
                  data_path,
                  trapezoid_height,
                  field_v2is,
                  paddings,
                  device,
                  factors,
                  min_len  = 0,
                  max_len  = None,
                  extra_text_helper = None,
                  num_threads = 0):

        _, w2i = field_v2is['token']
        _, p2i = field_v2is['polar']
        x2i = lambda x: xtype_to_logits(x, to_str = False)
        v2is = w2i, p2i, x2i
        
        if num_threads < 1:
            from utils.types import num_threads
        with open(data_path, 'rb') as fr:
            lines = list(fr)
        works = distribute_jobs(lines, num_threads)
        q = Queue()

        for i in range(num_threads):
            w = StanWorker(q, works[i], v2is, trapezoid_height)
            w.start()
            works[i] = w
        def core_fn(words, length, tree_str, factored):
            keeper = StanTreeKeeper(None, v2is, trapezoid_height)
            keeper.update_factored(factored)
            return words, length, keeper
        text, lengths, keepers = mp_workers(works, q, core_fn, num_threads)
        return cls('token polar xtype', keepers, lengths, text,
                    trapezoid_height,
                    field_v2is,
                    paddings,
                    device,
                    factors,
                    min_len,
                    max_len,
                    extra_text_helper)

    def __init__(self,
                 heads,
                 keepers,
                 lengths,
                 text,
                 trapezoid_height,
                 field_v2is,
                 paddings,
                 device,
                 factors,
                 min_len,
                 max_len,
                 extra_text_helper):

        heads = tuple(heads.split())
        if extra_text_helper:
            c2i = field_v2is['char'][1] if 'char' in field_v2is else None
            extra_text_helper = extra_text_helper(text, device, c2i)
        super().__init__(heads, lengths, factors, min_len, max_len, extra_text_helper)

        self._paddings_device_height = paddings, device, trapezoid_height
        self._keepers = tuple(keepers)

    def at_idx(self, idx, factor, length, helper_outputs):
        sample = self._keepers[idx]
        if factor is None:
            sample = sample.get() #?
        else:
            sample = sample[factor]
        sample['length'] = length
        return sample

    def _collate_fn(self, batch):
        dtype = np.int32
        field_columns = {}
        paddings, device, height = self._paddings_device_height

        for field, column in zip(self.heads, zip(*batch)):
            if field == 'length':
                batch_size = len(column)
                lengths = np.asarray(column, dtype)
                max_len = np.max(lengths)
                if paddings:
                    max_len += 2 # BOS and EOS
                    offsets = (max_len - lengths) // 2
                    field_columns['offset'] = offsets
                else:
                    field_columns['offset'] = np.zeros_like(lengths)
                full_triangular_len = s_index(max_len)
                tensor = lengths
            elif field in fields: # word or tags
                tensor = np.zeros([batch_size, max_len], dtype)
                for i_, (values, length) in enumerate(zip(column, lengths)):
                    if paddings:
                        start = offsets[i_]
                        end = start + length
                        bid, eid = paddings[field]
                        tensor[i_,    :start]  = bid
                        tensor[i_, start:end] = values
                        tensor[i_,      end:] = eid
                    else:
                        tensor[i_, :length] = values
                        # try:
                        # except:
                        #     import pdb; pdb.set_trace()
            else: # label or xtype
                tensor = np.zeros([batch_size, full_triangular_len], dtype = np.uint8)
                cumu_length = 0
                track_label = field in ('label', 'polar')
                if track_label:
                    segments = []
                    mask_length = np.zeros([batch_size], dtype)
                    seg_length = np.zeros([batch_size, max_len], dtype)
                    top3_label = np.stack([np.concatenate(x[-1:-3:-1]) for x in column]) # [batch, 3]

                for l_, layer in enumerate(zip_longest(*column)):
                    max_layer_len = max(len(x) for x in layer if x is not None)
                    if paddings:
                        max_layer_len += 2
                    cumu_length += max_layer_len
                    l_start = full_triangular_len - cumu_length
                    l_end   = l_start + max_layer_len
                    if track_label:
                        segments.append(max_layer_len)
                    for i_, seq in enumerate(layer):
                        if seq is None:
                            continue
                        seq_len = len(seq)
                        if track_label:
                            mask_length[i_] += max_layer_len
                            seg_length[i_, -1 - l_] = seq_len
                        if paddings:
                            bid, eid = paddings[field]
                            start = l_start + offsets[i_]
                            end   = start + seq_len
                            tensor[i_, l_start:start] = bid
                            tensor[i_, start:end] = seq
                            tensor[i_, end:l_end] = eid
                        else:
                            end = l_start + seq_len
                            tensor[i_, l_start:end] = seq
                tensor = tensor[:, -cumu_length:]

            field_columns[field] = tensor

        field_columns['mask_length'] = cumu_length - mask_length
        field_columns['top3_label']  = top3_label
        for f, column in field_columns.items():
            field_columns[f] = torch.as_tensor(column,
                                               dtype  = (None if f == 'xtype' else torch.long),
                                               device = device)

        segments.reverse()
        # height_segments = []
        # while segments:
        #     for i in count():
        #         if i % height == 0:
        #             height_segments.append(0)
        #         height_segments[-1] += segments.pop()
        #         if not segments:
        #             break
        # height_segments.reverse()
        field_columns['height'] = height
        field_columns['segment'] = segments
        field_columns['seg_length'] = seg_length[:, -len(segments):]

        # if len(segments) > 15: # even still sooooo sparse
        #     p_ = torch.arange(full_triangular_len, device = device)[None, :]
        #     x_ = p_ >= field_columns['mask_length'][:, None]
        #     import pdb; pdb.set_trace()

        return field_columns
