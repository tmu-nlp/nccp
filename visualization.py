#!/usr/bin/env python

import numpy as np
from collections import namedtuple, Counter, defaultdict
from itertools import count
from os.path import getsize, expanduser
from os import listdir, remove
import sys, pdb
from nltk.tree import Tree
from utils.file_io import join, isfile, parpath
from utils.pickle_io import pickle_load, pickle_dump
from utils.param_ops import HParams

inf_none_gen = (None for _ in count())

DiscoThresholds = namedtuple('DiscoThresholds', 'right, joint, direc')
IOVocab = namedtuple('IOVocab', 'vocabs, IOHead_fields, IOData_fields, thresholds')
class TensorVis:
    @classmethod
    def from_vfile(cls, fpath):
        vocabs, IOHead_fields, IOData_fields, thresholds = pickle_load(fpath)
        return cls(parpath(fpath), HParams(vocabs), IOHead_fields, IOData_fields, thresholds)

    def __init__(self, fpath, vocabs, IOHead_fields, IOData_fields, thresholds = None, clean_fn = None, fname = 'vocabs.pkl'):
        files = listdir(fpath)
        if fname in files:
            assert isfile(join(fpath, fname))
            if callable(clean_fn): # TODO
                clean_fn(files)
            anew = False
        else:
            assert isinstance(vocabs, HParams)
            pickle_dump(join(fpath, fname), IOVocab(vocabs._nested, IOHead_fields, IOData_fields, thresholds))
            anew = True
        self._anew = anew
        self._fpath = fpath
        self._vocabs = vocabs
        self._head_type = namedtuple('IOHead', IOHead_fields)
        self._data_type = namedtuple('IOData', IOData_fields)
        self._threshold = thresholds

    @property
    def is_anew(self):
        return self._anew

    def join(self, fname):
        return join(self._fpath, fname)

    @property
    def vocabs(self):
        return self._vocabs

    @property
    def IOHead(self):
        return self._head_type

    @property
    def IOData(self):
        return self._data_type

    @property
    def threshold(self):
        return self._threshold

from contextlib import ExitStack
class DiscontinuousTensorVis(TensorVis):
    def __init__(self, fpath, vocabs, thresholds):
        IOHead_fields = 'token, tag, label, right, joint, direc, tree, segment, seg_length'
        IOData_fields = 'token, tag, label, right, joint, direc, tree, segment, seg_length, mpc_word, mpc_phrase, warning, scores, tag_score, label_score, right_score, joint_score, direc_score'
        super().__init__(fpath, vocabs, IOHead_fields, IOData_fields, thresholds)

    def set_head(self, batch_id, size, *args):
        assert len(self._head_type._fields) == len(args)
        pickle_dump(self.join(f'head.{batch_id}_{size}.pkl'), args)

    def set_data(self, batch_id, epoch, *args):
        assert len(self._data_type._fields) == len(args)
        pickle_dump(self.join(f'data.{batch_id}_{epoch}.pkl'), args)

def tee_trees(join_fn, mode, lengths, trees, batch_id, bin_width):
    ftrees = {}
    write_all = batch_id is None
    write_len = isinstance(bin_width, int)
    with ExitStack() as stack:
        if write_all:
            fh = stack.enter_context(open(join_fn(f'{mode}.tree'), 'a'))
        else:
            fh = stack.enter_context(open(join_fn(f'{mode}.{batch_id}.tree'), 'w'))
        for wlen, tree in zip(lengths, trees):
            print(tree, file = fh)
            if write_len:
                wbin = wlen // bin_width
                if wbin in ftrees:
                    fw = ftrees[wbin]
                else:
                    fw = open(join_fn(f'{mode}.bin_{wbin}.tree'), 'a')
                    ftrees[wbin] = stack.enter_context(fw)
                print(tree, file = fw)
    return ftrees.keys()

from data.triangle import head_to_tree as tri_h2t
from data.triangle import data_to_tree as tri_d2t
from data.trapezoid import head_to_tree as tra_h2t
from data.trapezoid import data_to_tree as tra_d2t
from utils.shell_io import parseval, rpt_summary
class ContinuousTensorVis(TensorVis):
    def __init__(self, fpath, vocabs):
        IOHead_fields = 'offset, length, token, tag, label, right, tree, segment, seg_length'
        IOData_fields = 'offset, length, token, tag, label, right, tree, segment, seg_length, mpc_word, mpc_phrase, warning, scores, tag_score, label_score, split_score, summary'
        super().__init__(fpath, vocabs, IOHead_fields, IOData_fields)

    def set_head(self, fhtree, offset, length, token, tag, label, right, trapezoid_info, *batch_id_size_bin_width):
        old_tag = tag

        if tag is None:
            tag = inf_none_gen

        if trapezoid_info is None:
            segment = seg_length = None
            func_args = zip(offset, length, token, tag, label, right)
            func_args = ((*args, self.vocabs) for args in func_args)
            head_to_tree = tri_h2t
        else:
            segment, seg_length = trapezoid_info
            func_args = zip(offset, length, token, tag, label, right, seg_length)
            func_args = ((*args, segment, self.vocabs) for args in func_args)
            head_to_tree = tra_h2t

        trees = []
        for args in func_args:
            tree = str(head_to_tree(*args))
            tree = ' '.join(tree.split())
            print(tree, file = fhtree)
            trees.append(tree)

        if batch_id_size_bin_width:
            batch_id, size, bin_width = batch_id_size_bin_width

            fname = self.join(f'head.{batch_id}_{size}.pkl')
            head  = self.IOHead(offset, length, token, old_tag, label, right, trees, segment, seg_length)
            pickle_dump(fname, tuple(head)) # type check
            return tee_trees(self.join, 'head', length, trees, batch_id, bin_width)

    def set_void_head(self, batch_id, size, offset, length, token):
        fname = self.join(f'head.{batch_id}_{size}.pkl')
        head  = self.IOHead(offset, length, token, None, None, None, None, None, None)
        pickle_dump(fname, tuple(head))

    def set_data(self, fdtree, on_error, batch_id, epoch,
                 offset, length, token, tag, label, right, mpc_word, mpc_phrase,
                 tag_score, label_score, split_score,
                 trapezoid_info, size_bin_width_evalb):

        tree_kwargs = dict(return_warnings = True, on_error = on_error)
        error_prefix = f'  [{batch_id} {epoch}'

        old_tag   = tag
        old_tag_s = tag_score
        if tag is None: tag = inf_none_gen
        if label is None:
            label_ = label_score
            tree_kwargs['error_root'] = 'NA'
        else:
            label_ = label
        trees = []
        batch_warnings = []
        if trapezoid_info is None:
            segment = seg_length = None
            func_args = zip(offset, length, token, tag, label_, right)
            func_args = ((*args, self.vocabs) for args in func_args)
            data_to_tree = tri_d2t
        else:
            segment, seg_length = trapezoid_info
            func_args = zip(offset, length, token, tag, label_, right, seg_length)
            func_args = ((*args, segment, self.vocabs) for args in func_args)
            data_to_tree = tra_d2t

        for i, args in enumerate(func_args):
            tree_kwargs['error_prefix'] = error_prefix + f']-{i} len={args[1]}'
            tree, warnings = data_to_tree(*args, **tree_kwargs)
            tree = str(tree)
            tree = ' '.join(tree.split())
            trees.append(tree)
            print(tree, file = fdtree) # TODO use stack to protect opened file close and delete
            batch_warnings.append(warnings)

        if size_bin_width_evalb:
            size, bin_width, evalb = size_bin_width_evalb
            if label is None: # unlabeled
                pickle_dump(self.join(f'data.{batch_id}_{epoch}.pkl'),
                            (offset, length, token, None, None, right, trees,
                             segment, seg_length, mpc_word, mpc_phrase,
                             batch_warnings, None, tag_score, label_score, split_score, None))
                return batch_warnings

            else: # supervised / labeled
                tee_trees(self.join, 'data', length, trees, batch_id, bin_width)

                fhead = f'head.{batch_id}_{size}.pkl'
                assert isfile(self.join(fhead)), f"Need a head '{fhead}'"
                fhead = self.join(f'head.{batch_id}.tree')
                fdata = self.join(f'data.{batch_id}.tree')

                if evalb is None: # sentiment
                    # 52VII
                    from data.stan_types import calc_stan_accuracy
                    idv, smy, key_score = calc_stan_accuracy(fhead, fdata, error_prefix, on_error)
                else: # constituency
                    proc = parseval(evalb, fhead, fdata)
                    idv, smy = rpt_summary(proc.stdout.decode(), True, True)

                    fname = self.join('summary.pkl')
                    if isfile(fname):
                        summary = pickle_load(fname)
                    else:
                        summary = {}
                    summary[(batch_id, epoch)] = smy
                    pickle_dump(fname, summary)

                    key_score = smy['F1']

                fdata = self.join(f'data.{batch_id}_{epoch}.pkl')
                data = self.IOData(offset, length, token, old_tag, label, right, trees,
                            segment, seg_length, mpc_word, mpc_phrase,
                            batch_warnings, idv, tag_score, label_score, split_score, smy)
                pickle_dump(fdata, tuple(data))

                return key_score

# dpi_value     = master.winfo_fpixels('1i')
# master.tk.call('tk', 'scaling', '-displayof', '.', dpi_value / 72.272)
# screen_shape = master.winfo_screenwidth(), master.winfo_screenheight()
# master.geometry("%dx%d+%d+%d" % (canvas_shape + tuple(s/2-c/2 for s,c in zip(screen_shape, canvas_shape))))
        
try:
    from utils.gui import *
    desktop = True
except ImportError:
    desktop = False

if desktop:
    def _font(x):
        font_name, font_min_size, font_max_size = x.split()
        font_min_size = int(font_min_size)
        font_max_size = int(font_max_size)
        assert font_min_size > 0, 'font minimun size should be positive' 
        assert font_max_size > font_min_size, 'font maximun size should be greather than min size'
        return font_name, font_min_size, font_max_size

    def _ratio(x):
        x = float(x) if '.' in x else int(x)
        assert x > 0, 'should be positive'
        return x

    def _frac(x):
        x = float(x)
        assert 0 < x < 1, 'should be a fraction'
        return x

    def _size(x):
        x = int(x)
        assert x > 0, 'should be positive'
        return x

    def _dim(x):
        x = int(x)
        assert 0 < x < 10, 'should be in [1, 9]'
        return x

    def _offset(x):
        return int(x)

    def _curve(x):
        x = x.strip()
        assert 'x' in x, 'should contain variable x'
        if ':' not in x:
            x = 'lambda x:' + x
        curve = eval(x)
        assert callable(curve), 'shoule be a function'
        assert isinstance(curve(0), float), 'should return a float number'
        return curve

    BoolList = namedtuple('BoolList', 'delta_shape, show_errors, show_paddings, show_nil, dark_background, inverse_brightness, align_coord, show_color, force_bottom_color, statistics')
    CombList = namedtuple('CombList', 'curve, dash, gauss, picker, spotlight')
    DynamicSettings = namedtuple('DynamicSettings', BoolList._fields + tuple('apply_' + f for f in CombList._fields) + CombList._fields)
    NumbList = namedtuple('NumbList', 'font, pc_x, pc_y, line_width, offset_x, offset_y, word_width, word_height, yx_ratio, histo_width, scatter_width')
    numb_types = NumbList(_font, _dim, _dim, _ratio, _offset, _offset, _size, _size, _ratio, _size, _size)
    comb_types = CombList(_curve, _frac, _ratio, _ratio, _ratio)
    PanelList = namedtuple('PanelList', 'hide_listboxes, detach_viewer')
    navi_directions = '⇤↑o↓⇥'

    from colorsys import hsv_to_rgb, hls_to_rgb
    from tempfile import TemporaryDirectory
    from nltk.draw import TreeWidget
    from nltk.draw.util import CanvasFrame
    from time import time, sleep
    from utils.math_ops import uneven_split, itp
    from math import exp, sqrt, pi
    from functools import partial
    from data.delta import warning_level, NIL
    from utils.param_ops import more_kwargs
    from utils.file_io import path_folder
    from concurrent.futures import ProcessPoolExecutor
    from data.triangle import triangle_to_layers
    from data.trapezoid import trapezoid_to_layers, inflate
    from data.cross import draw_str_lines
    # from multiprocessing import Pool

    class PathWrapper:
        def __init__(self, fpath, sftp):
            fpath, folder = path_folder(fpath)
            self._folder = '/'.join(folder[-2:])
            if sftp is not None:
                sftp.chdir(fpath)
                fpath = TemporaryDirectory()
            self._fpath_sftp = fpath, sftp

        def join(self, fname):
            fpath, sftp = self._fpath_sftp
            if sftp is not None:
                f = join(fpath.name, fname)
                if fname not in listdir(fpath.name):
                    print('networking for', fname)
                    start = time()
                    sftp.get(fname, f)
                    print('transfered %d KB in %.2f sec.' % (getsize(f) >> 10, time() - start))
                else:
                    print('use cached', fname)
                return f
            return join(fpath, fname)

        @property
        def base(self):
            fpath, sftp = self._fpath_sftp
            if sftp is not None:
                return fpath.name
            return fpath

        @property
        def folder(self):
            return self._folder

        def listdir(self):
            fpath, sftp = self._fpath_sftp
            if sftp is not None:
                return sftp.listdir()
            return listdir(fpath)

        def __del__(self):
            fpath, sftp = self._fpath_sftp
            if sftp is not None:
                # fpath.cleanup() # auto cleanup
                sftp.close()

    get_batch_id = lambda x: int(x[5:-4].split('_')[0])

    class TreeViewer(Frame):
        def __init__(self,
                     root,
                     fpath):
            super().__init__(root) # To view just symbolic trees ptb or xml.

    class TreeExplorer(Frame):
        def __init__(self,
                     root,
                     fpath,
                     initial_bools = BoolList(True, False, False, False, False, True, False, True, False, False),
                     initial_numbs = NumbList('System 6 15', (1, 1, 9, 1), (2, 1, 9, 1), (4, 2, 10, 2), (0, -200, 200, 10), (0, -200, 200, 10), (80, 60, 200, 5), (22, 12, 99, 2), (0.9, 0.5, 2, 0.1), (60, 50, 120, 10), (60, 50, 120, 10)),
                     initial_panel = PanelList(False, False),
                     initial_combs = CombList((True, 'x ** 0.5'), (False, (0.5, 0.1, 0.9, 0.1)), (True, (0.04, 0.01, 0.34, 0.01)), (True, (0.2, 0.1, 0.9, 0.1)), (False, (200, 100, 500, 100)))):
            vocabs = fpath.join('vocabs.pkl')
            if isfile(vocabs):
                self._tvis = TensorVis.from_vfile(vocabs)
            else:
                raise ValueError(f"The folder should at least contains a vocab file '{vocabs}'")

            self._fpath_heads = fpath, None
            self._last_init_time = None

            super().__init__(root)
            self.master.title(fpath.folder)
            headbox = Listbox(self, relief = SUNKEN, font = 'TkFixedFont')
            sentbox = Listbox(self, relief = SUNKEN)
            self._boxes = headbox, sentbox
            self.initialize_headbox()
            self._sent_cache = {}
            self._cross_warnings = {}

            headbox.bind('<<ListboxSelect>>', self.read_listbox)
            sentbox.bind('<<ListboxSelect>>', self.read_listbox)

            pack_kwargs = dict(padx = 10, pady = 5)
            intr_kwargs = dict(pady = 2)

            control = [1 for i in range(len(initial_numbs))]
            control[0] = dict(char_width = 17)
            control_panel = Frame(self, relief = SUNKEN)
            ckb_panel = Frame(control_panel)
            etr_panel = Frame(control_panel)
            self._checkboxes = make_namedtuple_gui(make_checkbox, ckb_panel, initial_bools, self._change_bool,   **intr_kwargs)
            self._entries    = make_namedtuple_gui(make_entry,    etr_panel, initial_numbs, self._change_string, control)
            ckb_panel.pack(side = TOP, fill = X, **pack_kwargs)
            etr_panel.pack(side = TOP, fill = X, **pack_kwargs) # expand means to pad
            self._last_bools = initial_bools
            self._last_numbs = NumbList(*(x[0] if isinstance(x, tuple) else x for x in initial_numbs))

            comb_panel = Frame(control_panel)
            self._checkboxes_entries = make_namedtuple_gui(make_checkbox_entry, comb_panel, initial_combs, (self._change_bool, self._change_string), (2, 1, 1, 1, 1))
            comb_panel.pack(side = TOP, fill = X, **pack_kwargs)
            self._last_combs = CombList(*((x[0], x[1][0]) if isinstance(x[1], tuple) else x for x in initial_combs))
            
            view_panel = Frame(control_panel)
            self._panels = make_namedtuple_gui(make_checkbox, view_panel, initial_panel, self.__update_viewer, **intr_kwargs)
            view_panel.pack(side = TOP, fill = X, **pack_kwargs)
            self._last_panel_bools = initial_panel

            navi_panel = Frame(control_panel)
            navi_panel.pack(fill = X)
            btns = tuple(Button(navi_panel, text = p) for p in navi_directions)
            for btn in btns:
                btn.bind("<Button-1>", self._navi)
                btn.pack(side = LEFT, fill = X, expand = YES)
            self._navi_btns = btns

            btn_panel = Frame(control_panel)
            btn_panel.pack(side = TOP, fill = X) # no need to expand, because of st? can be bottom
            st = Button(btn_panel, text = 'Show Trees', command = self._show_one_tree )
            sa = Button(btn_panel, text = '♾', command = self._show_all_trees)
            sc = Button(btn_panel, text = '◉', command = self._save_canvas   )
            st.pack(side = LEFT, fill = X, expand = YES)
            sa.pack(side = LEFT, fill = X)
            sc.pack(side = LEFT, fill = X)
            self._panel_btns = st, sa, sc

            self._control_panel = control_panel
            control_panel.bind('<Key>', self.shortcuts)
            headbox.bind('<Key>', self.shortcuts)
            sentbox.bind('<Key>', self.shortcuts)

            self._viewer = None
            self._selected = tuple()
            self._init_time = 0
            self.__update_layout(True, True)

        def initialize_headbox(self):
            fpath = self._fpath_heads[0]
            headbox = self._boxes[0]
            headbox.delete(0, END)
            fnames = fpath.listdir()
            heads = [f for f in fnames if f.startswith('head.') and f.endswith('.pkl')]
            heads.sort(key = get_batch_id)
            
            if len(heads) == 0:
                raise ValueError("'%s' is an invalid dir" % fpath.base)

            if 'summary.pkl' in fnames:
                from math import isnan
                summary = pickle_load(fpath.join('summary.pkl'))
                summary_fscores = defaultdict(list)
                for (batch_id, epoch), smy in summary.items():
                    if not isnan(smy['F1']):
                        summary_fscores[batch_id].append((smy['F1'], smy.get('DF', None)))
            else:
                summary_fscores = {}

            max_id_len = len(str(max(summary_fscores.keys()))) if summary_fscores else 4
            for h in heads:
                bid = get_batch_id(h)

                if bid in summary_fscores:
                    if len(summary_fscores[bid]) == 1:
                        f1, df = summary_fscores[bid][0]
                        fscores = f'  {f1:.2f}' if df is None else f'  {f1:.2f} ({df:.2f})'
                    else:
                        f1, df = max(summary_fscores[bid], key = lambda x: x[0])
                        fscores = f'  ≤{f1:.2f}' if df is None else f'  ≤{f1:.2f} ({df:.2f})'
                else:
                    fscores = ''
                str_bid, length = h[5:-4].split('_')
                length = '≤' + length
                headbox.insert(END, str_bid.rjust(max_id_len) + length.rjust(3) + fscores)
            self._fpath_heads = fpath, heads
            self._last_init_time = time()

        # def __del__(self):
        #     if self._rpt.alabelc:
        #         print(f'terminate receiving {len(self._rpt.alabelc)} rpt files')
        #         self._rpt.pool.terminate()

        def __update_layout(self, hide_listboxes_changed, detach_viewer_changed):
            headbox, sentbox = self._boxes
            control_panel = self._control_panel
            viewer = self._viewer

            if detach_viewer_changed:
                if viewer and viewer.winfo_exists():
                    self._init_time = viewer.time
                    viewer.destroy()
                # widget shall be consumed within a function, or they will be visible!
                master = Toplevel(self) if self._last_panel_bools.detach_viewer else self
                viewer = ThreadViewer(master, self._tvis, self._change_title)
                viewer.bind('<Key>', self.shortcuts)
                self._viewer = viewer

            if hide_listboxes_changed:
                if self._last_panel_bools.hide_listboxes:
                    headbox.pack_forget()
                    sentbox.pack_forget()
                else:
                    control_panel.pack_forget()
                    viewer.pack_forget()

                    headbox.pack(fill = Y, side = LEFT)
                    sentbox.pack(fill = BOTH, side = LEFT, expand = YES)
                    control_panel.pack(fill = Y, side = LEFT)
                    viewer.pack(fill = BOTH, expand = YES)

            self.pack(fill = BOTH, expand = YES)

        def _change_title(self, prefix, epoch):
            title = [self._fpath_heads[0].folder, prefix]
            # bid, _, _, sid = self._selected
            # key = f'{bid}_{epoch}'
            # if key in self._rpt.sent:
            #     scores = self._rpt.sent[key][sid]
            #     if len(scores) == 4:
            #         scores = '  '.join(i+f'({j:.2f})' for i, j in zip(('5C@R', 'N-P@R', '5C', 'N-P'), scores))
            #     else:
            #         scores = tuple(scores[i] for i in (1, 3, 4, 11))
            #         scores = '  '.join(i+f'({str(j)})' for i, j in zip(('len.', 'P.', 'R.', 'tag.'), scores))
                # else:
                #     r = self._fpath_heads[0].join(f'data.{key}.rpt')
                #     scores = self._rpt.sent[r[5:-4]] = rpt_summary(r, sents = True)
                    
                # title.append(scores)
            self.master.title(' | '.join(title))

        def read_listbox(self, event):
            headbox, sentbox = self._boxes
            choice_t = event.widget.curselection()
            if choice_t:
                fpath, heads = self._fpath_heads
                i = int(choice_t[0]) # head/inter-batch id or sentence/intra-batch id
                if event.widget is headbox:
                    IOHead = self._tvis.IOHead
                    IOData = self._tvis.IOData
                    head = fpath.join(heads[i])
                    bid, num_word = (int(i) for i in heads[i][5:-4].split('_'))
                    head = IOHead(*pickle_load(head))
                    if head.tag is None and head.label is not None:
                        polar_vocab = self._tvis.vocabs.polar
                        neg_set = set(polar_vocab.index(i) for i in '01')
                        pos_set = set(polar_vocab.index(i) for i in '34')
                            
                    sentbox.delete(0, END)
                    is_a_conti_task = 'offset' in IOHead._fields
                    if is_a_conti_task: # tri/tra
                        offsets = head.offset
                        lengths = head.length
                        is_triangular = head.segment is None and head.label is not None
                    else: # cross
                        batch_size = head.token.shape[0]
                        offsets = (1 for _ in range(batch_size))
                        lengths = head.seg_length[:, 0]
                    for sid, (offset, length, words) in enumerate(zip(offsets, lengths, head.token)):
                        if head.tag is None and head.label is not None:
                            negation = any(i in head.label[sid] for i in pos_set) and any(i in head.label[sid] for i in neg_set)
                        else:
                            negation = False
                        mark = '*' if negation else ''
                        mark += f'{sid + 1}'
                        # if warning_cnt:
                        #     mark += " ◌•▴⨯"[warning_level(warning_cnt)]
                        mark += '\t'
                        tokens = '' if head.label is None else ' '
                        tokens = tokens.join(self._tvis.vocabs.token[idx] for idx in words[offset:offset + length])
                        sentbox.insert(END, mark + tokens)

                    head_ = []
                    if is_a_conti_task: # depend on task senti/parse | tri
                        if is_triangular:
                            sample_gen = (inf_none_gen if h is None else h for h in head)
                            for sample in zip(*sample_gen):
                                values = []
                                for field, value in zip(IOHead._fields, sample):
                                    if value is not None and field in ('label', 'right'):
                                        value = triangle_to_layers(value) 
                                    values.append(value)
                                head_.append(IOHead(*values))

                            prefix, suffix = f'data.{bid}_', '.pkl'
                            for fname_time in fpath.listdir():
                                if fname_time.startswith(prefix) and fname_time.endswith(suffix):
                                    if fname_time not in self._sent_cache:
                                        data       = IOData(*pickle_load(fpath.join(fname_time)))
                                        sample_gen = (inf_none_gen if x is None else x for x in data[:-1])
                                        data_      = []
                                        for sample in zip(*sample_gen):
                                            values = []
                                            for field, value in zip(IOData._fields, sample): # TODO: open for unsup?
                                                if value is not None and field in ('label', 'right', 'label_score', 'split_score'):
                                                    value = triangle_to_layers(value)
                                                values.append(value)
                                            sample = IOData(*values, inf_none_gen)
                                            stat = SentenceEnergy(num_word, sample.mpc_word, sample.mpc_phrase,
                                                                  sample.offset, sample.length, None, None)
                                            data_.append((sample, stat))
                                        self._sent_cache[fname_time] = data_, data[-1]
                        else: # trapezoidal
                            sample_gen = (inf_none_gen if h is None or f == 'segment' else h for f,h in zip(IOHead._fields, head))
                            for sample in zip(*sample_gen):
                                values = dict(zip(IOHead._fields, sample))
                                for field in ('label', 'right'):
                                    if values[field] is not None: # trapezoids for supervised parsing
                                        values[field] = inflate(trapezoid_to_layers(values[field], head.segment, values['seg_length']))
                                values['segment'] = head.segment
                                head_.append(IOHead(**values))

                            prefix, suffix = f'data.{bid}_', '.pkl'
                            for fname_time in fpath.listdir():
                                if fname_time.startswith(prefix) and fname_time.endswith(suffix):
                                    if fname_time not in self._sent_cache:
                                        data       = IOData(*pickle_load(fpath.join(fname_time)))
                                        sample_gen = (inf_none_gen if d is None or f in ('segment', 'summary') else d for f,d in zip(IOData._fields, data))
                                        data_      = []
                                        for sample in zip(*sample_gen):
                                            values = dict(zip(IOData._fields, sample))
                                            for field in ('label', 'right', 'label_score', 'split_score'):
                                                if field == 'label' and values[field] is None:
                                                    values[field] = values['label_score']
                                                if values[field] is not None: # trapezoid for supervised parsing
                                                    values[field] = inflate(trapezoid_to_layers(values[field], data.segment, values['seg_length']))
                                            values['segment'] = data.segment
                                            sample = IOData(**values)
                                            stat = SentenceEnergy(num_word, sample.mpc_word, sample.mpc_phrase,
                                                                  sample.offset, sample.length,
                                                                  sample.segment, sample.seg_length)
                                            data_.append((sample, stat))
                                        self._sent_cache[fname_time] = data_, data[-1]
                    else: # discontinuous task
                        sample_gen = (inf_none_gen if f == 'segment' else h for f,h in zip(IOHead._fields, head))
                        for sample in zip(*sample_gen):
                            values = dict(zip(IOHead._fields, sample))
                            for field in ('label', 'right', 'direc'):
                                values[field] = trapezoid_to_layers(values[field], head.segment, head.segment, big_endian = False)
                            joint_segment = (head.segment - 1)[:-1]
                            values['joint'] = trapezoid_to_layers(values['joint'], joint_segment, joint_segment, big_endian = False)
                            values['segment'] = head.segment
                            head_.append(IOHead(**values))
                        
                        prefix, suffix = f'data.{bid}_', '.pkl'
                        for fname_time in fpath.listdir():
                            if fname_time.startswith(prefix) and fname_time.endswith(suffix):
                                if fname_time not in self._sent_cache:
                                    data       = IOData(*pickle_load(fpath.join(fname_time)))
                                    sample_gen = (inf_none_gen if f == 'segment' else d for f,d in zip(IOData._fields, data))
                                    data_      = []
                                    for sample in zip(*sample_gen):
                                        values = dict(zip(IOData._fields, sample))
                                        for field in ('label', 'right', 'direc', 'label_score', 'right_score', 'direc_score'):
                                            values[field] = trapezoid_to_layers(values[field], data.segment, data.segment, big_endian = False)
                                        joint_segment = [x - 1 for x in data.segment[:-1]]
                                        for field in ('joint', 'joint_score'):
                                            values[field] = trapezoid_to_layers(values[field], joint_segment, joint_segment, big_endian = False)
                                        values['segment'] = data.segment
                                        sample = IOData(**values)
                                        stat = SentenceEnergy(num_word, sample.mpc_word, sample.mpc_phrase,
                                                              1, None, sample.segment, sample.seg_length, False)
                                        data_.append((sample, stat))
                                        # import pdb; pdb.set_trace()
                                    self._sent_cache[fname_time] = data_, data[-1]
                    self._selected = bid, num_word, head_

                elif event.widget is sentbox:
                    bid, num_word, head = self._selected[:3]
                    self._selected = bid, num_word, head, i
                    self.__update_viewer()
            else:
                print('nothing', choice_t, event)

        def _change_bool(self):
            if self._viewer.ready():
                self._viewer.minor_change(self.dynamic_settings(), self.static_settings(0, 1, 2, 3, 4, 5))

        def _change_string(self, event):
            if any(not hasattr(self, attr) for attr in '_entries _panels _checkboxes_entries _checkboxes'.split()):
                return
            if event is None or event.char in ('\r', '\uf700', '\uf701'):
                ss = self.static_settings()
                ss_changed = NumbList(*(t != l for t, l in zip(ss, self._last_numbs)))
                if any(ss_changed[6:]): # font and PCs will not cause resize
                    self.__update_viewer(force_resize = True)
                elif self._viewer.ready():
                    self._viewer.minor_change(self.dynamic_settings(), ss[:6])
                self._last_numbs = ss
                if event and event.char == '\r':
                    self._control_panel.focus()
                
        def shortcuts(self, key_press_event):
            char = key_press_event.char
            # self.winfo_toplevel().bind(key, self._navi)
            if char == 'w':
                self._checkboxes.statistics.ckb.invoke()
                sub_state = 'normal' if self._last_bools.statistics else 'disable'
                self._checkboxes_entries.gauss.ckb.configure(state = sub_state)
                self._checkboxes_entries.gauss.etr.configure(state = sub_state)
            elif char == 'n':
                self._checkboxes.show_nil.ckb.invoke()
            elif char == 'b':
                self._checkboxes.show_paddings.ckb.invoke()
            elif char == 'e':
                self._checkboxes.show_errors.ckb.invoke()
            elif char == '|':
                self._checkboxes.align_coord.ckb.invoke()
            elif char == ',':
                self._checkboxes.force_bottom_color.ckb.invoke()
            # elif char == 'v':
            #     self._checkboxes.hard_split.ckb.invoke()
            #     if self._last_bools.apply_dash:
            #         self._checkboxes.apply_dash.ckb.invoke()
            elif char == '.':
                self._checkboxes_entries.dash.ckb.invoke()
                # if self._last_bools.hard_split: # turn uni dire off
                #     self._checkboxes.hard_split.ckb.invoke()
            elif char == '\r':
                self._checkboxes_entries.curve.etr.icursor("end")
                self._checkboxes_entries.curve.etr.focus()
            elif char == 'i':
                self._checkboxes.dark_background.ckb.invoke()
                self._checkboxes.inverse_brightness.ckb.invoke()
            elif char == 'u':
                self._checkboxes_entries.curve.ckb.invoke()
            elif char == 'p':
                self._checkboxes_entries.picker.ckb.invoke()
            elif char == 'g':
                self._checkboxes_entries.gauss.ckb.invoke()
            elif char == 'l':
                self._checkboxes_entries.spotlight.ckb.invoke()
            elif char == 'q':
                self._panels.hide_listboxes.ckb.invoke()
            elif char == 'z':
                self._panel_btns[0].invoke()
            elif char == 'x':
                self._panel_btns[1].invoke()
            elif char == 'c':
                self._panel_btns[2].invoke()
            elif char == 'a':
                self._navi('⇤')
            elif char == 's':
                self._navi('↑')
            elif char == ' ':
                self._navi('o')
            elif char == 'd':
                self._navi('↓')
            elif char == 'f':
                self._navi('⇥')
            else:
                print('???', key_press_event)

        def __update_viewer(self, force_resize = False):
            panel_bools = get_checkbox(self._panels)
            changed = (t^l for t, l in zip(panel_bools, self._last_panel_bools))
            changed = PanelList(*changed)
            self._last_panel_bools = panel_bools
            
            viewer = self._viewer
            self.__update_layout(changed.hide_listboxes, changed.detach_viewer or not viewer.winfo_exists())
            viewer = self._viewer

            if len(self._selected) < 4:
                print('selected len:', len(self._selected))
                return

            bid, num_word, head, sid = self._selected
            # import pdb; pdb.set_trace()
            prefix = f"data.{bid}_"
            suffix = '.pkl'
            timeline = []
            for fname in self._sent_cache:
                if fname.startswith(prefix):
                    timeline.append(fname)

            timeline.sort(key = lambda kv: float(kv[len(prefix):-len(suffix)]))

            num_time = len(timeline)
            if force_resize or not self._viewer.ready(num_word, num_time):
                ds = self.dynamic_settings()
                ss = self.static_settings()
                viewer.configure(num_word, num_time, ds, ss, self._init_time)
                
            viewer.set_framework(head, {t:self._sent_cache[t] for t in timeline})
            viewer.show_sentence(sid)
            viewer.update() # manually update canvas

        def dynamic_settings(self):
            cs = (n for b,n in self._last_combs)
            ns = get_entry(self._checkboxes_entries, comb_types, (n for b,n in self._last_combs), 1)
            self._last_combs = CombList(*zip(cs, ns))

            bs = get_checkbox(self._checkboxes)
            ds = bs + get_checkbox(self._checkboxes_entries, 1) + ns

            # changed_bs = (t^l for t, l in zip(bs, self._last_bools))
            # changed_bs = BoolList(*changed_bs)
            self._last_bools = bs

            # if bs.show_errors and not bs.show_nil:
            #     if changed_bs.show_errors:
            #         self._checkboxes.show_nil.ckb.invoke()
            #     elif changed_bs.show_nil:
            #         self._checkboxes.show_errors.ckb.invoke()
            return DynamicSettings(*ds)

        def static_settings(self, *ids):
            if ids:
                enties = tuple(self._entries[i]    for i in ids)
                types  = tuple(numb_types[i]       for i in ids)
                lasts  = tuple(self._last_numbs[i] for i in ids)
                return get_entry(enties, types, lasts)
            return get_entry(self._entries, numb_types, self._last_numbs)

        def _show_one_tree(self):
            if self._viewer.ready():
                self._viewer.show_tree(False)#, self._viewer, self._viewer._boards[0])

        def _show_all_trees(self):
            if self._viewer.ready():
                self._viewer.show_tree(True)

        def _navi(self, event):
            if self._viewer.ready():
                if isinstance(event, str):
                    self._viewer.navi_to(event)
                elif event.widget in self._navi_btns:
                    self._viewer.navi_to(navi_directions[self._navi_btns.index(event.widget)])

        def _save_canvas(self):
            if not self._viewer.ready():
                return
            bid, _, _, i = self._selected
            options = dict(filetypes   = [('postscript', '.eps')],
                           initialfile = f'{bid}-{i}-{self._viewer.time}.eps',
                           parent      = self)

            filename = filedialog.asksaveasfilename(**options)
            if filename:
                self._viewer.save(filename)

        def _calc_batch(self):
            pass

    def to_circle(xy, xy_orig = None):
        if xy_orig is not None:
            xy = xy - xy_orig
        x, y = (xy[:, i] for i in (0,1))
        reg_angle = np.arctan2(y, x) / np.pi
        reg_angle += 1
        reg_angle /= 2
        reg_power = np.sqrt(np.sum(xy ** 2, axis = 1))
        return np.stack([reg_angle, reg_power], axis = 1), np.max(reg_power)

    def filter_data_coord(x, offset_length, filtered):
        if offset_length is None:
            coord = range(x.shape[0])
        else:
            offset, length = offset_length
            end = offset + length
            coord = range(offset, end)
            x = x[offset:end]
            if filtered is not None:
                filtered = filtered[offset:end]

        if filtered is not None:
            x = x[filtered]
            coord = (c for c, f in zip(coord, filtered) if f)

        return x, coord

    class LayerMPCStat:
        def __init__(self, mpc):
            self._global_data = mpc
            self._xy_dims     = None
            self._histo_cache = {}
            self._scatt_cache = {}

        def _without_paddings(self, offset_length):
            return filter_data_coord(self._global_data, offset_length, None)

        def histo_data(self, width, gaussian, distance_or_bin_width, global_xmax = None, offset_length = None, filtered = None):
            key = width, gaussian, distance_or_bin_width, global_xmax is None, offset_length is None, filtered is None
            if key in self._histo_cache:
                return self._histo_cache[key]
            x, coord = filter_data_coord(self._global_data[:, 0], offset_length, filtered)
            if global_xmax is None:
                xmin = np.min(x)
                xmax = np.max(x)
            else:
                xmin = 0
                xmax = global_xmax
            if xmin == xmax:
                xmin -= 0.1
                xmax += 0.1

            x = x - xmin # new memory, not in-place
            x /= xmax - xmin # in range [0, 1]
            if gaussian:
                a = (2 * distance_or_bin_width ** 2)
                yi = np.empty([width])
                for i in range(width):
                    xi = i / width
                    dx = (xi - x) ** 2
                    yi[i] = np.sum(1 * np.exp(-dx / a))
                yi /= np.max(yi)
                xy = tuple(zip(range(width), yi))
            else:
                num_backle = width // distance_or_bin_width
                tick = 1 / num_backle
                tock = num_backle + 1
                x //= tick # bin_width is float, even for floordiv
                x /= tock # discrete x for coord [0, 1), leave 1 for final
                x += 0.5 / tock
                collapsed_x = Counter(x * width)
                ymax = max(collapsed_x.values()) if collapsed_x else 1
                xy = tuple((x, y/ymax) for x, y in collapsed_x.items())
            # for old_key in self._histo_cache:
            #     if old_key[1] == gaussian:
            #         self._histo_cache.pop(old_key)
            self._histo_cache[key] = cached = xy, xmin, xmax, tuple(zip(coord, x))
            return cached

        def scatter_data(self, xy_min_max = None, offset_length = None, filtered = None):
            key = xy_min_max is None, offset_length is None, filtered is None
            if key in self._scatt_cache:
                return self._scatt_cache[key]
            xy, coord = filter_data_coord(self._global_data[:, self._xy_dims], offset_length, filtered)
            if xy_min_max is None:
                xmin = np.min(xy, 0)
                xmax = np.max(xy, 0)
            else:
                xmin, xmax = xy_min_max

            g_orig = np.zeros_like(xmin)
            invalid = g_orig < xmin
            xmin = np.where(invalid, g_orig, xmin)
            invalid = g_orig > xmax
            xmax = np.where(invalid, g_orig, xmax)

            if np.all(xmin == xmax):
                xmin -= 0.1
                xmax += 0.1

            xy = xy - xmin
            xy /= xmax - xmin
            g_orig -= xmin
            g_orig /= xmax - xmin
            self._scatt_cache[key] = cached = xy, xmin, xmax, g_orig, tuple(coord)
            return cached
            # print('xy', xy.shape[0])
            # print('seq_len', seq_len if seq_len else 'None')
            # print('filtered', f'{sum(filtered)} in {len(filtered)}' if filtered else 'None')
            # print('coord', coord)

        def colorify(self, m_max, xy_dims):
            self._xy_dims = xy_dims
            self._global_m_max = m_max
            ene = np.expand_dims(self._global_data[:, 0], 1) / m_max
            ori, sature_max = to_circle(self._global_data[:, xy_dims])
            self._render = np.concatenate([ene, ori], axis = 1)
            return sature_max

        def seal(self, sature_max):
            self._render[:, 2] /= sature_max
            return self

        def __getitem__(self, idx):
            return self._render[idx]

        def __len__(self):
            return self._global_data.shape[0]

    class SentenceEnergy:
        def __init__(self, size, mpc_word, mpc_phrase, offset, length, segment, seg_length, is_contiuous = True):
            mpc_all = mpc_phrase
            if segment is None:
                stats = phrase = tuple(LayerMPCStat(l) for l in triangle_to_layers(mpc_phrase))
            else:
                if is_contiuous:
                    layers = inflate(trapezoid_to_layers(mpc_phrase, segment, seg_length)) # TODO check  seg seg
                else:
                    layers = trapezoid_to_layers(mpc_phrase, segment, segment, big_endian = False)
                stats = phrase = tuple(l if l is None else LayerMPCStat(l) for l in layers)

            if mpc_word is not None:
                mpc_all = np.concatenate([mpc_word, mpc_phrase])
                token = LayerMPCStat(mpc_word) # TODO check [offset:offset + length]
                stats = (token,) + phrase
            else:
                token = None

            self._min_all = xmax = np.min(mpc_all, 0)
            self._max_all = xmin = np.max(mpc_all, 0)
            for i, stat in enumerate(stats):
                if is_contiuous:
                    if i == 0:
                        lo = offset, length
                        ct = 0
                    else:
                        _len = length - i + 1 # pitari without +1
                        if _len <= 0 or stat is None:
                            continue
                        lo = offset, _len
                else: # discontinuous
                    if i == 0:
                        lo = 1, seg_length[i]
                    else:
                        lo = 1, seg_length[i - 1]
                        
                x, _ = stat._without_paddings(lo)
                assert x.shape[0] != 0, 'should not be'
                _max = np.max(x, 0)
                x = x[x[:, 0] > 0]
                _min = np.min(x, 0)
                invalid = _min < xmin
                xmin = np.where(invalid, _min, xmin)
                invalid = _max > xmax
                xmax = np.where(invalid, _max, xmax)
                if lo[1] == 1:
                    break

            self._min_val = xmin
            self._max_val = xmax

            self._stats  = stats
            self._word   = token
            self._phrase = phrase
            self._xy_dim = None
            self._cache  = {}

        def histo_max(self, with_padding):
            if with_padding:
                return self._max_all[0]
            return self._max_val[0]

        def scatter_min_max(self, with_nil, with_padding):
            if with_nil:
                _min = self._min_all[self._xy_dim]
            else:
                _min = self._min_val[self._xy_dim]
            if with_padding:
                _max = self._max_all[self._xy_dim]
            else:
                _max = self._max_val[self._xy_dim]
            return _min, _max

        def make(self, show_paddings, *xy_dims):
            key = xy_dims, show_paddings
            self._xy_dim = xy_dims = np.asarray(xy_dims)
            if key in self._cache:
                self._word, self._phrase = self._cache[key]
                return

            _max = self._max_all if show_paddings else self._max_val
            sature_max = 0
            for stat in self._stats :
                if stat is None:
                    continue
                sature_max = max(sature_max, stat.colorify(_max[0], xy_dims))

            if self._word:
                self._word = self._word.seal(sature_max)
            self._phrase = tuple(s if s is None else s.seal(sature_max) for s in self._phrase)
            self._cache[key] = self._word, self._phrase

        @property
        def token(self):
            if self._word:
                return self._word
            return self._phrase[0]

        @property
        def tag(self):
            return self._phrase[0]

        @property
        def phrase(self):
            return self._phrase

    def disp(x):
        if x >= 1:
            return str(x)[:3]
        if x <= -1:
            return str(x)[:4]
        if x >= 0.01:
            return str(x)[1:4]
        if x <= -0.01:
            return '-' + str(x)[2:5]
        if x == 0:
            return '0'
        return '+0' if x > 0 else '-0'

    def make_color(mutable_x,
                   show_color = False,
                   inverse_brightness = False,
                   curve = None,
                   fallback_color = 'orange'):
        if show_color:
            v, h, s = mutable_x[:3]
        else:
            if isinstance(mutable_x, float):
                v = mutable_x
            else:
                v = mutable_x[0] if mutable_x.shape else mutable_x
            h = s = 0
        if curve is not None:
            v = curve(v)
        if v < 0 or v > 1:
            return fallback_color
        if inverse_brightness:
            v = 1 - v
        def channel(c):
            x = hex(int(c * 255))
            n = len(x)
            if c < 0:
                return 'z'
            if n == 3:
                return '0' + x[-1]
            elif n == 4:
                return x[2:]
        return '#' + ''.join(channel(x) for x in hsv_to_rgb(h, s, v))

    def make_histogram(stat_board, offx, offy, width, height,
                       stat, offset_length, histo_max,
                       half_word_height,
                       stat_font,
                       stat_color,
                       xlab,
                       filtered = None,
                       distance = None,
                       bin_width = 1):
        gaussian = distance and distance > 0
        xy, xmin, xmax, coord_x = stat.histo_data(width, gaussian, distance if gaussian else bin_width, histo_max, offset_length, filtered)
        if half_word_height and stat_font and xlab:
            stat_board.create_text(offx, offy + height + half_word_height,
                                   text = disp(xmin),
                                   fill = stat_color, anchor = W, font = stat_font)
            stat_board.create_text(offx + width, offy + height + half_word_height,
                                   text = disp(xmax),
                                   fill = stat_color, anchor = E, font = stat_font)
        # base line
        stat_board.create_line(offx,         offy + height,
                               offx + width, offy + height,
                               fill = stat_color)# does not work, width = 0.5)
        
        
        for x, y in xy:
            stat_board.create_line(offx + x, offy + height - y * height,
                                   offx + x, offy + height,
                                   fill  = stat_color, 
                                   width = 1 if gaussian else bin_width)
        return {c:x * width for c, x in coord_x}

    def make_scatter(stat_board, offx, offy, width, height, r,
                     stat, offset_length, scatter_min_max,
                     stat_color,
                     half_word_height,
                     stat_font,
                     to_color,
                     background,
                     xlab, ylab, clab,
                     filtered = None):
        # globals().update({k:v for k,v in zip(BoardConfig._fields, config)})
        # stat_board.create_line(offx, offy, offx, offy + height, fill = stat_color)
        # stat_board.create_line(offx, offy, offx + width,  offy, fill = stat_color)
        # stat_board.create_line(offx, offy + height, offx + width, offy + height, fill = stat_color)
        # stat_board.create_line(offx + width,  offy, offx + width, offy + height, fill = stat_color)
        # if distance and distance > 0:
        #     for i in range(width):
        #         for j in range(int(height)):
        #             x = i / width
        #             y = j / height
        #             z = sum(exp(-((x - xj) ** 2 )/ (2*distance**2)-((y - yj) ** 2 )/ (2*distance**2)) for xj, yj in xy) / len(xy)
        #             stat_board.create_line(offx + i,     offy + height - j,
        #                                    offx + i + 1, offy + height - j, fill = make_color(z))

        xy, xy_min, xy_max, xy_orig, coord = stat.scatter_data(scatter_min_max, offset_length, filtered)
            
        if half_word_height and stat_font and (xlab or ylab or clab):
            xmin, ymin = (disp(i) for i in xy_min)
            xmax, ymax = (disp(i) for i in xy_max)
            if xlab:
                stat_board.create_text(offx, offy + height + half_word_height,
                                        text = xmin, fill = stat_color,
                                        font = stat_font, anchor = W)
                stat_board.create_text(offx + width, offy + height + half_word_height,
                                        text = xmax, fill = stat_color,
                                        font = stat_font, anchor = E)
            if ylab:
                stat_board.create_text(offx - half_word_height * 1.3, offy,
                                        text = ymax, fill = stat_color,
                                        font = stat_font)#, angle = 90)
                stat_board.create_text(offx - half_word_height * 1.3, offy + height,
                                        text = ymin, fill = stat_color,
                                        font = stat_font)
            if clab:
                stat_board.create_text(offx - half_word_height * 1.3, offy + height / 2,
                                        text = f'{len(xy)}•', fill = 'SpringGreen3',
                                        font = stat_font)#, anchor = E)

        stat_board.create_rectangle(offx,         offy,
                                    offx + width, offy + height,
                                    fill = background, outline = background)
        x, y = xy_orig
        lx = offx + x * width
        ly = offy + height - y * height
        stat_board.create_line(lx - 4, ly,
                               lx + 4, ly, fill = 'red')
        stat_board.create_line(lx, ly - 4,
                               lx, ly + 4, fill = 'red')

        scatter_coord_item = {}
        for c, (x, y) in zip(coord, xy):
            x *= width
            y *= height
            item = stat_board.create_oval(offx + x - r, offy + height - y - r,
                                          offx + x + r, offy + height - y + r,
                                          fill = to_color(stat[c]), outline = '')
            scatter_coord_item[c] = item
        return scatter_coord_item

    # NumbList: 'word_width, word_height, yx_ratio, histo_width, scatter_width'
    #  effect:      b             s,b   <-   s,b        s               s
    # dark_background, inverse_brightness, delta_shape, show_errors, statistics, show_paddings, show_nil, align_coord, show_color
    #        sb                 b              sb            b              s?              sb            s           s            b
    fields = ', num_word, half_word_width, half_word_height, line_dx, line_dy, deco_dx, deco_dy, upper_padding, canvas_width, canvas_height, stat_font, stat_pad_left, stat_pad_between, stat_pad_right'
    FrameGeometry = namedtuple('FrameGeometry', ','.join(NumbList._fields) + fields)
    BoardConfig = namedtuple('BoardConfig', DynamicSettings._fields + FrameGeometry._fields)

    class ThreadViewer(Frame):
        def __init__(self,
                     master,
                     vocab_bundle,
                     time_change_callback):
            super().__init__(master)
            self._time_change_callback = time_change_callback
            self._boards = Canvas(self), Canvas(self)
            self._time_slider = None, Scale(self, command = self._change_time)
            self._vocab_bundle = vocab_bundle
            self._spotlight_subjects = None
            
        def configure(self,
                      num_word,
                      num_time,
                      dynamic_settings,
                      static_settings,
                      init_time):
            # calculate canvas
            half_word_width  = static_settings.word_width >> 1
            half_word_height = static_settings.word_height >> 1
            line_dx = half_word_width - static_settings.word_height / static_settings.yx_ratio
            line_dy = line_dx * static_settings.yx_ratio
            deco_dx = static_settings.line_width / np.sqrt(1 + static_settings.yx_ratio ** 2) / 2
            deco_dy = deco_dx * static_settings.yx_ratio
            upper_padding = max(half_word_height, static_settings.line_width / 2)
            canvas_width  = num_word * static_settings.word_width
            canvas_height = (num_word + 2) * (static_settings.word_height + line_dy) + upper_padding # +2 for token and tag layer
            stat_paddings = (28, 10, 22)
            bcfg = num_word, half_word_width, half_word_height, line_dx, line_dy, deco_dx, deco_dy, upper_padding, canvas_width, canvas_height, ('helvetica', 10)
            self._frame_geometry = FrameGeometry(*(static_settings + bcfg + stat_paddings))
            self._conf = BoardConfig(*(dynamic_settings + self._frame_geometry))
            self._last_show_paddings = dynamic_settings.show_paddings

            # resize canvas
            board, stat_board = self._boards
            board.config(width  = canvas_width,
                         height = canvas_height,
                         bd = 0, highlightthickness = 0, # cancel white label_line_bo
                         cursor = 'fleur',
                         scrollregion = '0 0 %d %d' % (canvas_width, canvas_height))

            stat_width = sum(stat_paddings) + static_settings.histo_width + static_settings.scatter_width
            stat_board.config(width  = stat_width,
                              height = canvas_height,
                              bd = 0, highlightthickness = 0,
                              scrollregion = '0 0 %d %d' % (stat_width, canvas_height))

            def scroll_start(event):
                board.scan_mark(event.x, event.y)
                stat_board.scan_mark( 0, event.y)
                self.focus()
            def scroll_move(event):
                board.scan_dragto(event.x, event.y, gain = 1)
                stat_board.scan_dragto( 0, event.y, gain = 1)
            board.bind("<ButtonPress-1>", scroll_start)
            board.bind("<B1-Motion>",     scroll_move)
            # def scroll_delta(event):
            #     print(event)
                # board.xview_scroll(-1 * int(event.delta / 60), "units")
            # board.bind("<MouseWheel>", scroll_delta)
            def moved(event):
                x = board.canvasx(event.x)
                y = board.canvasy(event.y)
                self.spotlight(x, y)
                # print(x, y, board.find_closest(x, y))
            board.bind("<Motion>", moved)

            last_num_time, time_slider = self._time_slider
            time_slider.config(to = num_time - 1, tickinterval = 1, showvalue = False)
            time_slider.set(init_time)
            self._time_slider = num_time, time_slider
            self._animation = []

            # initial view position
            if self._conf.delta_shape:
                self._boards[0].yview_moveto(1)
                self._boards[1].yview_moveto(1)
            else:
                self._boards[0].yview_moveto(0)
                self._boards[1].yview_moveto(0)
            self._refresh_layout()
            # hscroll = Scrollbar(self, orient = HORIZONTAL)
            # vscroll = Scrollbar(self, orient = VERTICAL)
            # self._hv_scrolls = None, vscroll
            # hscroll.config(command = board.xview)
            # vscroll.config(command = board.yview)
            # hscroll.pack(fill = X, side = BOTTOM)
            # vscroll.pack(fill = Y, side = RIGHT)
            # xscrollcommand = self._hv_scrolls[0].set,
            # yscrollcommand = self._hv_scrolls[1].set,
            # need support for 2-finger gesture
            # board.bind("<Button-6>", scroll_start)
            # board.bind("<Button-7>", scroll_move)

        # @property funny mistake
        def ready(self, num_word = None, num_time = None):
            last_num_time = self._time_slider[0]
            if last_num_time is None:
                return False
            elif num_time is not None and last_num_time != num_time:
                return False
            if num_word is not None and self._conf.num_word != num_word:
                return False
            return True

        @property
        def time(self):
            if self._time_slider[0] is None:
                return 0
            return self._time_slider[1].get()

        def set_framework(self, head, time_data):
            if len(time_data) != self._time_slider[0]:
                raise ValueError(f'Set Invalid timesteps! {len(time_data)} vs. {self._time_slider[0]}')
            self._head_time = head, tuple(time_data.items()) # [(fname, ((sents, summary), (sents, summary), ...)), ..]

        def show_sentence(self, sid): # across timesteps
            head, time_data = self._head_time

            # head in [sentence]: IOHead
            self._head = _head = head[sid]

            # data in [epoch, sentence]: IOData
            self._data = []
            for _, (batch, _) in time_data:
                self._data.append(batch[sid])

            self._refresh_board(self._time_slider[1].get())
            if not self._conf.show_paddings:
                offset = _head.offset if 'offset' in _head._fields else 1
                self._boards[0].xview_moveto(offset / self._conf.num_word)

        def minor_change(self, dynamic_settings, dynamic_geometry):
            self._conf = BoardConfig(*(dynamic_settings + dynamic_geometry + self._frame_geometry[len(dynamic_geometry):]))
            if dynamic_settings.statistics ^ self._last_show_paddings:
                self._refresh_layout()
                self._last_show_paddings = dynamic_settings.statistics
            self._refresh_board(self._time_slider[1].get())

        def _change_time(self, time):
            self._refresh_board(int(time))

        def _refresh_layout(self):
            # dynamic layout
            for w in (self._time_slider + self._boards)[1:]:
                w.pack_forget()
            if self._time_slider[0] > 1:
                self._time_slider[1].pack(fill = Y, side = LEFT)
            if self._conf.statistics:
                self._boards[1].pack(side = LEFT, expand = YES)
                self._boards[0].pack(side = LEFT, expand = YES)
            else:
                self._boards[0].pack(side = LEFT, expand = YES)

        def _refresh_board(self, tid):
            board, stat_board = self._boards
            board.delete('elems')
            stat_board.delete(ALL)

            fg_color = 'black'
            bg_color = 'white'
            stat_fg = 'gray5'
            stat_bg = 'gray94'
            x_fg = 'gray80'
            if self._conf.dark_background:
                fg_color, bg_color = bg_color, fg_color
                stat_fg = x_fg = 'gray40'
                stat_bg = 'gray10'
            to_color = partial(make_color,
                               show_color         = self._conf.show_color and not self._conf.apply_dash,
                               inverse_brightness = self._conf.inverse_brightness,
                               curve              = self._conf.curve if self._conf.apply_curve else None) #self._curve if apply_curve else
            for b in self._boards:
                b.configure(background = bg_color)

            _, time = self._head_time
            bid, epoch = time[tid][0][5:-4].split('_')
            title = f'Batch: {bid} Epoch: {epoch} '

            data, stat = self._data[tid]
            stat.make(self._conf.show_paddings, self._conf.pc_x, self._conf.pc_y)
            if data.scores is None:
                board_item_coord, level_unit = self.__draw_board(data, stat, fg_color, to_color)
            else:
                len_score = len(data.scores)
                is_continuous = 'offset' in data._fields
                if is_continuous:
                    board_item_coord, level_unit = self.__draw_board(data, stat, fg_color, to_color)

                    if len_score == 12: # conti. parsing
                        title += ' |  ' + '  '.join(i+f'({data.scores[j]})' for i, j in zip(('len.', 'P.', 'R.', 'tag.'), (1, 3, 4, 11)))
                    elif len_score == 6: # sentiment
                        title += ' |  ' + '  '.join(i+f'({j:.1f})' for i, j in zip(('5r.', '3r.', '2r.', '5f.', '3f.', '2f.'), data.scores))
                else:
                    tm, tp, tg, dm, dp, dg, mt, gt = data.scores
                    title += f' |  ({tp}>{tm}<{tg}) ({dp}>{dm}<{dg}) ({mt}<{gt})'
                    board_item_coord, level_unit = self.__draw_board_x(data, stat, fg_color, x_fg, to_color)

            if self._spotlight_subjects:
                self._spotlight_subjects = (board_item_coord, self._conf.word_width, level_unit) + self._spotlight_subjects[-3:]
            else:
                self._spotlight_subjects = board_item_coord, self._conf.word_width, level_unit, None, None, None

            if self._conf.statistics:
                if is_continuous:
                    offset = data.offset
                    length = data.length
                else:
                    offset = 1
                    length = data.seg_length
                self.__draw_stat_board(data.label, offset, length, stat, stat_fg, stat_bg, to_color)
            
            self._time_change_callback(title, epoch) # [(fname, data)]

        def navi_to(self, navi_char, steps = 0, num_frame = 24, duration = 1):
            self.clear_animation()
            board, stat_board = self._boards
                
            num_word = self._conf.num_word
            show_paddings = self._conf.show_paddings
            head = self._head

            xpad = board.xview() # %
            ypad = board.yview() # %
            ox, oy = xpad[0], ypad[0]
            xpad = xpad[1] - xpad[0]
            ypad = ypad[1] - ypad[0]
            if 'offset' in head._fields:
                l_pad = (head.offset / num_word)
                ratio = 1 - (num_word - head.length) / num_word - l_pad
            else:
                l_pad = 1 / num_word
                ratio = 1 - (num_word - head.seg_length[0] - 2) / num_word - l_pad # -2?
            
            r_pad = ratio - xpad
            c_pad = 0.5 - xpad / 2 if show_paddings else (l_pad + r_pad) / 2
            o_pad = ratio / 2

            print(f'current view ({ox * 100:.0f}, {oy * 100:.0f}, {xpad * 100:.0f}, {ypad * 100:.0f})')
            print(f'sent l_pad:{l_pad * 100:.0f} ratio: {ratio * 100:.0f}, r_pad: {r_pad * 100:.0f}')

            if self._conf.delta_shape:
                navi_left   = (0        if show_paddings else l_pad, 1 - ypad)
                navi_right  = (1 - xpad if show_paddings else r_pad, 1 - ypad)
                navi_top    = (c_pad, 0 if show_paddings else 1 - ratio)
                navi_center = (c_pad, 0.5 - ypad / 2 if show_paddings else 1 - o_pad - ypad / 2)
                navi_bottom = (c_pad, 1 - ypad)
            else:
                navi_left   = (0        if show_paddings else l_pad, 0)
                navi_right  = (1 - xpad if show_paddings else r_pad, 0)
                navi_top    = (c_pad, 0)
                navi_center = (c_pad, 0.5 - ypad / 2 if show_paddings else o_pad - ypad / 2)
                navi_bottom = (c_pad, 1 - ypad if show_paddings else ratio - ypad)

            if steps > 0:
                clip = lambda l,x,u: l if x < l else (u if x > u else x)
                if navi_char in '⇤⇥':
                    dx = steps * (self._conf.word_width)
                    if navi_char == '⇤':
                        dx = -dx
                    dx /= self._conf.canvas_width # %
                    dy = 0
                elif navi_char in '↑↓':
                    dx = 0
                    dy = steps * (self._conf.word_height + self._conf.line_dy)
                    if navi_char == '↑':
                        dy = -dy
                    dy /= self._conf.canvas_height # %
                nx = clip(navi_left[0], ox + dx, navi_right [0])
                ny = clip(navi_top [1], oy + dy, navi_bottom[1])
            else:
                nx, ny = {'⇤': navi_left, '⇥': navi_right, '↑': navi_top, '↓': navi_bottom, 'o': navi_center}[navi_char]

            if nx == ox and ny == oy:
                return
            print(f'next view ({nx * 100:.0f}, {ny * 100:.0f})')
                
            def smoothen(ox, nx):
                x = np.linspace(ox, nx, num_frame)
                if ox == nx:
                    return x
                x_mean = (ox + nx) * 0.5
                x_diff = (nx - ox) * 0.5
                x -= x_mean
                x /= x_diff
                np.sin(x * np.pi / 2, out = x)
                x *= x_diff
                x += x_mean
                return x
                
            i = int(duration / num_frame * 1000)
            j = 0
            def move_anime(xi, yi, l):
                board.xview_moveto(xi)
                board.yview_moveto(yi)
                stat_board.yview_moveto(yi)
                l.pop(0)
            for xi, yi in zip(smoothen(ox, nx), smoothen(oy, ny)):
                self._animation.append(board.after(j, move_anime, xi, yi, self._animation))
                j += i

        def clear_animation(self):
            while self._animation:
                a = self._animation.pop()
                self._boards[0].after_cancel(a)
                self._boards[1].after_cancel(a)

        def save(self, fname):
            if self._conf.show_paddings:
                x = 0
                y = 0
                w = self._conf.canvas_width
                h = self._conf.canvas_height
            else:
                h = self._head
                if 'offset' in h._fields:
                    x = self._conf.word_width * h.offset
                    w = self._conf.word_width * h.length

                    n = self._conf.num_word - h.length # for token and tag layers
                else:
                    x = self._conf.word_width
                    w = self._conf.word_width * h.seg_length[0]

                    n = self._conf.num_word - h.seg_length[0] # for token and tag layers

                l = self._conf.word_height + self._conf.line_dy
                if self._conf.delta_shape:
                    y = n * l
                    h = self._conf.canvas_height - y
                else:
                    y = 0
                    h = self._conf.canvas_height - n * l
            
            self._boards[0].postscript(file = fname, x = x, y = y, width  = w, height = h,
                                       colormode = 'color' if self._conf.show_color else 'gray')

        def spotlight(self, x, y):
            board, stat_board = self._boards
            board_item_coord, w, h, last_picker, last_spotlight, last_bbox = self._spotlight_subjects

            if self._conf.apply_spotlight:
                radius = self._conf.spotlight
                if last_spotlight:
                    nx, ny, _, _ = board.bbox(last_spotlight)
                    board.move(last_spotlight, x - radius - nx, y - radius - ny)
                else:
                    light = make_color(self._conf.picker / 2, inverse_brightness = not self._conf.dark_background)
                    last_spotlight = board.create_oval(x - radius, y - radius, x + radius, y + radius, fill = light, outline = '')
                    self._spotlight_subjects = board_item_coord, w, h, last_picker, last_spotlight, last_bbox
                board.tag_lower(last_spotlight)
            elif last_spotlight:
                board.delete(last_spotlight)
                last_spotlight = None

            if not self._conf.apply_picker:
                board.delete(last_picker)
                self._spotlight_subjects = board_item_coord, w, h, None, last_spotlight, last_bbox
                return

            bbox = None
            for bbox in board_item_coord:
                l, t = bbox
                if l < x < l + w and t < y < t + h:
                    if bbox == last_bbox:
                        # print('avoid repainting')
                        return
                    elif last_picker is not None:
                        # print('repaint')
                        board.delete(last_picker)
                    light = make_color(self._conf.picker, inverse_brightness = not self._conf.dark_background)
                    act = board.create_rectangle(*bbox, l + w, t + h, fill = light, outline = '')
                    board.tag_lower(act)
                    _, coord = board_item_coord[bbox]
                    # for item in board_item:
                    #     board.tag_raise(item)
                    self._spotlight_subjects = board_item_coord, w, h, act, last_spotlight, bbox
                    if self._conf.statistics:
                        items, positions, reflection = self._spotlight_objects
                        for i in reflection:
                            stat_board.delete(i)
                        x,y,x_,y_ = stat_board.bbox(items[coord])
                        stat_ref = stat_board.create_oval(x-1,y-1,x_+1,y_+1, fill = 'red', outline = '')
                        x,y,h = positions[coord]
                        histo_ref = stat_board.create_line(x,y,x,y+h, fill = 'red')
                        self._spotlight_objects = items, positions, (stat_ref, histo_ref)
                    break
                else:
                    bbox = None
            if bbox is None and last_picker is not None:
                # print('remove spotlight when nothing found')
                board.delete(last_picker)
                self._spotlight_subjects = self._spotlight_subjects[:-3] + (None, last_spotlight, None)
                if self._conf.statistics:
                    items, positions, reflection = self._spotlight_objects
                    for i in reflection:
                        stat_board.delete(i)
                    self._spotlight_objects = items, positions, tuple()

        def show_tree(self, show_all_trees, *frame_canvas):
            if not isinstance(self._data[0][0].tree, str):
                vocabs = self._vocab_bundle.vocabs
                def get_lines(fields, stamp):
                    length = fields.seg_length[0]
                    words = fields.token[1:length + 1]
                    tags = fields.tag[1:length + 1]
                    bottom = tuple((bid, vocabs.token[wid], vocabs.tag[tid]) for bid, (wid, tid) in enumerate(zip(words, tags)))
                    return draw_str_lines(bottom, fields.tree, attachment = stamp)
                lines = get_lines(self._head, ' (gold)')
                if show_all_trees:
                    for tid in range(self._time_slider[0]):
                        data, _ = self._data[tid]
                        lines.append('')
                        lines.extend(get_lines(data, f' (predict-{tid})'))
                else:
                    tid = self._time_slider[1].get()
                    data, _ = self._data[tid]
                    lines.append('')
                    lines.extend(get_lines(data, f' (predict-{tid})'))
                widget = Text(Toplevel(self), wrap = NONE, font = 'TkFixedFont')
                widget.insert(END, '\n'.join(lines))
                # widget.config(state = DISABLED)
                widget.pack(fill = BOTH, expand = YES)
                return

            if frame_canvas:
                frame, canvas = frame_canvas
            else:
                frame  = CanvasFrame()
                canvas = frame.canvas()

            if self._head.tree:
                label = Tree.fromstring(self._head.tree)
                label.set_label(label.label() + ' (corpus)')
                label = TreeWidget(canvas, label, draggable = 1) # bbox is not sure
                label.bind_click_trees(label.toggle_collapsed)
                frame.add_widget(label)
                below = label.bbox()[3] + pad
            else:
                below = 0
            pad = 20
            inc = 0
            right_bound = canvas.winfo_screenwidth() # pixel w.r.t hidpi (2x on this mac)

            def at_time(pred, left_top, force_place = False):
                pred = TreeWidget(canvas, pred,  draggable = 1)
                pred_wh = pred.bbox()
                pred_wh = pred_wh[2] - pred_wh[0], pred_wh[3] - pred_wh[1]
                if not force_place and pred_wh[0] + left_top[0] > right_bound:
                    left_top = 0, below + inc
                pred.bind_click_trees(pred.toggle_collapsed)
                frame.add_widget(pred, *left_top)
                return left_top + pred_wh

            left_top = (0, below)
            if show_all_trees:
                tids = []
                trees = []
                for tid in range(self._time_slider[0]):
                    data, _ = self._data[tid]
                    tree = Tree.fromstring(data.tree)
                    if tree in trees:
                        tids[trees.index(tree)].append(tid)
                    else:
                        trees.append(tree)
                        tids.append([tid])
                for ts, pred in zip(tids, trees):
                    pred.set_label(pred.label() + ' (predict-%s)' % ','.join(str(tid) for tid in ts))
                    ltwh = at_time(pred, left_top, force_place = left_top[0] == 0)
                    inc = max(inc, ltwh[3])
                    if ltwh[0] != left_top[0]: # new line
                        below += inc + pad
                    left_top = (ltwh[0] + ltwh[2] + pad, below)
            else:
                tid = self._time_slider[1].get()
                data, _ = self._data[tid]
                pred = Tree.fromstring(data.tree)
                pred.set_label(pred.label() + f' (predict-{tid})')
                at_time(pred, left_top, force_place = True)

        def __draw_stat_board(self, label_layers, offset, length, stat, fg_color, bg_color, to_color):
            stat_board = self._boards[1]
            vocabs = self._vocab_bundle.vocabs
            nil = vocabs.label.index(NIL) if vocabs.tag else -1
            scatter_coord_item = {}
            histo_coord_position = {}
            # half_word_height, word_height, line_dy, delta_shape, canvas_height, show_paddings, show_nil   | histo_width | scatter_width | -> stat_width (pad_left, )

            line_dy = self._conf.line_dy
            incre_y = line_dy + self._conf.word_height
            if self._conf.delta_shape:
                offy = self._conf.canvas_height - incre_y
                incre_y = -incre_y
                bottom_offy = offy + incre_y
            else:
                bottom_offy = offy = self._conf.upper_padding

            scatter_width = self._conf.scatter_width
            histo_width   = self._conf.histo_width
            bottom_height = 2 * line_dy + self._conf.word_height
            pad_left      = self._conf.stat_pad_left
            histo_offset = pad_left + scatter_width + self._conf.stat_pad_between
            def level_tag(offy, tag):
                if self._conf.delta_shape:
                    offy = offy - incre_y - self._conf.half_word_height
                else:
                    offy += self._conf.half_word_height - self._conf.upper_padding
                stat_board.create_text(histo_offset + histo_width + self._conf.stat_pad_right, offy, text = tag, fill = 'deep sky blue', anchor = E)

            _scatter = partial(make_scatter,
                               stat_board       = stat_board,
                               offx             = pad_left,
                               width            = scatter_width,
                               r                = self._conf.line_width / 2,
                               scatter_min_max  = stat.scatter_min_max(self._conf.show_nil, self._conf.show_paddings) if self._conf.align_coord else None,
                               stat_color       = fg_color,
                               stat_font        = self._conf.stat_font,
                               half_word_height = self._conf.half_word_height,
                               background       = bg_color,
                               to_color         = to_color)
            
            _histo = partial(make_histogram,
                             stat_board       = stat_board,
                             offx             = histo_offset,
                             width            = histo_width,
                             histo_max        = stat.histo_max(self._conf.show_paddings) if self._conf.align_coord else None,
                             stat_color       = fg_color,
                             stat_font        = self._conf.stat_font,
                             half_word_height = self._conf.half_word_height,
                             distance  = self._conf.gauss if self._conf.apply_gauss else None,
                             bin_width = self._conf.gauss * histo_width)
            bottom_offset_length = None if self._conf.show_paddings else (offset, length[0] if length.shape else length)
            level_tag(offy, tag = 'W')

            if self._conf.align_coord:
                height = line_dy + self._conf.word_height * 0.75
                xlab = False
            else:
                height = line_dy
                xlab = True
            sci = _scatter(offy = bottom_offy, stat = stat.token, offset_length = bottom_offset_length, height = bottom_height, xlab = True, ylab = True, clab = not xlab).items()
            hcp = _histo  (offy = bottom_offy, stat = stat.token, offset_length = bottom_offset_length, height = bottom_height, xlab = True).items()
            for (i, it), (j, ip) in zip(sci, hcp):
                ip = ip + histo_offset, bottom_offy, bottom_height
                scatter_coord_item  [('w', i)] = it
                histo_coord_position[('w', j)] = ip
            offy += incre_y * 2 # skip .tag
            nega = 0
            for l, (plabel_layer, layer_phrase_energy) in enumerate(zip(label_layers, stat.phrase)):
                if layer_phrase_energy is None:
                    offy += incre_y
                    nega += 1
                    continue
                if self._conf.show_paddings or length[l] if length.shape else (l < length): # watch out for not showing and len <= 2
                    cond_level_len  = None if self._conf.show_paddings else (offset, length[l] + offset if length.shape else (length - l))
                    cond_nil_filter = None if self._conf.show_nil else plabel_layer > nil # interesting! tuple is not a good filter here, list is proper!
                    level_tag(offy, str(l - nega))
                    sci = _scatter(offy = offy, stat = layer_phrase_energy, offset_length = cond_level_len, filtered = cond_nil_filter, height = height, xlab = xlab, ylab = xlab, clab = not xlab).items()
                    hcp = _histo  (offy = offy, stat = layer_phrase_energy, offset_length = cond_level_len, filtered = cond_nil_filter, height = height, xlab = xlab).items()
                    for (i, it), (j, ip) in zip(sci, hcp):
                        scatter_coord_item  [(l, j)] = it
                        histo_coord_position[(l, i)] = ip + histo_offset, offy, height
                offy += incre_y
            self._spotlight_objects = scatter_coord_item, histo_coord_position, tuple()

        def __draw_board(self, data, stat, fg_color, to_color):
            head   = self._head
            board  = self._boards[0]
            vocabs = self._vocab_bundle.vocabs
            board_item_coord = {}
            apply_dash = self._conf.apply_dash
            tail_ = 1 - self._conf.dash
            tail_ = 1 / tail_

            line_dx = self._conf.line_dx
            line_dy = -self._conf.line_dy
            # line_xy = self._conf.line_dx / self._conf.line_dy
            line_ldx = self._conf.word_height / self._conf.yx_ratio
            word_center     = self._conf.half_word_height + self._conf.offset_y    # >--- token ---<
            level_unit      = self._conf.word_height + self._conf.line_dy            # token lines
            tag_label_center  = word_center + level_unit                        # >--- tag label ---<
            tag_label_line_bo = 2 * level_unit            + self._conf.offset_y # token lines tag lines
            line_width = self._conf.line_width
            r = line_width // 2
            text_offy = 0 #2
            w_p_s = self._conf.word_height + self._conf.offset_y, level_unit + self._conf.offset_y, level_unit + self._conf.word_height + self._conf.offset_y # token >--> tag >--> label
            decorate = isinstance(line_width, int) and line_width % 2 == 0
            capstyle = ROUND if not apply_dash and decorate else None
            deco_dx = 0 if decorate else self._conf.deco_dx
            deco_dy = 0 if decorate else self._conf.deco_dy
            font_name, font_min_size, font_max_size = self._conf.font
            round_int = lambda x: int(round(x))
            errors = []
            if self._conf.delta_shape:
                line_dy = self._conf.line_dy
                deco_dy = - deco_dy
                word_center        = self._conf.canvas_height - word_center
                tag_label_center   = self._conf.canvas_height - tag_label_center
                tag_label_line_bo  = self._conf.canvas_height - tag_label_line_bo
                w_p_s = tuple(self._conf.canvas_height - b for b in w_p_s)
                               
            for i, w in enumerate(head.token):
                if not self._conf.show_paddings and not (head.offset <= i < head.offset + head.length):
                    continue

                center_x = (i + 0.5) * self._conf.word_width + self._conf.offset_x
                left_x   = center_x - self._conf.half_word_width
                if self._conf.delta_shape:
                    wbox = (left_x, w_p_s[1]         )
                    pbox = (left_x, tag_label_line_bo)
                else:
                    wbox = (left_x, 0       )
                    pbox = (left_x, w_p_s[1])

                word_color = to_color(1.0 if apply_dash else (stat.token[i] if self._conf.force_bottom_color else stat.tag[i]))
                token = vocabs.token[w]

                if data.tag is None:
                    word_node = board.create_text(center_x, tag_label_center, 
                                                  text = token, font = (font_name, font_max_size),
                                                  fill = word_color, tags = ('elems', 'node'))
                    word_line = board.create_line(center_x,  w_p_s[2],
                                                  center_x,  tag_label_line_bo,
                                                  width = line_width,
                                                  fill = word_color,
                                                  capstyle = capstyle,
                                                  tags = ('elems', 'line'))
                    board_item_coord[pbox] = (word_node, word_line), ('p', i)
                else:
                    word_node = board.create_text(center_x, word_center + text_offy,
                                                  text = token, font = (font_name, font_max_size),
                                                  fill = word_color, tags = ('elems', 'node'))
                    word_line = board.create_line(center_x,  w_p_s[0],
                                                  center_x,  w_p_s[1],
                                                  width = line_width,
                                                  fill = word_color,
                                                  capstyle = capstyle,
                                                  tags = ('elems', 'line'))
                    board_item_coord[wbox] = (word_node, word_line), ('w', i)
                    tp = head.tag[i]
                    pp = data.tag[i]
                    # print(len(stat.tag), len(stat.token), len(stat.phrase[0])) shorter??
                    tag_color = to_color(data.tag_score if apply_dash else stat.tag[i] if i < len(stat.tag) else (0,0,0))
                    tag_node = board.create_text(center_x, tag_label_center + text_offy,
                                                 fill = tag_color, font = (font_name, font_max_size if apply_dash else round_int(font_max_size * data.tag_score[i])),
                                                 text = f'{vocabs.tag[pp]}' if not self._conf.show_errors or pp == tp else f'{vocabs.tag[pp]}({vocabs.tag[tp]})',
                                                 tags = ('elems', 'node'))
                    tag_line = board.create_line(center_x,  w_p_s[2],
                                                 center_x,  tag_label_line_bo,
                                                 width = line_width,
                                                 fill = tag_color,
                                                 capstyle = capstyle,
                                                 tags = ('elems', 'line'))
                    if pp != tp and self._conf.show_errors:
                        err = board.create_rectangle(*board.bbox(tag_node),
                                                     outline = 'red',
                                                     dash = (1, 2),
                                                     tags = ('elems', 'err'))
                        elems = tag_node, tag_line, err
                        errors.append(err)
                    else:
                        elems = tag_node, tag_line
                    board_item_coord[pbox] = elems, (0, i)

            if data.tag is None:
                nil = None
            elif vocabs.tag: # in both meanning: label and pol cate
                nil = vocabs.label.index(NIL)
            else:
                nil = -1
            layer_tracker = None

            for l, (label_layer, right_layer) in enumerate(zip(data.label, data.right)): # no way to show head.right ?
                last_line_bo = tag_label_line_bo
                if self._conf.delta_shape:
                    tag_label_center = tag_label_line_bo - self._conf.half_word_height
                    line_y           = tag_label_line_bo - self._conf.word_height
                    tag_label_line_bo -= level_unit
                else:
                    tag_label_center = tag_label_line_bo + self._conf.half_word_height
                    line_y           = tag_label_line_bo + self._conf.word_height
                    tag_label_line_bo += level_unit
                # layer_label = layers[1]
                # layer_len   = len(layer_label)
                # layer_len_diff = self._conf.num_word - layer_len
                if label_layer is None:
                    if layer_tracker is None:
                        last_right = data.right[l - 1]
                        last_exist = data.label[l - 1]
                        if np.issubdtype(last_exist.dtype, np.integer): # TODO: unlabeled dtype is float, all Trues
                            last_exist = last_exist[:, 0] > -1 if data.tag is None else last_exist > 0
                        layer_tracker = []
                        last_stat = stat.phrase[l - 1]._global_data
                        for p, (lhr, rhr, lhe, rhe) in enumerate(zip(last_right, last_right[1:], last_exist, last_exist[1:])):
                            rw_relay = lhe and lhr
                            lw_relay = rhe and not rhr
                            if lhr and not rhr:
                                get_itp = lambda res: itp(last_stat[p, 1:], last_stat[p + 1, 1:], res)
                            else:
                                get_itp = None
                            if rw_relay or lw_relay:
                                center_x = ((l + 1)/2 + p) * self._conf.word_width + self._conf.offset_x
                                layer_tracker.append((center_x, line_y, get_itp))
                    continue

                for p, (ps, pr) in enumerate(zip(label_layer, right_layer)):
                    if not self._conf.show_paddings and not (head.offset <= p < head.offset + head.length - l):
                        continue
                    elif not self._conf.show_nil and (ps[0] < 0 if isinstance(ps, np.ndarray) else ps == nil): # not use ts because of spotlight
                        continue
                        
                    center_x = (l/2 + p + .5) * self._conf.word_width + self._conf.offset_x
                    left_x   = center_x - self._conf.half_word_width
                    sbox = (left_x, tag_label_line_bo) if self._conf.delta_shape else (left_x, last_line_bo)
                    mpc_color = to_color(stat.phrase[l][p])
                    if apply_dash:
                        label_color = to_color(data.label_score[l][p])
                    else:
                        label_color = mpc_color

                    if data.tag is None and head.label is not None: # sentiment labels
                        ts = head.label[l][p]
                        elems = []
                        error = None
                        pps = sorted(zip(((vocabs.polar[psi], psi) for psi in ps), data.label_score[l][p]), key = lambda tt: tt[0][0])
                        for pid, ((p_polar, psi), p_score) in enumerate(pps):
                            b_font_size = font_max_size - font_min_size
                            b_font_size = round_int(b_font_size * p_score)
                            b_font_size += font_min_size
                            elem = board.create_text(center_x, tag_label_center + text_offy,
                                                     text = f'{p_polar}',
                                                     fill = label_color,
                                                     tags = ('elems', 'node'),
                                                     font = (font_name, b_font_size,))
                            elems.append(elem)
                            xs, _, xe, _ = board.bbox(elem)
                            if pid:
                                b_widths.append(xe - xs)
                            else:
                                b_widths  = [xe - xs]
                            if self._conf.show_errors and psi == ts and psi != ps[0]:
                                error = pid
                        b_offset = 0
                        mid_pos = sum(b_widths) / 2
                        for pid, (bw, elem) in enumerate(zip(b_widths, elems)):
                            b_offset += bw
                            x_move = b_offset - bw / 2 - mid_pos
                            board.move(elem, x_move, 0)
                            
                            if error == pid:
                                elem = board.create_rectangle(*board.bbox(elem),
                                                              outline = 'red', dash = (1, 2),
                                                              tags = ('elems', 'err'))
                                errors.append(elem)
                                elems .append(elem)
                    else:
                        if issubclass(label_layer.dtype.type, np.integer):
                            if self._conf.show_errors and head.label[l] is not None:
                                ts = head.label[l][p]
                                draw_error_box = ps != ts
                                label_text = f'{vocabs.label[ps]}({vocabs.label[ts]})' if draw_error_box else f'{vocabs.label[ps]}'
                            else:
                                draw_error_box = False
                                label_text = f'{vocabs.label[ps]}'
                        else: # tokenization
                            draw_error_box = False
                            label_text = f'{ps * 100:.1f}%'
                        
                        elems = [board.create_text(center_x, tag_label_center + text_offy,
                                                   text = label_text,
                                                   fill = label_color,
                                                   tags = ('elems', 'node'),
                                                   font = (font_name, font_max_size if apply_dash else round_int(font_max_size * data.label_score[l][p])),)]
                        if draw_error_box:
                            # x,y,x_,y_ = board.bbox(elems[0])x-7,y-3,x_+2,y_, 
                            elems.append(board.create_rectangle(*board.bbox(elems[0]),
                                                                outline = 'red', dash = (1, 2),
                                                                tags = ('elems', 'err')))
                            errors.append(elems[-1])
                    rhs_itp = None
                    if layer_tracker is not None:
                        _, _, get_itp = layer_tracker[p]
                        if callable(get_itp):
                            rhs_itp = get_itp(stat.phrase[l]._global_data[p, 1:])[1].mean()
                    elif l > 0:
                        last_right = data.right[l - 1]
                        last_exist = data.label[l - 1]
                        last_stat = stat.phrase[l - 1]._global_data
                        if np.issubdtype(last_exist.dtype, np.integer): # TODO: unlabeled dtype is float, all Trues
                            last_exist = last_exist[:, 0] > -1 if data.tag is None else last_exist > 0
                        if last_right[p] and not last_right[p + 1] and last_exist[p] and last_exist[p + 1]:
                            rhs_itp = itp(last_stat[p, 1:], last_stat[p + 1, 1:], stat.phrase[l]._global_data[p, 1:])[1].mean()
                    if rhs_itp is not None:
                        elems.append(board.create_text(center_x, tag_label_center + text_offy + self._conf.word_height,
                                                       text = f'{rhs_itp:.2f}',
                                                       fill = label_color,
                                                       tags = ('elems', 'itp'),
                                                       font = (font_name, round_int(font_max_size * 0.8)),))

                    if not self._conf.show_paddings:
                        if l >= head.length - 1:
                            continue

                    if apply_dash:
                        if pr:
                            score = data.split_score[l][p]
                            to_x = center_x + line_dx
                        else:
                            score = 1 - data.split_score[l][p]
                            to_x = center_x - line_dx
                        score -= 0.5
                        score *= 2
                        color = to_color(score)
                        score -= self._conf.dash
                        score *= tail_
                        width = line_width * score
                        if width < 1:
                            width = 1
                        dash_ = None
                        if score < 0:
                            dash_ = int(-tail_ / score)
                            if dash_ > 255:
                                dash_ = 255
                            dash_ = (dash_, 1)

                        elems.append(board.create_line(center_x, line_y, to_x, tag_label_line_bo,
                                                       width = width, fill = color, dash = dash_,
                                                       tags = ('elems', 'line')))
                    else:
                        right_score = data.split_score[l][p]
                        left_score  = 1 - right_score
                        offset_right_x = line_dx * right_score
                        offset_left_x  = line_dx * left_score
                        to_left_x  = center_x - offset_left_x
                        to_right_x = center_x + offset_right_x
                        to_left_y  = line_y - line_dy * left_score
                        to_right_y = line_y - line_dy * right_score
                        elems.append(board.create_line( to_left_x - deco_dx,  to_left_y + deco_dy,
                                                       center_x   - deco_dx,     line_y - deco_dy,
                                                       to_right_x + deco_dx, to_right_y + deco_dy,
                                                       capstyle = capstyle,
                                                       width = line_width, fill = mpc_color,
                                                       tags = ('elems', 'r_line')))
                        # if offset_right_x > 0.5:
                        # if offset_left_x > 0.5:
                        #     elems.append(board.create_line(center_x  + deco_dx,    line_y - deco_dy,
                                                           
                        #                                    width = line_width, fill = mpc_color,
                        #                                    tags = ('elems', 'l_line')))
                        # if decorate:
                        #     if offset_right_x > 0.5:
                        #         elems.append(board.create_oval(to_right_x - r, to_right_y - r,
                        #                                        to_right_x + r, to_right_y + r,
                        #                                        fill = mpc_color, outline = '', tags = ('elems', 'r_dot')))
                        #     if offset_left_x > 0.5:
                        #         elems.append(board.create_oval(to_left_x - r, to_left_y - r,
                        #                                        to_left_x + r, to_left_y + r,
                        #                                        fill = mpc_color, outline = '', tags = ('elems', 'l_dot')))
                        #     elems.append(board.create_oval(center_x - r, line_y - r,
                        #                                    center_x + r, line_y + r,
                        #                                    fill = mpc_color, outline = '', tags = ('elems', 'dot')))
                        color = mpc_color

                    if layer_tracker is not None:
                        last_x, last_y, _ = layer_tracker[p]
                        
                        if last_x == center_x:
                            to_x = center_x
                        elif last_x < center_x:
                            to_x = center_x - line_ldx
                        else:
                            to_x = center_x + line_ldx
                        # else:
                        #     dy = (last_y - last_line_bo) if last_y > last_line_bo else (last_line_bo - last_y)
                        #     if last_x < center_x:
                        #         to_x = center_x - line_ldx * dy / (center_x - last_x) * line_xy
                        #     else:
                        #         to_x = center_x + line_ldx * dy / (last_x - center_x) * line_xy
                        elems.append(board.create_line(last_x, last_y, to_x, last_line_bo,
                                                       width = line_width, fill = color,# dashoffset = r,
                                                       capstyle = capstyle,
                                                       dash = (1, line_width << 1), tags = ('elems', 'c_line')))
                        # if not apply_dash and decorate:
                        #     elems.append(board.create_oval(last_x - r, last_y - r,
                        #                                    last_x + r, last_y + r,
                        #                                    fill = mpc_color, outline = '', tags = ('elems', 'd_dot')))
                        #     elems.append(board.create_oval(to_x - r, last_line_bo - r,
                        #                                    to_x + r, last_line_bo + r,
                        #                                    fill = mpc_color, outline = '', tags = ('elems', 'u_dot')))
                    board_item_coord[sbox] = elems, (l, p)
                layer_tracker = None
                # if layer_len == 1#: or not self._conf.show_paddings and np.any(layer_label):
                #     break
            for elem in errors:
                board.tag_raise(elem)
            return board_item_coord, level_unit

        def __draw_board_x(self, data, stat, fg_color, jnt_color, to_color):
            head   = self._head
            board  = self._boards[0]
            vocabs = self._vocab_bundle.vocabs
            threshold = self._vocab_bundle.threshold
            board_item_coord = {}
            apply_dash = self._conf.apply_dash

            word_center     = self._conf.half_word_height + self._conf.offset_y    # >--- token ---<
            level_unit      = self._conf.word_height + self._conf.line_dy            # token lines
            tag_label_center  = word_center + level_unit                        # >--- tag label ---<
            tag_label_line_bo = 2 * level_unit            + self._conf.offset_y # token lines tag lines
            line_width = self._conf.line_width
            text_offy = 0 #2
            w_p_s = self._conf.word_height + self._conf.offset_y, level_unit + self._conf.offset_y, level_unit + self._conf.word_height + self._conf.offset_y # token >--> tag >--> label
            decorate = isinstance(line_width, int) and line_width % 2 == 0
            deco_dx = 0 if decorate else self._conf.deco_dx
            deco_dy = 0 if decorate else self._conf.deco_dy
            font_name, font_min_size, font_max_size = self._conf.font
            round_int = lambda x: int(round(x))
            errors = []
            if not apply_dash and decorate:
                capstyle = ROUND
                r = line_width // 2
            else:
                capstyle =  None
                r = 0
            if self._conf.delta_shape:
                r = - r
                line_dy = self._conf.line_dy
                deco_dy = - deco_dy
                word_center        = self._conf.canvas_height - word_center
                tag_label_center   = self._conf.canvas_height - tag_label_center
                tag_label_line_bo  = self._conf.canvas_height - tag_label_line_bo
                w_p_s = tuple(self._conf.canvas_height - b for b in w_p_s)
            half_text_height = None
            
            track_colors = []
            track_positions = []
            for i, w in enumerate(head.token):
                if not self._conf.show_paddings and not 0 < i <= head.seg_length[0]:
                    continue

                center_x = (i + 0.5) * self._conf.word_width + self._conf.offset_x
                left_x   = center_x - self._conf.half_word_width
                track_positions.append(center_x)
                if self._conf.delta_shape:
                    wbox = (left_x, w_p_s[1]         )
                    pbox = (left_x, tag_label_line_bo)
                else:
                    wbox = (left_x, 0       )
                    pbox = (left_x, w_p_s[1])

                if self._conf.show_errors or not self._conf.force_bottom_color:
                    track_color = 1.0 if apply_dash else stat.tag[i]
                    word_color = tag_color = to_color(track_color)
                    if self._conf.show_errors:
                        track_colors.append(track_color)
                else: # .7 - x ** 10000 * 0.1
                    word_color = fg_color if apply_dash else to_color(stat.token[i])
                    tag_color  = to_color(data.tag_score if apply_dash else stat.tag[i] if i < len(stat.tag) else (0,0,0))

                word_node = board.create_text(center_x, word_center + text_offy,
                                              text = vocabs.token[w], font = (font_name, font_max_size),
                                              fill = word_color, tags = ('elems', 'node'))
                tp = head.tag[i]
                pp = data.tag[i]
                tag_node = board.create_text(center_x, tag_label_center + text_offy,
                                             fill = tag_color, font = (font_name, font_max_size if apply_dash else round_int(font_max_size * (1 if self._conf.show_errors else data.tag_score[i]))),
                                             text = f'{vocabs.tag[pp]}' if self._conf.show_errors or pp == tp else f'{vocabs.tag[pp]}({vocabs.tag[tp]})',
                                             tags = ('elems', 'node'))

                w_line = board.create_line(center_x,  w_p_s[0] + r,
                                           center_x,  w_p_s[1] - r,
                                           width = line_width,
                                           fill = word_color,
                                           capstyle = capstyle,
                                           tags = ('elems', 'line'))
                t_line = board.create_line(center_x,  w_p_s[2] + r,
                                           center_x,  tag_label_line_bo - r,
                                           width = line_width,
                                           fill = tag_color,
                                           capstyle = capstyle,
                                           tags = ('elems', 'line'))
                board_item_coord[wbox] = (word_node, w_line), ('w', i)

                if pp != tp and self._conf.show_errors:
                    err = board.create_rectangle(*board.bbox(tag_node),
                                                 outline = 'red',
                                                 dash = (1, 2),
                                                 tags = ('elems', 'err'))
                    elems = tag_node, t_line, err
                else:
                    elems = tag_node, t_line
                board_item_coord[pbox] = elems, ('p', i)
                if half_text_height is None:
                    _, ys, _, ye = board.bbox(word_node)
                    half_text_height = (ye - ys) * 0.45

            if self._conf.show_errors:
                layers = enumerate(zip(head.label, head.right, head.direc, head.right, head.direc, head.segment, head.seg_length))
            else:
                layers = enumerate(zip(data.label, data.right, data.direc, data.right_score, data.direc_score, data.segment, data.seg_length))
            for lid, (label_layer, right_layer, direc_layer, right_score, direc_score, seg_size, layer_size) in layers:
                last_line_bo = tag_label_line_bo
                if self._conf.delta_shape:
                    tag_label_center = tag_label_line_bo - self._conf.half_word_height
                    line_y           = tag_label_line_bo - self._conf.word_height
                    tag_label_line_bo -= level_unit
                else:
                    tag_label_center = tag_label_line_bo + self._conf.half_word_height
                    line_y           = tag_label_line_bo + self._conf.word_height
                    tag_label_line_bo += level_unit

                last_x = None
                track_positions.append(None) # layer flag
                if not self._conf.show_errors and lid >= len(data.joint):
                    break
                for pid, (ps, pr, pd, prs, pds) in enumerate(zip(label_layer, right_layer, direc_layer, right_score, direc_score)):
                    if not self._conf.show_paddings and not (0 < pid <= layer_size):
                        last_pr = pr
                        continue
                    center_x = track_positions.pop(0)
                    # print(center_x, end = ', ')
                    if self._conf.show_errors:
                        pls = 1
                        track_color = track_colors.pop(0)
                        mpc_color = label_color = to_color(track_color)
                    else:
                        track_color = 1.0
                        pls = data.label_score[lid][pid]
                        mpc_color = to_color(stat.phrase[lid][pid])
                        if apply_dash:
                            label_color = to_color(pls)
                        else:
                            label_color = mpc_color
                    node = board.create_text(center_x, tag_label_center + text_offy,
                                             text = vocabs.label[ps],
                                             fill = label_color,
                                             tags = ('elems', 'node'),
                                             font = (font_name, font_max_size if apply_dash else round_int(font_max_size * pls)),)
                    to_x = center_x + uneven_split(threshold.right, prs) * self._conf.half_word_width
                    if self._conf.delta_shape:
                        to_y = line_y - self._conf.line_dy * pds
                    else:
                        to_y = line_y + self._conf.line_dy * pds
                    line = board.create_line(center_x, line_y, to_x, to_y,
                                             width = line_width, fill = mpc_color, capstyle = capstyle,
                                             arrow = LAST if pd else None, tags = ('elems', 'line'))
                    is_joint = (head if self._conf.show_errors else data).joint[lid][pid - 1] if pid else False
                    new_center_x = None
                    if pid and last_pr and not pr:
                        if is_joint:
                            last_x = track_positions.pop()
                            new_center_x = (last_x + center_x) / 2
                            track_positions.append(new_center_x)
                            track_colors.append((track_colors.pop() + track_color) / 2)
                        elif self._conf.show_errors and (last_pd or pd):
                            track_colors.insert(-1, track_color)
                            track_positions.append(center_x)
                        else:
                            track_positions.append(center_x)
                    else:
                        track_colors.append(track_color)
                        track_positions.append(center_x)
                    if last_x is None:
                        elems = node, line
                    else:
                        if new_center_x is None:
                            new_center_x = (last_x + center_x) / 2
                        js = uneven_split(threshold.joint, (head.joint if self._conf.show_errors else data.joint_score)[lid][pid - 1])
                        elems = [node, line]
                        if is_joint:
                            fill_color = jnt_color
                            dash = None
                            from_x = last_x + self._conf.half_word_width
                            to_x = center_x - self._conf.half_word_width
                            if from_x < to_x:
                                elems.append(board.create_line(from_x, tag_label_center, to_x, tag_label_center,
                                                               fill = fill_color, dash = (4, 4), tags = ('elems', 'j_line')))
                        else:
                            js = -js
                            fill_color = ''
                            dash = (2, 2)
                        radius = half_text_height * js
                        elems.append(board.create_oval(new_center_x - radius, tag_label_center - radius,
                                                       new_center_x + radius, tag_label_center + radius,
                                                       fill = fill_color, outline = fg_color, dash = dash, tags = ('elems', 'jnt')))
                    left_x = center_x - self._conf.half_word_width
                    sbox = (left_x, tag_label_line_bo) if self._conf.delta_shape else (left_x, last_line_bo)
                    board_item_coord[sbox] = elems, (lid, pid)
                    last_pd = pd
                    last_pr = pr
                    last_x = center_x
                # print(track_positions[0])
                assert track_positions.pop(0) is None, track_positions
                    
                if layer_size == 1:
                    break

            # for elem in errors:
            #     board.tag_raise(elem)
            return board_item_coord, level_unit


    import argparse
    import getpass
    def get_args():
        parser = argparse.ArgumentParser(
            prog = 'Visual', usage = '%(prog)s DIR [options]',
            description = 'A handy viewer for parsing and its joint experiments', add_help = False
        )
        parser.add_argument('dir', metavar = 'DIR', help = 'indicate a local or remote directory', type = str)
        parser.add_argument('-h', '--host',     help = 'remote host',     type = str, default = 'localhost')
        parser.add_argument('-u', '--username', help = 'remote username', type = str, default = getpass.getuser())
        parser.add_argument('-p', '--port',     help = 'remote port, necessary for remote connection or your tunnel, will ask for password', type = int, default = -1)
        args = parser.parse_args()
        if not isinstance(args.dir, str):
            parser.print_help()
            print('Please provide an folder with -d/--dir', file = sys.stderr)
            exit()
        if args.port > 0 or args.host != 'localhost':
            import pysftp
            import paramiko
            cnopts = pysftp.CnOpts()
            cnopts.hostkeys = None
            hostname = args.host
            username = args.username
            port     = args.port
            cfile = join(expanduser('~'), '.ssh', 'config')
            if isfile(cfile):
                user_ssh_config = paramiko.config.SSHConfig()
                with open(cfile) as cfile:
                    user_ssh_config.parse(cfile)
                if args.host in user_ssh_config.get_hostnames():
                    config = user_ssh_config.lookup(args.host)
                    hostname = config['hostname']
                    username = config.get('user', args.username)
                    if port < 0:
                        port = config.get('port', 22)

            password = getpass.getpass('Password for %s:' % username)
            # try:
            sftp = pysftp.Connection(hostname,
                                     username = username,
                                     password = password,
                                     port     = port,
                                     cnopts   = cnopts)
            files = sftp.listdir(args.dir)
            return PathWrapper(args.dir, sftp)
        return PathWrapper(args.dir, None)

    if __name__ == '__main__':
        # root.geometry("300x300+300+300")
        root = Tk()
        app = TreeExplorer(root, get_args())
        root.mainloop()