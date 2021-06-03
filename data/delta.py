from collections import defaultdict, namedtuple, Counter
from array import array
from copy import deepcopy
from random import random
from nltk.tree import Tree
from utils.types import E_ORIF4, NIL
from utils.math_ops import s_index, t_index

OrderX = namedtuple('OrderX', 'sorted, left_slice, neutral_slice, right_slice')
LogitX = namedtuple('LogitX', 'jnt, det, dir, rgt, phs, ftg')
OriFct = namedtuple('OriFct', E_ORIF4)

NIL = '<nil>'
LNR = '<->'
LOX, NOX, ROX = LNR
E_XDIM = LogitX(*range(6))

__JNT = 1 << E_XDIM.jnt
__DET = 1 << E_XDIM.det
__DIR = 1 << E_XDIM.dir
__RGT = 1 << E_XDIM.rgt
__PHS = 1 << E_XDIM.phs
__FTG = 1 << E_XDIM.ftg

get_jnt = lambda x: (x & __JNT) > 0
get_det = lambda x: (x & __DET) > 0
get_dir = lambda x: (x & __DIR) > 0
get_rgt = lambda x: (x & __RGT) > 0
get_phs = lambda x: (x & __PHS) > 0
get_ftg = lambda x: (x & __FTG) > 0

def xtype_to_logits(xtype, to_str = True):
    if xtype == NOX:
        return '0' if to_str else 0
    logits = 0
    if xtype[0] != NOX:
        logits |= __DIR
        if xtype[0] == ROX:
            logits |= __RGT
    if 's' in xtype:
        logits |= __DET
    if 'j' in xtype:
        logits |= __JNT
    if 'p' in xtype:
        logits |= __PHS
    if 'f' in xtype:
        logits |= __FTG
    return str(logits) if to_str else logits

def logits_to_xtype(logits):
    if logits == 0:
        return '-'
    s = ''
    if logits & __DIR:
        s += '>' if logits & __RGT else '<'
    else:
        s += '-'
    if logits & __JNT:
        s += 'j'
    if logits & __PHS:
        s += 'p'
    if logits & __DET:
        s += 's'
    if logits & __FTG:
        s += 'f'
    return s

def get_logits(right, directional, is_joint, is_phrase, cnf_stable, has_ftag):
    logits = 0
    if directional:
        logits |= __DIR
    if right:
        logits |= __RGT
    if cnf_stable:
        logits |= __DET
    if is_joint:
        logits |= __JNT
    if is_phrase:
        logits |= __PHS
    if has_ftag:
        logits |= __FTG
    return logits

def lnr_order(vocabs):
    ov = defaultdict(list)
    for v in vocabs:
        ov[v[0]].append(v[1:])
    od = []
    cu = [0]
    for i in LNR:
        ov[i].sort()
        od.extend(i+j for j in ov[i])
        cu.append(cu[-1] + len(ov[i]))
    return OrderX(od, slice(*cu[:2]), slice(*cu[1:3]), slice(*cu[2:4]))

def get_xtype(path, LNR = (-1, 0, 1)):
    return (LNR[0] if path[-1] else LNR[2]) if path else LNR[1]

def preproc_cnf(mtree,
                replace_junc = '@',
                pos_in_syn   = '#',
                lower        = False,
                word_trace   = False):
    # get rid of trace mtree e.g. (-NONE- *)
    # combine other unary branch e.g. replace NP-NNP with NNP
    # import pdb; pdb.set_trace()
    for i in reversed(range(len(mtree.leaves()))):
        word_path = mtree.leaf_treeposition(i)
        word, update = mtree[word_path], False
        if '\\' in word:
            word = word.replace('\\', '')
            update = True
        elif word == '-LRB-':
            update = True
            word = '('
        elif word == '-RRB-':
            update = True
            word = ')'
        elif lower:
            word = word.lower()
            update = True
        if update:
            mtree[word_path] = word
        pos_path = word_path[:-1]
        syn_path = pos_path[:-1] # leaf must be unary
        pos_tag = mtree[pos_path].label()
        if '-' in pos_tag: # -NONE-
            remove = pos_tag.index('-')
            if remove and remove != len(pos_tag) - 1: # ignore -NONE-
                mtree[pos_path].set_label(pos_tag[:remove])
        if word_trace:
            remove = (word[0] == '*' and (word[-1] == '*' or '*' in word[1:] and word[-1].isdigit()))
        # elif callable(trace_remove):
        #     remove = trace_remove(word, pos_tag)
        else:
            remove = pos_tag == '-NONE-'
        if remove: # POS
            if len(mtree[syn_path]) > 1: # more than one child
                # NP (SBAR) (-NONE- *-1)
                del mtree[pos_path]
            else: # NP -NONE- *-1
                while syn_path and len(mtree[syn_path[:-1]]) == 1:
                    syn_path = syn_path[:-1]
                del mtree[syn_path]
        elif pos_in_syn and len(mtree[syn_path]) > 1:
            pos_unary = mtree[pos_path]
            child_pos = pos_path[-1]
            if child_pos == 0 or len(mtree[syn_path][child_pos:]) > 1:
                any_non_traces = True
            else:
                any_non_traces = False
                for cousin in mtree[syn_path][0:child_pos]: # previous cousins
                    any_non_traces |= any(t != '-NONE-' for _, t in cousin.pos())
            if pos_unary.height() == 2 and any_non_traces:
                mtree[pos_path] = Tree(pos_in_syn + pos_unary.label(), [pos_unary])
    if not replace_junc:
        return
    for b in mtree.subtrees():
        l, update = b.label(), False
        assert len(l), f'Invalid tree with empty label: \n{str(mtree)}'

        if b.height() < 3: # pos tag
            if l[-1] == '-': # -LRB- -RRB-
                l = l[1:-1]
                update = True
            elif word_trace:
                if l[-1] == '3' or l in ('NP', 'PP'):
                    l = l[:-1]
                    update = True
                elif l == 'CARD':
                    l = 'CL'
                    update = True
                elif l == 'FS':
                    l = 'CONJ'
                    update = True
        else:
            if l[-1] == '-':
                l = pos_in_syn + l[2:-1] # #-LRB- to #LRB
                update = True
            if l[-1].isdecimal(): # remove -1 or =1
                p = l.rfind('=')
                if p > 0:
                    l = l[:p]
                else:
                    l = l[:l.rfind('-')]
                update = True
            if '-' in l:
                l = l.replace('-', replace_junc) # NP-SUB
                update = True
            if l in ('ADVP|PRT', 'PRT|ADVP'):
                l = 'ADVP+PRT'
                update = True
        if word_trace:
            if l == 'PP{MIYAKE_ISLAND_OUTSIDE}':
                l = 'PP'
                update = True
            if l in ('MENTION', 'EDITOR'):
                l = 'IP'
            if l in ('P', 'N', 'CONJ'):
                l += 'P'
                update = True
            if ';' in l:
                l = l.replace(';', replace_junc)
                update = True
        if update:
            b.set_label(l)

def count_binarized_lr_children(tree):
    l, r = 0, 0
    for t in tree.subtrees(): 
        if len(t) > 1:
            if len(t[0]) > 1:
                l += 1
            if len(t[1]) > 1:
                r += 1
    return l, r

import sys
def explain_warnings(warnings, label_layers, tag_layer):
    templates = ['pos %(l)s and pos_in_syn %(p)s are not consistent',
                 'leftmost %(l)s directs away from %(p)s',    'rightmost %(r)s directs away from %(p)s', # bad
                 'left %(l)s goes through %(p)s',             'right %(r)s goes through %(p)s', # hard
                 'right %(r)s changes to %(p)s during relay', 'left %(l)s changes to %(p)s during relay', # okay strange 
                 'discard %(p)s',                             'root _%(p)s was a subphrase', # okay almost
                 '%(l)s and %(r)s join into %(p)s',           'lose subtree'] # err
    info = []
    for l, i, t in warnings:
        if i < 0:
            info.append(templates[i])
            continue
        if l == -1:
            data = dict(
                p = label_layers[0][i],
                l = tag_layer[i],
            )
        else:
            data = dict(
                p = label_layers[l][i],
                l = label_layers[l-1][i]   if l else tag_layer[i],
                r = label_layers[l-1][i+1] if l else tag_layer[i+1],
            )
        info.append(f'[{l}.{i}]', templates[t] % data)
    return info

templates = ['pos and pos_in_syn not consistent',
             'left/rightmost child directs away', # 1,2 not okay
             'child goes through <nil>s', # 3,4
             'tag changes during relay', # 5,6
             'discard non-<nil> parent',
             'root is a subphrase', # okay almost
             'children join into <nil>',
             'lose subtree']

def explain_one_error(err):
    return templates[err[2]] + f' at layer {err[0]}'

def sumup_warnings(warnings):
    cnt = defaultdict(int)
    for wtype, wcnt in Counter(warnings[:, 2]).items():
        if wtype <= 0: # without left or right attribute
            i = wtype
        elif wtype < 7: # left or right
            i = (wtype - 1) // 2 + 1
        else: # without again
            i = wtype - 3
        cnt[i] += wcnt
    for wtype, wcnt in cnt.items():
        yield templates[wtype], wcnt

import numpy as np
def warning_level(warnings):
    if isinstance(warnings, np.ndarray):
        warnings = warnings[:, 2]
    if len(warnings) == 0:
        return 0
    if 1 in warnings or 2 in warnings:
        return -2
    if -1 in warnings or -2 in warnings:
        return -1
    if 3 in warnings or 4 in warnings: # go through <nil>
        return 3
    if 1 in warnings or 2 in warnings or 7 in warnings: # go into paddings / discard something
        return 2
    return 1 # 0,5,6,8: pos/pos_in_syn, tag change, top is subtree

def get_tree_from_triangle(word_layer, tag_layer, label_layers, right_layers, pos_in_syn = '#', _sub = '_'):
    def _phrase(t): # -> list
        return t[:] if t.label()[0] == _sub else [t]

    warnings   = []
    last_layer = []
    if tag_layer is None:
        pos_in_syn = ''
        for i, (w, s) in enumerate(zip(word_layer, label_layers[0])):
            w = {'(': '-LRB-', ')': '-RRB-'}.get(w, w)
            last_layer.append(Tree(s, [w]))
    else:
        for i, (w, p, s) in enumerate(zip(word_layer, tag_layer, label_layers[0])):
            w = {'(': '-LRB-', ')': '-RRB-'}.get(w, w)
            if p in ('LRB', 'RRB'):
                p = '-' + p + '-'
            tagged_leaf = Tree(p, [w])
            if s[0] == _sub:
                tree = tagged_leaf
            elif s[0] == pos_in_syn:
                s = s[1:]
                if s == p:
                    tree = tagged_leaf
                else:
                    tree = Tree(s, [tagged_leaf])
                    warnings.append((-1, i, 0))
            elif '+' in s:
                tree = tagged_leaf
                segs = s.split('+')
                while segs:
                    tree = Tree(segs.pop(), [tree])
            else:
                tree = Tree(s, [tagged_leaf])
            last_layer.append(tree)

    leftmost, rightmost = 0, len(last_layer) - 1
    for layer_cnt, (right, upper) in enumerate(zip(right_layers, label_layers[1:])):
        this_layer = []
        rightmost -= 1
        smooth = len(right) == len(upper) + 1
        skipped_none = 0

        for i, p in enumerate(upper):
            if p[0] == pos_in_syn:
                p = p[1:]
            if smooth:
                l_child, r_child = last_layer[i], last_layer[i+1]
                lcrward, rclward =      right[i],  not right[i+1]
                left_relay       = l_child and lcrward
                right_relay      = r_child and rclward
            else:
                while True: # 2 or 3 hours ???
                    if i+skipped_none+1 == len(last_layer):
                        raise ValueError((layer_cnt, i+skipped_none, -1), last_layer, warnings)
                    l_child, r_child = last_layer[i+skipped_none], last_layer[i+skipped_none+1]
                    lcrward, rclward =      right[i+skipped_none],  not right[i+skipped_none+1]
                    left_relay       = l_child and lcrward
                    right_relay      = r_child and rclward
                    if left_relay or right_relay:
                        break
                    skipped_none += 1

            if i == leftmost and not lcrward: # left most shall be restrictly not nil and rightwards
                raise ValueError((layer_cnt, i, 1), last_layer, warnings)
                # warnings.append((layer_cnt, i, 1))
                # if r_child is None:
                #     this_layer.append(l_child)
                # else:
                #     this_layer.append(Tree(l_child.label(), ([l_child] if l_child.height() == 2 else l_child[:]) + _phrase(r_child)))
            elif i == rightmost and not rclward: # counterpart
                raise ValueError((layer_cnt, i, 2), last_layer, warnings)
                # warnings.append((layer_cnt, i, 2))
                # if l_child is None:
                #     this_layer.append(r_child)
                # else:
                #     this_layer.append(Tree(r_child.label(), _phrase(l_child) + ([r_child] if r_child.height() == 2 else r_child[:])))
            elif p == NIL: # phrase boundary -> nil
                if layer_cnt and left_relay and right_relay:
                    raise ValueError((layer_cnt, i, -2), last_layer, warnings)
                elif left_relay:
                    warnings.append((layer_cnt, i, 3))
                    this_layer.append(l_child)
                elif right_relay:
                    warnings.append((layer_cnt, i, 4))
                    this_layer.append(r_child)
                else:
                    this_layer.append(None)
            elif left_relay and right_relay: # word/phrase joint
                if '+' in p:
                    segs = p.split('+')
                    tree = Tree(segs.pop(), _phrase(l_child) + _phrase(r_child) )
                    while segs:
                        tree = Tree(segs.pop(), [tree])
                    this_layer.append(tree)
                else:
                    this_layer.append(Tree(p, _phrase(l_child) + _phrase(r_child) ))
            elif right_relay:
                if p[0] != _sub and not p.startswith(r_child.label()) and r_child.height() > 2: # should we keep the label?
                    # if r_child.label().startswith('_') or p.startswith('_'):
                    # if : # maybe not, less warning and more accurate
                    #     print('protect', r_child, p)
                    #     r_child = Tree(p, r_child)
                    # else:
                    r_child.set_label(p)
                    warnings.append((layer_cnt, i, 5))
                this_layer.append(r_child)
            elif left_relay:
                if p[0] != _sub and not p.startswith(l_child.label()) and l_child.height() > 2:
                    # if : # maybe not, less warning and more accurate
                    #     print('protect', l_child, p)
                    #     l_child = Tree(p, l_child)
                    # else:
                    l_child.set_label(p)
                    warnings.append((layer_cnt, i, 6))
                this_layer.append(l_child)
            else:
                warnings.append((layer_cnt, i, 7))
                this_layer.append(None)
        if len(word_layer) != sum(len(t.leaves()) for t in this_layer if t):
            raise ValueError((layer_cnt, -1, -1), last_layer, warnings)
        last_layer = this_layer
    root = last_layer[0]
    root_label = root.label()
    if root_label[0] == _sub:
        warnings.append((layer_cnt, 0, 8))
        # root.set_label('S' if root_label == '_SUB' else i[root_label:])
    return root, warnings

def after_to_tree(token_layer, tag_layer, label_layers, right_layers,
                    return_warnings = False,
                    on_warning      = None,
                    on_error        = None,
                    error_prefix    = '',
                    error_root      = 'S'):
    try:
        tree, warnings = get_tree_from_triangle(token_layer, tag_layer, label_layers, right_layers)
    except ValueError as e:
        error, last_layer, warnings = e.args
        if callable(on_error):
            on_error(error_prefix, explain_one_error(error))
        tree = Tree(error_root, [x for x in last_layer if x]) # Trust the model: TODO report failure rate
        warnings.append(error)
    if warnings and callable(on_warning) and tag_layer is not None:
        on_warning(explain_warnings(warnings, label_layers, tag_layer))
    if return_warnings: # [:, 2] > 8 is error
        warnings = np.asarray(warnings, dtype = np.int8)
        warnings.shape = (-1, 3)
        return tree, warnings
    return tree

def explore_unary(func, unary, *args):
    if unary.height() > 2:
        return Tree(unary.label(), list(func(t, *args) for t in unary))
    return unary.copy()

def midout_factored(tree):
    num_children = len(tree)
    if num_children > 1:
        if num_children > 2:
            mid = num_children >> 1
            if num_children % 2 == 1: # odd number
                mid += int(random() > 0.5) # remove place bias
        else:
            mid = 1

        if mid > 1:
            l_child = midout_factored(Tree(tree.label() + '|<o>', tree[:mid]))
        else:
            l_child = midout_factored(tree[0])

        if num_children - mid > 1:
            r_child = midout_factored(Tree(tree.label() + '|</o>', tree[mid:]))
        else:
            r_child = midout_factored(tree[-1])
        return Tree(tree.label(), [l_child, r_child])
    return explore_unary(midout_factored, tree)

def midin_factored(tree, take_left = None):
    num_children = len(tree)
    if take_left is None:
        take_left = random() > 0.5
    if num_children > 1:
        if take_left:
            l_child = midin_factored(tree[0], take_left)
            take_left = not take_left
            if num_children > 2:
                r_child = midin_factored(Tree(tree.label() + '|</i>', tree[1:]), take_left)
            else:
                r_child = midin_factored(tree[1], take_left)
        else:
            take_left = not take_left
            if num_children > 2:
                l_child = midin_factored(Tree(tree.label() + '|<i>', tree[:-1]), take_left)
            else:
                l_child = midin_factored(tree[0], take_left)
            take_left = not take_left
            r_child = midin_factored(tree[-1], take_left)
        return Tree(tree.label(), [l_child, r_child])
    return explore_unary(midin_factored, tree)

class X:
    @classmethod
    def pyramid(cls, bottom_len, sep = '@', sub = '_'):
        return tuple(cls(i, sep, sub) for i in range(s_index(bottom_len)))
        
    def __init__(self, sid, sep, sub):
        self._sid   = sid
        self._sep   = sep
        self._sub   = sub
        self._goods = None # dynamic
        self._xtype = NOX # dynamic
        self._tid   = t_index(sid)

    def reset(self):
        if self._goods:
            self._goods = None
            self._xtype = NOX

    def prepare(self, path, label_callback, pop_time): # TODO: test
        path = list(path)
        if any(path.pop() for _ in range(pop_time)):
            raise ValueError('One pos contains two words')
        self._goods = self._tid, path, label_callback
        self._xtype = get_xtype(path, '<->') + 'j'

    def leave(self, remove = False):
        if self._goods is None:
            return self._sid # <nil> node

        (lid, offset), path, label_callback = self._goods
        # 0 <- parent with two kids
        # 0 1, thus from the view point of a child:
        # 1 for left (right/2nd/1 child), 0 for right (left/1st/0 child)
        # i = len(path) - 1
        # while i >= 0 and path[-1] == path[i]:
        #     self._profoundity += 1
        #     i -= 1
        if path[-1]: # as left child goes right
            next_tid = lid - 1, offset - 1
        else: # as right child goes left
            next_tid = lid - 1, offset
        if remove:
            xtype = self._xtype
            self.reset()
            return xtype, (path, label_callback)
        return next_tid, (path, label_callback)

    def arrive(self, goods, legacy = None):
        if self._goods is None:
            # pass the necessary goods
            self._goods = goods
            self._xtype = get_xtype(goods[1], '<->') if legacy is None else legacy
        else: # join with other
            tid, path, lcb = self._goods
            path = path.copy()
            if self._goods[0] != goods[0] or len(path) != len(self._goods[1]) or path.pop() == goods[1][-1]:
                raise ValueError("bad join")
            self._goods = tid, path, lcb
            self._xtype = get_xtype(path, '<->') + 'j'
    
    def __str__(self):
        if self._goods is None:
            return "nil Xnode %d %r" % (self._sid, self.tid)
        tid, path, lcb = self._goods
        return "Xnode %d %r\t%s %s" % (self._sid, tid, get_xtype(path, 'v-^'), lcb(path))

    def labels(self):
        # A@B(-125)|<C-D@SBJ(-2)-E-(-)LRB(-)> # ('-' removed in cnf_preproc)
        if self._goods is None:
            return
        path, lcb = self._goods[1:]
        syn = lcb(path)
        bar = syn.find('|')
        if bar >= 0:
            syn = syn[:bar] # redundancy |<...>
        if '+' in syn:
            pls, segs = [], set()
            for s in syn.split('+'):
                sep = s.find(self._sep)
                if sep >= 0:
                    seg = s.split(self._sep)
                    s = seg[0]
                    segs |= set(seg[1:])
                if s not in pls and s[0] != '#': # incredible in ktb
                    pls.append(s)
            # pls.sort()
            syn = '+'.join(pls)
            for seg in segs:
                syn += self._sep + seg
        if bar < 0:
            segs = tuple((self._sep + seg) if i else seg for i, seg in enumerate(syn.split(self._sep)))
            if 'j' == self._xtype[-1]:
                return syn, segs
        else: # CNF sub phrase - dummy tag
            segs = tuple((self._sep if i else self._sub) + seg for i, seg in enumerate(syn.split(self._sep)))
        return syn, segs[:1]

    @property
    def xtype(self):
        xtype = self._xtype
        if self._goods:
            path, lcb = self._goods[1:]
            if 'j' == xtype[-1]: # [<->]j?
                lbl = lcb(path)
                if '|' not in lbl and not any(i == lbl[0] for i in '#_'):
                    xtype += 'p' # [<->]jp, use these labels to locate phrases
                    # - is nil tag
                    # [<>] are relaying branches
                    # [<->]j are sub phrases
                    # -jp shall be sentence
                    # [<>]jp are phrases
            path_len = len(path)
            if path_len:
                if path_len == 1 or path[-1] == path[-2] or  '|' not in lcb(path[:-1]):
                    xtype += 's'
        return xtype

    @property
    def tid(self):
        return self._tid

class DeltaX:
    @classmethod
    def from_penn(cls, tree, factor = 'left', word_trace = False, do_preproc = True):
        if do_preproc:
            preproc_cnf(tree, word_trace = word_trace) # open in the furture
        if factor in ('left', 'right'):
            tree = deepcopy(tree)
            tree.chomsky_normal_form(factor)
        elif factor == 'midin':
            tree = midin_factored(tree)
        else:
            tree = midout_factored(tree)
        tree.collapse_unary(collapseRoot = True)
        lrc = count_binarized_lr_children(tree)
        return cls(tree, 2), lrc

    @classmethod
    def from_penn_quad(cls, tree, word_trace = False):
        preproc_cnf(tree, word_trace = word_trace) # open in the furture
        midi = midin_factored(tree)
        mido = midout_factored(tree)
        eert = deepcopy(tree)
        tree.chomsky_normal_form('left')
        eert.chomsky_normal_form('right')
        l_lrc = count_binarized_lr_children(tree)
        r_lrc = count_binarized_lr_children(eert)
        i_lrc = count_binarized_lr_children(midi)
        o_lrc = count_binarized_lr_children(mido)
        tree.collapse_unary(collapseRoot = True)
        eert.collapse_unary(collapseRoot = True)
        midi.collapse_unary(collapseRoot = True)
        mido.collapse_unary(collapseRoot = True)
        # if sum(l_lrc) != sum(r_lrc):
        #     print(l_lrc, sum(l_lrc), '-', r_lrc, sum(r_lrc), '=', sum(l_lrc) - sum(r_lrc))
        #     tree.draw()
        #     eert.draw()
        # else:
        #     print('o')
        dxs = OriFct(cls(tree, 2), cls(eert, 2), cls(midi, 2), cls(mido, 2))
        lcr = OriFct(l_lrc, r_lrc, i_lrc, o_lrc)
        return dxs, lcr

    @classmethod
    def from_stan(cls, tree, lower = False):
        for i, (w, p) in enumerate(tree.pos()):
            if w == '-LRB-':
                w = '('
            elif w == '-RRB-':
                w = ')'
            elif lower:
                w = w.lower()
            p = tree.leaf_treeposition(i)
            tree[p] = w
        return cls(tree, 1)

    def __init__(self, tree, pop_time):
        get_label = lambda path: tree[path].label()
        self._wp  = tree.pos()
        l         = len(self._wp)
        n         = X.pyramid(l)
        self._pt  = pop_time
        for i, x in enumerate(n[-l:]): # diff from neural index
            x.prepare(tree.leaf_treeposition(i), get_label, pop_time)
        self._depth = l
        self._nodes = n
        self._segments = None

    def word_tag(self, *v2is):
        if v2is:
            w2i, t2i = v2is
            return zip(*((w2i(w), t2i(t)) for w, t in self._wp))
        return zip(*self._wp)

    def build_trapezoid(self, every_n = 3):
        n = self._nodes
        lives = self._depth
        self._segments = []
        while True:
            self._segments.append(lives)
            end = lives - every_n
            if end < 1:
                end = 1
            for l in range(lives, end, -1): # till the top
                bottom = n[s_index(l-1):s_index(l)]
                for x in bottom:
                    status = x.leave()
                    if isinstance(status, tuple):
                        tid, goods = status
                        goods = (tid,) + goods
                        n[s_index(*tid)].arrive(goods)
            if end == 1:
                break
            top = n[s_index(end-1):s_index(end)]
            top = [x for x in top if x._goods is not None]
            lives = len(top)
            level = lives - 1
            for l, x in enumerate(top):
                xtype, goods = x.leave(remove = True)
                tid = level, l
                goods = (tid,) + goods
                n[s_index(level, l)].arrive(goods, xtype)
        if self._segments[-1] - every_n == 1:
            self._segments.append(1)

    def build_triangle(self, debug_mode = False):
        l = self._depth
        n = self._nodes

        # build the triangular in a bottom-up fashion
        if debug_mode:
            debug_mode = ['Verbose message']
            
        for l in range(self._depth, 1, -1): # till the top
            bottom = n[-l:] # without s_index
            if debug_mode:
                debug_mode.append("Nodes: %d" % l)
            for x in bottom:
                if debug_mode:
                    debug_mode.append(' ' + str(x))
                status = x.leave()
                if isinstance(status, tuple):
                    tid, goods = status
                    goods = (tid,) + goods
                    n[s_index(*tid)].arrive(goods)
            n = n[:-l]
        self._segments = [l]

        if debug_mode:
            debug_mode.append("Final:")
            debug_mode.append(' ' + str(n[0]))
            debug_mode.reverse()
            print('\n'.join(debug_mode))

    def to_triangles(self):
        if self._segments is None:
            self.build_triangle()
        ftags = {}
        labels = []
        xtypes = []

        for sid, n in enumerate(self._nodes):
            xty = n.xtype
            res = n.labels()
            if res is None:
                labels.append(NIL)
            else:
                labels.append(res[1][0])
                if len(res[1]) > 1:
                    xty += 'f'
                    ftags[sid] = res[1][1]
            xtypes.append(xty)
                
        return labels, xtypes, ftags #, prof, de

    def trapezoid_gen(self, every_n, *i2vs):
        if i2vs:
            l2i, x2i = i2vs
        if self._segments is not None:
            for n in self._nodes[:s_index(self._depth-1)]:
                n.reset()
        self.build_trapezoid(every_n)
        for seg_start in self._segments:
            for l in range(every_n):
                labels = []
                xtypes = []
                start = s_index(seg_start-1-l)
                end   = s_index(seg_start-l)
                if start == end:
                    break
                for n in self._nodes[start:end]:
                    xty = n.xtype
                    lbl = n.labels()
                    lbl = NIL if lbl is None else lbl[1][0]
                    if i2vs:
                        lbl = l2i(lbl)
                        xty = x2i(xty)
                    labels.append(lbl)
                    xtypes.append(xty)
                yield labels, xtypes

    def __str__(self):
        words, tags = self.word_tag()
        labels, xtypes, _ = self.to_triangles()
        num_words = len(words)
        l_desc = ''
        x_desc = ''
        for level in range(num_words):
            start = s_index(level)
            end = start + level + 1
            pad = ' ' * 3 * (num_words - level)
            l_desc += pad
            l_desc += ''.join(l.center(6) for l in labels[start:end]) + '\n'
            x_desc += pad
            x_desc += ''.join(l.center(6) for l in xtypes[start:end]) + '\n'
        l_desc += ' ' * 3
        l_desc += ''.join(l.center(6) for l in tags) + '\n'
        l_desc += ' ' * 3
        l_desc += ''.join(l.center(6) if len(l) < 6 else (l[:5] + '…') for l in words) + '\n'
        return l_desc + x_desc[:-1]

def bottom_up_ftags(ftags, seq_to_str = True):

    def bottom_up(sid):
        tid = t_index(sid)
        # print(tid)
        return -tid[0], tid[1]

    pos_ftags = []
    for i, fcoord in enumerate(sorted(ftags, key = bottom_up)):
        # for ft in :
        pos_ftags.append((i, ftags[fcoord]))
    
    ftags, fseqs = [], []
    for i, (fpos, ftag) in enumerate(pos_ftags):
        sid = (fpos - pos_ftags[i-1][0]) if i else 0
        if seq_to_str:
            sid = str(sid)
        ftags.append(ftag)
        fseqs.append(sid)
    return ftags, fseqs

def write_tensors(labels, xtypes, tensor_labels, tensor_xtypes, offset, paddings = None, vocab = None, skip_top = 0):
    tensor_vlen = tensor_labels.shape[0] + skip_top
    tensor_height, oset = t_index(tensor_vlen)
    assert oset == 0
    # assert tensor_labels.shape == tensor_xtypes.shape
    py_len = len(labels)
    py_height, oset = t_index(py_len)
    assert oset == 0
    assert py_len == len(xtypes)
    height_diff = tensor_height - py_height
    assert height_diff >= 0
    if paddings:
        l_bos, l_eos, x_bos, x_eos = paddings
        eos_d = height_diff - offset

    for src, (lbl, xty) in enumerate(zip(labels, xtypes)):
        if xty:
            lid, oset = t_index(src)
            dst = s_index(lid + height_diff, oset + offset) - skip_top
            if vocab is not None:
                lbl = vocab[lbl]
            tensor_labels[dst] = lbl
            tensor_xtypes[dst] = xty
            if paddings:
                if oset == 0:
                    start = dst - offset
                    tensor_labels[start:dst] = l_bos
                    tensor_xtypes[start:dst] = x_bos
                if oset == lid:
                    start = dst + 1
                    end = start + eos_d
                    tensor_labels[start:end] = l_eos
                    tensor_xtypes[start:end] = x_eos