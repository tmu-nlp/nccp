from data.delta import preproc_cnf, Tree, defaultdict
from data.cross import _read_dpenn, draw_str_lines as _draw_str_lines
from data.mp import DM

class MaryDM(DM):
    @staticmethod
    def tree_gen_fn(i2w, i2t, i2l, segments, token, tag, label, fence, seg_length):
        for tokens, tags, labels, fences, seg_lengths in zip(token, tag, label, fence, seg_length):
            layers_of_label = []
            layers_of_fence = []
            label_start = 0
            fence_start = 0
            for l_cnt, (l_size, l_len) in enumerate(zip(segments, seg_lengths)):
                label_layer = tuple(i2l(i) for i in labels[label_start: label_start + l_len])
                layers_of_label.append(label_layer)
                if l_cnt:
                    layers_of_fence.append(fences[fence_start: fence_start + l_len + 1])
                    fence_start += l_size + 1
                else:
                    ln = l_len
                if l_len == 1:
                    break
                label_start += l_size
            wd = [i2w[i] for i in tokens[:ln]]
            tg = [i2t[i] for i in   tags[:ln]]
            tree, _ = get_tree_from_signals(wd, tg, layers_of_label, layers_of_fence, 'VROOT')
            yield ' '.join(str(tree).split())

    @staticmethod
    def arg_segment_fn(seg_id, seg_size, batch_size, args):
        # import pdb; pdb.set_trace()
        start = seg_id * seg_size
        if start < batch_size:
            return args[:1] + tuple(x[start: (seg_id + 1) * seg_size] for x in args[1:])

def add_efficient_subs(stretched_tree, sub = '_', max_range = None):
    if stretched_tree.height() > 3:
        parent_label = stretched_tree.label()
        modified = False
        children = []
        for t in stretched_tree:
            m, t = add_efficient_subs(t, sub, max_range)
            modified |= m
            children.append(t)
        heights = set(t.height() for t in children)
        if max_range is None:
            upper = max(heights)
        else:
            upper = min(heights) + max_range
            upper = min(upper, max(heights))
        for ht in range(min(heights), upper):
            new_children = []
            for child in children:
                if not new_children or child.height() > ht:
                    new_children.append([child])
                else:
                    if new_children[-1][-1].height() > ht:
                        new_children.append([child])
                    else:
                        new_children[-1].append(child)
            children = []
            for group in new_children:
                if len(group) == 1:
                    children.append(group.pop())
                else:
                    modified = True
                    children.append(Tree(sub + parent_label, group))
        if modified:
            return True, Tree(parent_label, children)
    return False, stretched_tree

def clear_label(label, umark = '+', fmark = '@'):
    '''Most unaries are introduce by preproc_cnf/remove trace'''
    segs = []
    for seg in label.split(umark):
        if fmark in seg:
            seg = seg[:seg.index(fmark)]
        if seg not in segs:
            segs.append(seg)
    return umark.join(segs)

keep_str = lambda x: x
def signals(tree, w2i = keep_str, t2i = keep_str, l2i = keep_str):
    paths = []
    bottom = []
    bottom_label = []
    words, tags = [], []
    for wid, (word, tag) in enumerate(tree.pos()):
        path = tree.leaf_treeposition(wid)[:-2]
        paths.append(path)
        words.append(w2i(word))
        tags .append(t2i(tag ))
        bottom_label.append(l2i(clear_label(tree[path].label())))

    layers_of_labels = [bottom_label]
    layers_of_splits = []
    while len(paths) > 1:
        label_layer = []
        split_layer = []
        next_paths  = []
        child_cnt   = 0
        for pid, path in enumerate(paths):
            path_len = len(path)
            is_a_child = path[-1] > 0 and path_len == last_path_len
            complete = is_a_child and len(tree[path[:-1]]) == child_cnt + 1
            if complete:
                if child_cnt > 1:
                    split_layer = split_layer[:1-child_cnt]
                while child_cnt:
                    # print(pid, 'pop cousin:k', label_layer.pop())
                    label_layer.pop()
                    cousin_path = next_paths.pop()
                    assert cousin_path[:-1] == path[:-1]
                    child_cnt -= 1
                parent_path = path[:-1]
                label = tree[parent_path].label()
                next_paths.append(parent_path)
            else:
                next_paths.append(path)
                label = tree[path].label()
                if is_a_child:
                    child_cnt += 1
                else:
                    child_cnt = 1
                split_layer.append(pid)
            label_layer.append(l2i(clear_label(label)))
            last_path_len = path_len
        paths = next_paths
        assert len(label_layer) == len(split_layer)
        split_layer.append(pid + 1)
        # print(' '.join(label_layer))
        # print(split_layer)
        # print(' '.join(f'{label}({end-start})' for label, start, end in zip(label_layer, split_layer, split_layer[1:])))
        # print()
        layers_of_labels.append(label_layer)
        layers_of_splits.append(split_layer)
    return words, tags, layers_of_labels, layers_of_splits

from utils.str_ops import height_ratio
def coord_vote(units, fence_location):
    if units.sum() > 0:
        pro_num_ratio = (units > 0).sum() / units.size
        unit_location = units.argmax()
    else:
        pro_num_ratio = (units <= 0).sum() / units.size
        unit_location = units.argmin()
    if fence_location > unit_location:
        return f'«{fence_location - unit_location - 1}{height_ratio(pro_num_ratio)}'
    return f'{height_ratio(pro_num_ratio)}{unit_location - fence_location}»'

def flatten_children(nodes):
    children = []
    for x in nodes:
        if isinstance(x, Tree):
            children.append(x)
        else:
            children.extend(x)
    return children

func_unary_char = '&'
def flatten_children_with_weights(bottom, start, weights, bar = '│'):
    children = []
    head_child = None
    max_weight = 0
    for nid, sub_tree in enumerate(bottom):
        mean, _ = weights[start + nid]
        label = f'{mean * 100:.0f}%'
        if isinstance(sub_tree, Tree):
            unary_label = sub_tree.label()
            if bar in unary_label:
                sub_tree.set_label(label + func_unary_char + unary_label)
            else:
                sub_tree = Tree(label, [sub_tree])
        else:
            sub_tree = Tree(label, flatten_children(sub_tree))
        children.append(sub_tree)
        if mean > max_weight:
            max_weight = mean
            head_child = sub_tree
    return children, head_child

def flatten_layer_with_fence_vote(bottom, fence_vote, bar = '│'):
    children = []
    for nid, sub_tree in enumerate(bottom):
        if nid == 0:
            lhs, rhs = fence_vote[:2]
            lhs = coord_vote(lhs, nid)
            rhs = coord_vote(rhs, nid + 1)
            lhs_len = len(lhs)
            rhs_len = len(rhs)
            if lhs_len > rhs_len:
                label = lhs + bar + rhs + ' ' * (lhs_len - rhs_len)
            elif lhs_len < rhs_len:
                label = lhs + ' ' * (rhs_len - lhs_len) + bar + rhs
            else:
                label = lhs + bar + rhs 
        else:
            rhs = fence_vote[nid + 1]
            rhs = coord_vote(rhs, nid + 1)
            lhs = ' ' * len(rhs)
            label = lhs + bar + rhs
        if isinstance(sub_tree, Tree):
            unary_label = sub_tree.label()
            if bar in unary_label:
                sub_tree.set_label(' ' + label + func_unary_char + unary_label) # L%FL%FLLL
            else:
                sub_tree = Tree(' ' + label, [sub_tree])
        else:
            sub_tree = Tree(' ' + label, flatten_children(sub_tree))
        children.append(sub_tree)
    return children

def unary_label_match(tree, label):
    while tree.height() > 2:
        if tree.label() == label:
            return True
        if len(tree) == 1:
            tree = tree[0]
        else:
            return False
    return False

def get_tree_from_signals(word, tag, layers_of_labels, layers_of_splits, 
                          fall_back_root           = None,
                          layers_of_weights        = None,
                          layers_of_fence_vote     = None,
                          mark_np_without_dt_child = False):
    bottom = []
    add_weight_base = layers_of_weights is not None
    balancing_bottom_sub = add_weight_base and any(x[0] not in '#_' for x in layers_of_labels[0])
    add_fence_vote_base = layers_of_fence_vote is not None
    unary_chars = '│' + func_unary_char 
    for w, t, label in zip(word, tag, layers_of_labels[0]):
        if w == '(':
            w = '-LRB-'
        elif w == ')':
            w = '-RRB-'
        if t in ('LRB', 'RRB'):
            t = '-' + t + '-'
        leaf = Tree(t, [w])
        if label[0] not in '#_':
            leaf = Tree(label, [leaf])
        elif balancing_bottom_sub:
            leaf = Tree('│', [leaf])
        bottom.append(leaf)

    if add_weight_base:
        headedness_stat = {}
    else:
        headedness_stat = None

    for lid, (split_layer, label_layer) in enumerate(zip(layers_of_splits, layers_of_labels[1:])):
        new_bottom = []
        # leave_cnt = 0
        add_weight = add_weight_base and lid < len(layers_of_weights)
        balancing_sub = add_weight_base and any(x[0] not in '#_' for x in label_layer)
        add_fence_vote = add_fence_vote_base and lid < len(layers_of_fence_vote)
        if add_fence_vote:
            bottom = flatten_layer_with_fence_vote(bottom, layers_of_fence_vote[lid])

        for label, start, end in zip(label_layer, split_layer, split_layer[1:]):
            if end - start == 1: # unary
                sub_tree = bottom[start]
                if label[0] in '#_' or (unary_label_match(sub_tree[0], label) if add_fence_vote else (sub_tree.label() == label)):
                    # relay sub
                    if balancing_sub:
                        relay_label = sub_tree.label()
                        if '│' in relay_label:
                            sub_tree.set_label(unary_chars + unary_chars + relay_label)
                        else:
                            sub_tree = Tree(unary_chars + '│', [sub_tree])
                else:
                    sub_tree = Tree(label, flatten_children(bottom[start:end]))
                # leave_cnt += len(sub_tree.leaves())
            elif label[0] == '#':
                assert fall_back_root is not None
                if balancing_sub:
                    children = flatten_children(bottom[start:end])
                    if add_weight:
                        children, head_child = flatten_children_with_weights(children, start, layers_of_weights[lid])
                        sub_tree = Tree(label, children)
                    else:
                        sub_tree = Tree(unary_chars + '│', sub_tree)
                else:
                    sub_tree = flatten_children(bottom[start:end])
                # leave_cnt += sum(len(x.leaves()) for x in sub_tree)
            else:
                children = bottom[start:end]
                if add_weight:
                    children, head_child = flatten_children_with_weights(children, start, layers_of_weights[lid])
                    sub_tree = Tree(label, children) # +2
                    try:
                        head_label = head_child[0].label()
                        if mark_np_without_dt_child and label == 'NP' and not any(x[0].label() == 'DT' for x in children):
                            head_label += '*'
                    except:
                        print(head_child[0])
                        import pdb; pdb.set_trace()
                    if '│' in head_label or func_unary_char in head_label:
                        import pdb; pdb.set_trace()
                    if label in headedness_stat:
                        label_cnt, head_cnts = headedness_stat[label]
                    else:
                        label_cnt = 0
                        head_cnts = defaultdict(int)
                    label_cnt += 1
                    head_cnts[head_label] += 1
                    headedness_stat[label] = label_cnt, head_cnts
                else:
                    children = flatten_children(children)
                    sub_tree = Tree(label, children) if label[0] != '_' else children
                # leave_cnt += len(sub_tree.leaves())
            # print(str(sub_tree))
            new_bottom.append(sub_tree)
        # import pdb; pdb.set_trace()
        bottom = new_bottom

        # if leave_cnt != bottom_len:
        #     quit_on_error = True
        # if quit_on_error:
        #     break
        
    if fall_back_root is None:
        tree = bottom.pop()
        assert not bottom
    elif len(bottom) > 1:
        if layers_of_weights and layers_of_weights[lid + 1]: # never be here
            bottom, head_idx = flatten_children_with_weights(bottom, 0, layers_of_weights[lid + 1])
        else:
            bottom = flatten_children(bottom)
        tree = Tree(fall_back_root, bottom)
    else:
        bottom = flatten_children(bottom)
        tree = bottom.pop()
    
    if add_weight_base or add_fence_vote_base:
        tree.un_chomsky_normal_form(unaryChar = func_unary_char)
    else:
        tree.un_chomsky_normal_form()
    if fall_back_root is None:
        return tree
    if add_weight_base:
        return tree, not bottom, headedness_stat
    return tree, not bottom # and not quit_on_error

def draw_str_lines(tree, wrap_len = 1):
    bottom_info, top_down, _ = _read_dpenn(tree)
    return _draw_str_lines(bottom_info, top_down, wrap_len = wrap_len)

class MAryX:
    def __init__(self, tree, word_trace = False):
        if word_trace:
            try:
                preproc_cnf(tree, word_trace = True)
            except:
                print(tree)
        else:
            preproc_cnf(tree)
        tree.collapse_unary(collapseRoot = True)
        self._raw_tree = tree

    @property
    def words(self):
        return self._raw_tree.leaves()

    @property
    def vocabs(self):
        words, tags, labels = (defaultdict(int) for i in range(3))
        for w, t in self._raw_tree.pos():
            words[w] += 1
            tags [t] += 1
        for tree in self._raw_tree.subtrees():
            if tree.height() > 2:
                labels[tree.label()] += 1
        return words, tags, labels

    def signals(self, *vocabs):
        return signals(self._raw_tree, *vocabs)

    def sub_signals(self, *vocabs):
        _, tree = add_efficient_subs(self._raw_tree)
        return signals(tree, *vocabs)

    # def __len__(self):
    #     return len(self._signals[0])