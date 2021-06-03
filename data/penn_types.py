C_PTB = 'ptb'
C_CTB = 'ctb'
C_KTB = 'ktb'
C_ABSTRACT = 'penn'
E_PENN = C_PTB, C_CTB, C_KTB

from data.io import make_call_fasttext, check_fasttext, check_vocab, split_dict
build_params = {C_PTB: split_dict('2-21',             '22',      '23'    ),
                C_CTB: split_dict('001-270,440-1151', '301-325', '271-300'),
                C_KTB: split_dict('300', '0-9', '10-20')}
                # C_KTB: dict(train_set = 'non_numeric_naming') }
ft_bin = {C_PTB: 'en', C_CTB: 'zh', C_KTB: 'ja'}
call_fasttext = make_call_fasttext(ft_bin)

from utils.types import none_type, false_type, true_type, binarization, NIL, frac_close_0
from utils.types import train_batch_size, train_max_len, train_bucket_len, vocab_size, trapezoid_height
nccp_data_config = dict(vocab_size       = vocab_size,
                        binarization     = binarization,
                        batch_size       = train_batch_size,
                        max_len          = train_max_len,
                        bucket_len       = train_bucket_len,
                        with_ftags       = false_type,
                        unify_sub        = true_type,
                        sort_by_length   = false_type,
                        nil_as_pads      = true_type,
                        trapezoid_height = trapezoid_height)

accp_data_config = dict(vocab_size       = vocab_size,
                        batch_size       = train_batch_size,
                        balanced         = frac_close_0,
                        max_len          = train_max_len,
                        bucket_len       = train_bucket_len,
                        unify_sub        = true_type,
                        sort_by_length   = false_type)

from utils.str_ops import histo_count, str_percentage, strange_to
from utils.pickle_io import pickle_load, pickle_dump
from sys import stderr
from os.path import join, isfile, dirname
from os import listdir
from contextlib import ExitStack
from tqdm import tqdm
from collections import Counter, defaultdict

from nltk.tree import Tree
from tempfile import TemporaryDirectory
from data.io import SourcePool, distribute_jobs
from random import seed

class CorpusReader:
    def __init__(self, path):
        self._path = path

    def break_corpus(self, shuffle_size = 100, rand_seed = 31415926):
        fpath = TemporaryDirectory()
        with ExitStack() as stack:
            files = []
            for i in range(shuffle_size):
                fw = open(join(fpath.name, f'{i:04}'), 'w')
                fw = stack.enter_context(fw)
                files.append(fw)
            seed(rand_seed)
            pool = SourcePool(files, True)
            for ori_file in tqdm(self.fileids(), desc = f'break ktb into {shuffle_size} files in {fpath.name}'):
                for string in self.parsed_sents(ori_file, True):
                    fw = pool()
                    fw.write(string)
            seed(None)
        self._path = fpath

    def fileids(self):
        if isinstance(self._path, str):
            return listdir(self._path)
        return listdir(self._path.name)

    def parsed_sents(self, fileids, keep_str = False):
        def wrap_tree(cumu_string):
            if keep_str:
                return cumu_string
            tree = Tree.fromstring(cumu_string)
            if tree.label() == '':
                tree = tree[0]
            return tree

        if isinstance(self._path, str):
            fpath = join(self._path,      fileids)
        else:
            fpath = join(self._path.name, fileids)
        with open(fpath) as fr:
            cumu_string = None
            for line in fr:
                if not keep_str:
                    line = line.rstrip()
                if line == '':
                    continue
                if line[0] == '(': # end
                    if cumu_string is not None:
                        yield wrap_tree(cumu_string)
                    cumu_string = line
                elif line[0] == '<' or len(line) <= 1: # start or end
                    if cumu_string is None: # not start yet
                        continue
                    yield wrap_tree(cumu_string)
                    cumu_string = None
                elif cumu_string is not None:
                    cumu_string += line
        if cumu_string is not None:
            if keep_str:
                cumu_string += '\n'
            yield wrap_tree(cumu_string)

def positional_iadd(a, b, op = None):
    for ai, bi in zip(a, b):
        for j, bij in enumerate(bi):
            if op: bij = op(bij)
            ai[j] += bij

def reduce_sum(xs):
    res = Counter()
    for i in xs:
        res += i
    return res

def select_and_split_corpus(corp_name, corp_path,
                            train_set, devel_set, test_set):
    from nltk.corpus import BracketParseCorpusReader
    if corp_name == C_PTB:
        folder_pattern = lambda x: f'{x:02}'
        reader = BracketParseCorpusReader(corp_path, r".*/wsj_.*\.mrg")
        def get_id(fpath):
            bar = fpath.index('/') # 23/xxx_0000.xxx
            fid = fpath[:bar]
            sid = fpath[bar+7:-4]
            return fid, sid
        get_fnames = lambda data_split: [fn for fn in reader.fileids() if dirname(fn) in data_split]
    elif corp_name == C_CTB:
        folder_pattern = lambda x: f'{x:04}'
        reader = CorpusReader(corp_path)
        get_id = lambda fpath: (fpath[5:-3], '-')
        get_fnames = lambda data_split: [fn for fn in reader.fileids() if fn[5:-3] in data_split]
    elif corp_name == C_KTB:
        folder_pattern = lambda x: f'{x:04}'
        reader = CorpusReader(corp_path)
        reader.break_corpus(int(train_set))
        train_set = '__rest__'
        get_id = lambda fpath: (fpath, '-')
        get_fnames = lambda data_split: [fn for fn in reader.fileids() if fn in data_split]

    devel_set = strange_to(devel_set, folder_pattern)
    test_set  = strange_to(test_set,  folder_pattern)
    if train_set == '__rest__':
        corpus = reader.fileids()
        non_train_set = set(devel_set + test_set)
        if corp_name == C_CTB:
            train_set = [fid[5:-3] for fid in corpus if fid not in non_train_set]
        else:
            train_set = [fid for fid in corpus if fid not in non_train_set]
    else:
        train_set = strange_to(train_set, folder_pattern)
    return reader, get_fnames, get_id, (train_set, devel_set, test_set)

def build(save_to_dir,
          corp_path,
          corp_name,
          train_set,
          devel_set,
          test_set,
          **kwargs):
    from multiprocessing import Process, Queue # seamless with threading.Thread
    from data.delta import DeltaX, bottom_up_ftags, xtype_to_logits, lnr_order, OriFct
    from utils.types import E_ORIF4, O_LFT, O_RGT, M_TRAIN, M_DEVEL, M_TEST, num_threads
    from itertools import count
    from time import sleep

    reader, get_fnames, get_id, (train_set, devel_set, test_set) = select_and_split_corpus(corp_name, corp_path, train_set, devel_set, test_set)

    assert any(t not in devel_set for t in train_set)
    assert any(tv not in test_set for tv in train_set + devel_set)
    corpus = set(train_set + devel_set + test_set)
    corpus = get_fnames(corpus)

    x2l = lambda xl: tuple(xtype_to_logits(x) for x in xl)
    class WorkerX(Process):
        def __init__(self, *args):
            Process.__init__(self)
            self._args = args

        def run(self):
            q, reader, fns, get_id = self._args
            inst_cnt   = 0
            unary_cnt  = defaultdict(list)
            train_length_cnt = defaultdict(int)
            valid_length_cnt = defaultdict(int)
            test_length_cnt  = defaultdict(int)
            non_train_set = devel_set + test_set
            lrcs = [[0, 0] for _ in E_ORIF4]
            word_trace = corp_name == C_KTB

            def stat_is_unary(sent_len, fid, sid):
                if sent_len < 2:
                    unary_cnt[fid].append(sid)
                    return True
                elif fid in devel_set:
                    valid_length_cnt[sent_len] += 1
                elif fid in test_set:
                    test_length_cnt [sent_len] += 1
                else:
                    train_length_cnt[sent_len] += 1
                return False

            for fn in fns:
                fid, sid = get_id(fn)

                for tree in reader.parsed_sents(fn):
                    if len(tree.leaves()) < 2:
                        unary_cnt[fid].append(sid)
                        continue

                    if any(len(b) > 2 for b in tree.subtrees()):
                        dxs, lrs = DeltaX.from_penn_quad(tree, word_trace = word_trace)
                        if stat_is_unary(len(tree.leaves()), fid, sid):
                            continue
                        ws, ps = dxs[0].word_tag()
                        ss, xs, fs, xs_ = [], [], [], []
                        for dx in dxs:
                            s, x, f = dx.to_triangles()
                            ss.append(s); xs.append(x); fs.append(f); xs_.append(x2l(x));
                        ss = OriFct(*ss); xs = OriFct(*xs); fs = OriFct(*fs); xs_ = OriFct(*xs_);
                        assert all(f == fs[0] for f in fs[1:])
                        assert all(sum(lrs[0]) == sum(c) for c in lrs[1:])
                        t = fid, 2, ws, ps, ss, xs, bottom_up_ftags(fs[0]), xs_
                    else:
                        dx, lr = DeltaX.from_penn(tree, 'left', word_trace = word_trace)
                        if stat_is_unary(len(tree.leaves()), fid, sid):
                            continue
                        ws, ps     = dx.word_tag()
                        ss, xs, fs = dx.to_triangles()
                        t = fid, 1, ws, ps, ss, xs, bottom_up_ftags(fs), x2l(xs)
                        lrs = OriFct(lr, lr, lr, lr)

                    positional_iadd(lrcs, lrs)
                    q.put(t)
                    inst_cnt += 1

            q.put((inst_cnt, unary_cnt, train_length_cnt, valid_length_cnt, test_length_cnt, lrcs))

    num_threads = min(num_threads, len(corpus))
    workers = distribute_jobs(corpus, num_threads)
    q = Queue()
    for i in range(num_threads):
        w = WorkerX(q, reader, workers[i], get_id)
        w.start()
        workers[i] = w

    from utils.file_io import create_join
    from data.io import save_vocab, sort_count
    tok_cnt, pos_cnt, ftag_cnt  = Counter(), Counter(), Counter()
    xty_cnts = [Counter() for _ in E_ORIF4]
    syn_cnts = [Counter() for _ in E_ORIF4]
    lrcs = [[0, 0] for _ in E_ORIF4]
    train_word_cnt = Counter()
    unary_counters = []
    train_length_cnt, valid_length_cnt, test_length_cnt = Counter(), Counter(), Counter()
    cnf_diff = [0, 0, 0]
    thread_join_cnt = 0
    with ExitStack() as stack, tqdm(desc = f'  Receiving samples from {num_threads} threads') as qbar:
        ftw  = stack.enter_context(open(join(save_to_dir, M_TRAIN + '.word'), 'w'))
        ftp  = stack.enter_context(open(join(save_to_dir, M_TRAIN + '.tag'),  'w'))
        ftf  = stack.enter_context(open(join(save_to_dir, M_TRAIN + '.ftag'), 'w'))
        fts  = stack.enter_context(open(join(save_to_dir, M_TRAIN + '.finc'), 'w'))
        fvw  = stack.enter_context(open(join(save_to_dir, M_DEVEL + '.word'), 'w'))
        fvp  = stack.enter_context(open(join(save_to_dir, M_DEVEL + '.tag'),  'w'))
        fvf  = stack.enter_context(open(join(save_to_dir, M_DEVEL + '.ftag'), 'w'))
        fvs  = stack.enter_context(open(join(save_to_dir, M_DEVEL + '.finc'), 'w'))
        f_w  = stack.enter_context(open(join(save_to_dir, M_TEST  + '.word'), 'w'))
        f_p  = stack.enter_context(open(join(save_to_dir, M_TEST  + '.tag'),  'w'))
        f_f  = stack.enter_context(open(join(save_to_dir, M_TEST  + '.ftag'), 'w'))
        f_s  = stack.enter_context(open(join(save_to_dir, M_TEST  + '.finc'), 'w'))
        ftxs = [stack.enter_context(open(join(save_to_dir, f'{M_TRAIN}.xtype.{o}'), 'w')) for o in E_ORIF4]
        ftls = [stack.enter_context(open(join(save_to_dir, f'{M_TRAIN}.label.{o}'), 'w')) for o in E_ORIF4]
        fvxs = [stack.enter_context(open(join(save_to_dir, f'{M_DEVEL}.xtype.{o}'), 'w')) for o in E_ORIF4]
        fvls = [stack.enter_context(open(join(save_to_dir, f'{M_DEVEL}.label.{o}'), 'w')) for o in E_ORIF4]
        f_xs = [stack.enter_context(open(join(save_to_dir, f'{M_TEST}.xtype.{o}' ), 'w')) for o in E_ORIF4]
        f_ls = [stack.enter_context(open(join(save_to_dir, f'{M_TEST}.label.{o}' ), 'w')) for o in E_ORIF4]

        for instance_cnt in count(): # Yes! this edition works fine!
            if q.empty():
                sleep(0.01)
            else:
                t = q.get()
                if len(t) == 8:
                    fid, num_directions, ws, ps, ss, dr, ft, xs = t
                    if fid in devel_set:
                        fw, fp, ff, fs, fxs, fls = fvw, fvp, fvf, fvs, fvxs, fvls
                        ftag_p = 1
                    elif fid in test_set:
                        fw, fp, ff, fs, fxs, fls = f_w, f_p, f_f, f_s, f_xs, f_ls
                        ftag_p = 2
                    else:
                        fw, fp, ff, fs, fxs, fls = ftw, ftp, ftf, fts, ftxs, ftls
                        ftag_p = 0
                        train_word_cnt += Counter(ws)
                    # for vocabulary, get all the tokens for fasttext
                    tok_cnt += Counter(ws) # why? namespace::
                    pos_cnt += Counter(ps) # because w/r namespaces are different
                    fw.write(' '.join(ws) + '\n')
                    fp.write(' '.join(ps) + '\n')
                    if num_directions == 1:
                        ftag_cnt += Counter(ft[0])
                        ff.write(' '.join(ft[0]) + '\n')
                        fs.write(' '.join(ft[1]) + '\n')
                        for dc, sc, fx, fl in zip(xty_cnts, syn_cnts, fxs, fls):
                            dc += Counter(dr)
                            sc += Counter(ss)
                            fx.write(' '.join(xs) + '\n')
                            fl.write(' '.join(ss) + '\n')
                    else:
                        cnf_diff[ftag_p] += 1
                        ftag_cnt += Counter(ft[0])
                        ff.write(' '.join(ft[0]) + '\n')
                        fs.write(' '.join(ft[1]) + '\n')

                        for fx, x, fl, l, dc, di, sc, si in zip(fxs, xs, fls, ss, xty_cnts, dr, syn_cnts, ss):
                            fx.write(' '.join(x) + '\n')
                            fl.write(' '.join(l) + '\n')
                            dc += Counter(di)
                            sc += Counter(si)
                    qbar.update(1)
                elif len(t) == 6:
                    thread_join_cnt += 1
                    ic, uc, tlc, vlc, _lc, lrbc = t
                    if qbar.total:
                        qbar.total += ic
                    else:
                        qbar.total = ic
                    qbar.desc = f'  {thread_join_cnt} of {num_threads} threads ended with {qbar.total} samples, receiving'

                    unary_counters.append(uc)
                    train_length_cnt.update(tlc)
                    valid_length_cnt.update(vlc)
                    test_length_cnt .update(_lc)
                    positional_iadd(lrcs, lrbc)
                    if thread_join_cnt == num_threads:
                        break
                else:
                    raise ValueError('Unknown data: %r' % t)
        for w in workers:
            w.join()

    totals = [sum(c.values()) for c in (train_length_cnt, valid_length_cnt, test_length_cnt)]
    cnf_diff[0] /= totals[0]
    cnf_diff[1] /= totals[1]
    cnf_diff[2] /= totals[2]

    print(f'Length distribution in [ Train set ] ({totals[0]}, cnf_diff = {str_percentage(cnf_diff[0])})', file = stderr)
    print(histo_count(train_length_cnt, bin_size = 10), file = stderr)
    print(f'Length distribution in [ Dev set ]   ({totals[1]}, cnf_diff = {str_percentage(cnf_diff[1])})', file = stderr)
    print(histo_count(valid_length_cnt, bin_size = 10), file = stderr)
    print(f'Length distribution in [ Test set ]  ({totals[2]}, cnf_diff = {str_percentage(cnf_diff[2])})', file = stderr)
    print(histo_count(test_length_cnt, bin_size = 10), file = stderr)

    unary_info = ''
    if unary_counters:
        unary_bose_logger = defaultdict(list)
        for uc in unary_counters:
            for fid, ulist in uc.items():
                unary_bose_logger[fid].extend(ulist)
        for fid, ROX in sorted(unary_bose_logger.items(), key = lambda x:x[0]):
            ROX = Counter(ROX)
            ROX = (f'{k}:{v}' if v > 1 else k for k,v in ROX.items())
            unary_info += fid + f"({','.join(ROX)});"
        print("Unary:", unary_info, file = stderr)
    pickle_dump(join(save_to_dir, 'info.pkl'), dict(tlc = train_length_cnt, vlc = valid_length_cnt, _lc = test_length_cnt, unary = unary_info, cnf = cnf_diff))

    left, right = E_ORIF4.index(O_LFT), E_ORIF4.index(O_RGT)
    syn_left_cnt  = syn_cnts[left]
    syn_right_cnt = syn_cnts[right]
    if syn_left_cnt.keys() != syn_right_cnt.keys():
        raise ValueError(f'Invalid penn_treebank data: left CNF label set != right CNF label set, please use full data!')
    # assert '<js' not in xty_left_cnt and '>js' not in xty_right_cnt, 'CNF processing is dirty, please check'
    xty_left_cnt  = xty_cnts[left]
    xty_right_cnt = xty_cnts[right]
    assert xty_left_cnt ['<js'] < xty_left_cnt ['>js'] # weaker assertion: bottom layer has attach the 'j' xtype as default.
    assert xty_right_cnt['<js'] > xty_right_cnt['>js'] # high probability
    xty_cnt = reduce_sum(xty_cnts)
    syn_cnt = reduce_sum(syn_cnts)
    if corp_name != C_KTB: # ktb contains a great amount of labels in tags!!!
        assert all(p not in syn_cnt for p in pos_cnt) and any(s[1:] in pos_cnt for s in syn_cnt), 'check # option in preproc'
    xty_op = lambda x: {f'{k}({xtype_to_logits(k)})':v for k,v in x.items()}
    xty_cnt       = xty_op(xty_cnt)
    xty_left_cnt  = xty_op(xty_left_cnt)
    xty_right_cnt = xty_op(xty_right_cnt)

    tok_file = join(save_to_dir, 'vocab.word')
    pos_file = join(save_to_dir, 'vocab.tag' )
    xty_file = join(save_to_dir, 'vocab.xtype')
    syn_file  = join(save_to_dir, 'vocab.label')
    ftag_file = join(save_to_dir, 'vocab.ftag')
    ts, vs = save_vocab(tok_file, tok_cnt, [NIL] + sort_count(train_word_cnt))
    _,  ps = save_vocab(pos_file, pos_cnt, [NIL])
    _,  ss = save_vocab(syn_file, syn_cnt, [NIL])
    _,  fs = save_vocab(ftag_file, ftag_cnt)
    _,  xs = save_vocab(xty_file, xty_cnt, lnr_order(xty_cnt)[0])
    for o, xcs, scs, lbc in zip(E_ORIF4, xty_cnts, syn_cnts, lrcs):
        xcs['ling-lb'] = lbc[0]
        xcs['ling-rb'] = lbc[1]
        save_vocab(join(save_to_dir, f'stat.xtype.{o}'), xcs, lnr_order(xcs)[0])
        save_vocab(join(save_to_dir, f'stat.label.{o}'), scs, lnr_order(scs)[0])
    return (ts, vs, ps, xs, ss, fs)

def check_data(save_dir, valid_sizes):
    try:
        # 44386 47074 47 8 99 20
        ts, vs, ps, xs, ss, fs = valid_sizes
        if ts > vs:
            raise ValueError(f'Train vocab({ts}) should be less than corpus vocab({vs})')
    except Exception as e:
        print(e, file = stderr)
        return False
    valid_sizes = vs, ps, xs, ss, fs
    vocab_files = 'vocab.word vocab.tag vocab.xtype vocab.label vocab.ftag'.split()
    x = all(check_vocab(join(save_dir, vf), vs) for vf, vs in zip(vocab_files, valid_sizes))

    fname = join(save_dir, 'info.pkl')
    if isfile(fname):
        info = pickle_load(fname)
        totals = (sum(info['tlc'].values()), sum(info['vlc'].values()), sum(info['_lc'].values()))
        print('Total:', sum(totals), file = stderr)
        print(f'Length distribution in [ Train set ] ({totals[0]})', file = stderr)
        print(histo_count(info['tlc'], bin_size = 10), file = stderr)
        print(f'Length distribution in [ Dev set ]   ({totals[1]})', file = stderr)
        print(histo_count(info['vlc'], bin_size = 10), file = stderr)
        print(f'Length distribution in [ Test set ]  ({totals[2]})', file = stderr)
        print(histo_count(info['_lc'], bin_size = 10), file = stderr)

    return x