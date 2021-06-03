from multiprocessing import Process, Queue, TimeoutError
from math import ceil
from time import time, sleep
from data.backend import before_to_seq
from os.path import join, dirname
from utils.shell_io import concatenate

t_sleep_deep = 0.1
t_sleep_shallow = 0.001

class D2T(Process):
    def __init__(self, idx, in_q, out_q, vocabs, tree_gen_fn, cat_dir):
        super().__init__()
        self._id_q_vocabs_fn = idx, in_q, out_q, vocabs, tree_gen_fn, cat_dir

    def run(self):
        idx, in_q, out_q, (i2w, i2t, i2l), tree_gen_fn, cat_dir = self._id_q_vocabs_fn
        t_sleep = t_sleep_shallow
        last_wake = time()
        while True:
            while in_q.empty():
                sleep(t_sleep)
                if time() - last_wake > 5:
                    t_sleep = t_sleep_deep
                else:
                    t_sleep = t_sleep_shallow
                continue
            signal = in_q.get()
            last_wake = time()
            if signal is None:
                out_q.put(idx)
                continue
            elif isinstance(signal, int):
                if signal < 0:
                    break
            key, tensor_args = signal
            tree_gen = tree_gen_fn(i2w, i2t, i2l, *tensor_args)
            if cat_dir:
                fname = join(cat_dir, 'mp.%d_%d.tree' % key)
                with open(fname, 'w') as fw:
                    fw.write('\n'.join(tree_gen))
                out_q.put((key, fname))
            else:
                out_q.put((key, list(tree_gen)))
            last_wake = time()
            

class DM:
    @staticmethod
    def tree_gen_fn():
        raise NotImplementedError()

    @staticmethod
    def arg_segment_fn(seg_id, seg_size, batch_size, args):
        raise NotImplementedError()

    def __init__(self, batch_size, vocabs, num_workers, fdata = None, cat_files = False):
        rin_q = Queue()
        rout_q = Queue()
        self._q_receiver = rin_q, rout_q, None
        fpath = dirname(fdata) if fdata and cat_files else None

        vocabs = before_to_seq(vocabs._nested)
        q_workers = []
        for seg_id in range(num_workers):
            in_q = Queue()
            d2t = D2T(seg_id, in_q, rin_q, vocabs, self.tree_gen_fn, fpath)
            d2t.start()
            q_workers.append((in_q, d2t))
        self._mp_workers = q_workers, ceil(batch_size / num_workers), batch_size, fdata, cat_files
        self._batch_id = 0
        self._timer = time()

    def timeit(self):
        self._timer = time()

    def batch(self, *args): # split a batch for many workers
        q_workers, seg_size, batch_size, fdata, cat_files = self._mp_workers
        # import pdb; pdb.set_trace()
        rin_q, rout_q, tr = self._q_receiver
        if tr is None:
            tr = TR(rin_q, rout_q, [False for _ in q_workers], fdata, cat_files)
            tr.start()
            self._q_receiver = rin_q, rout_q, tr
            
        for seg_id, (in_q, _) in enumerate(q_workers):
            major_args = self.arg_segment_fn(seg_id, seg_size, batch_size, args)
            if major_args:
                in_q.put(((self._batch_id, seg_id), major_args))
        self._batch_id += 1

    def batched(self):
        q_workers, _, _, fdata, _ = self._mp_workers
        for in_q, _ in q_workers:
            in_q.put(None)
        rin_q, rout_q, tr = self._q_receiver
        trees_time = rout_q.get()
        if fdata:
            trees = None
        else:
            trees, trees_time = trees_time # trees
        self._timer = trees_time - self._timer
        tr.join()
        self._q_receiver = rin_q, rout_q, None
        self._batch_id = 0
        return trees

    @property
    def duration(self):
        return self._timer

    def close(self):
        q_workers, _, _, _, _ = self._mp_workers
        for in_q, d2t in q_workers:
            in_q.put(-1)
            try:
                d2t.join(timeout = 0.5)
            except TimeoutError:
                print('terminate', d2t)
                d2t.terminate()


class TR(Process):
    def __init__(self, in_q, out_q, checklist, fdata, cat_files, flatten_batch = True):
        super().__init__()
        self._q = in_q, out_q, checklist, fdata, cat_files, flatten_batch

    def run(self):
        i_trees = {}
        in_q, out_q, checklist, fdata, cat_files, flatten_batch = self._q
        t_sleep = t_sleep_shallow
        last_wake = time()
        while True:
            while in_q.empty():
                sleep(t_sleep)
                if time() - last_wake > 5:
                    t_sleep = t_sleep_deep
                else:
                    t_sleep = t_sleep_shallow
                continue
            signal = in_q.get()
            last_wake = time()
            if isinstance(signal, int):
                checklist[signal] = True # idx
                # print(checklist)
                if all(checklist):
                    break
                continue
            key, trees = signal
            i_trees[key] = trees
            last_wake = time()
        end_time = time()
            
        # print(sorted(i_trees))
        if fdata:
            if cat_files:
                cat_files = []
                for key in sorted(i_trees):
                    cat_files.append(i_trees[key])
                concatenate(cat_files, fdata)
            else:
                with open(fdata, 'w') as fw:
                    for key in sorted(i_trees):
                        fw.write('\n'.join(i_trees[key]) + '\n')
            out_q.put(end_time)
        else:
            trees = []
            for key in sorted(i_trees):
                bid, _ = key
                # print(key, len(i_trees[key]))
                if flatten_batch:
                    trees.extend(i_trees[key])
                elif bid == len(trees):
                    trees.append(i_trees[key])
                else:
                    trees[bid].extend(i_trees[key])
            trees = '\n'.join(trees) if flatten_batch else trees
            out_q.put((trees, end_time))