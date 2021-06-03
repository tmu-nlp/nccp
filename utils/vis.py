from multiprocessing import Process, Queue
from time import sleep, time

class VisWorker(Process): # designed for decoding batch
    def __init__(self, vis, in_q, out_q):
        super().__init__()
        self._vio = vis, in_q, out_q

    def run(self):
        vis, in_q, out_q = self._vio
        vis._before()
        proc_time = 0
        while True:
            if in_q.empty():
                sleep(0.001)
                continue
            inp = in_q.get()
            if inp is None:
                break
            args, kw_args = inp
            start      = time()
            try:
                vis._process(*args, **kw_args)
            except Exception as err:
                if hasattr(vis, 'close'):
                    vis.close()
                raise err
            proc_time += time() - start
        out_q.put((vis._after(), vis._attrs, proc_time))

class BaseVis:
    def __init__(self, epoch):
        self._epoch = epoch
        self._attrs = {}
        
    def _before(self):
        raise NotImplementedError()

    def _process(self, *args, **kw_args):
        raise NotImplementedError()

    def _after(self):
        raise NotImplementedError()

    @property
    def epoch(self):
        return self._epoch

    def register_property(self, attr_name, value):
        self._attrs[attr_name] = value

    def __getattr__(self, attr_name):
        return self._attrs[attr_name]

class VisRunner:
    def __init__(self, vis, async_ = False):
        self._vis   = vis
        self._async = async_
        self._timer = 0
        
    def before(self):
        if self._async:
            iq, oq = Queue(), Queue()
            worker = VisWorker(self._vis, iq, oq)
            self._async = worker, iq, oq
            worker.start()
        else:
            self._vis._before()
        self._duration = time()

    def process(self, *args, **kw_args):
        if self._async:
            _, iq, _ = self._async
            iq.put((args, kw_args))
        else:
            start = time()
            try:
                self._vis._process(*args, **kw_args)
            except Exception as err:
                if hasattr(self._vis, 'close'):
                    self._vis.close()
                raise err
            
            self._timer += time() - start

    def after(self):
        self._duration = time() - self._duration
        if self._async:
            worker, iq, oq = self._async
            iq.put(None) # end while loop | _after
            out, attrs, proc_time = oq.get()
            worker.join() # this should be after oq.get()
            self._timer = proc_time
            self._vis._attrs.update(attrs)
            return out
        return self._vis._after()

    def __del__(self):
        if isinstance(self._async, tuple):
            self._async[0].terminate()

    @property
    def proc_time(self):
        return self._timer

    @property
    def before_after_duration(self):
        return self._duration

    @property
    def is_async(self):
        return isinstance(self._async, tuple)

    def __getattr__(self, attr_name):
        return getattr(self._vis, attr_name)