
from datetime import datetime
from utils.file_io import join, create_join, listdir, isdir, isfile, remove, rm_rf, rename
from utils.file_io import DelayedKeyboardInterrupt, copy_with_prefix_and_rename, basename
from utils.yaml_io import load_yaml, save_yaml
from utils.param_ops import zip_nt_params, dict_print, change_key, unzip_nt_params
from sys import stderr
from itertools import count
from math import isnan
import torch

_rt_file = 'register_and_tests.yaml'
_rt_lock = 'register_and_tests.lock'
_sv_file = 'settings_and_validation.yaml'
_sv_lock = 'settings_and_validation.lock'

def _rt_file_lock(task_dir):
    rt_file = join(task_dir, _rt_file)
    rt_lock = join(task_dir, _rt_lock)
    return rt_file, rt_lock

def _sv_file_lock(instance_dir):
    sv_file = join(instance_dir, _sv_file)
    sv_lock = join(instance_dir, _sv_lock)
    return sv_file, sv_lock

class Recorder:
    '''A Recorder provides environment for an Operator, created in a Manager, operated by the Operator.'''
    
    def __init__(self, task_dir, task_module, config_dict_or_instance, instance_name = None, keep_top_k = 4, evalb = None, read_only = False):
        # with DelayedKeyboardInterrupt():
        new_instance = isinstance(config_dict_or_instance, dict)
        if read_only:
            assert not new_instance, 'parallelism of optuna trials should be based on a trained instance'
            self._sv_unlock = None # !!! an inactive recorder !!!

        rt_file, rt_lock = _rt_file_lock(task_dir)
        if new_instance:
            rt, unlock = load_yaml(rt_file, rt_lock, True)
            if len(rt):
                name_len = max(len(i) for i in rt.keys())
                inames = tuple(int(i) for i in rt.keys())
                for instance in count():
                    if instance in inames:
                        continue
                    break
            else:
                instance = 0
                name_len = 1
            instance = str(instance)
            if len(instance) < name_len:
                instance = '0' * (name_len - len(instance)) + instance
            rt[instance] = {}
            unlock()
            save_yaml(rt, rt_file, rt_lock) # final confirm
            if instance_name:
                instance_dir = f'{instance}.{instance_name}'
            else:
                instance_dir = instance
            instance_dir = create_join(task_dir, instance_dir)
            config_dict_or_instance['results'] = {}
            sv_file, sv_lock = _sv_file_lock(instance_dir)
            save_yaml(config_dict_or_instance, sv_file, sv_lock)
        else:
            rt = load_yaml(rt_file, rt_lock)
            for instance_dir in listdir(task_dir):
                instance = instance_dir.split('.')[0]
                if instance.isdigit() and int(instance) == int(config_dict_or_instance):
                    break
                instance = None
            assert instance in rt, f'instance {config_dict_or_instance} not registered.'
            instance_dir = create_join(task_dir, instance_dir)
            sv_file, sv_lock = _sv_file_lock(instance_dir)
            assert isfile(sv_file), f"'{sv_file}' is not found."

        self._instance_dir = instance, instance_dir
        self._module     = task_module
        self._ckpt_fname = join(instance_dir, 'checkpoint')
        self._model_dir  = create_join(instance_dir, 'models')
        if not read_only:
            _, self._sv_unlock = load_yaml(sv_file, sv_lock, True)
        self._rt_file_lock = rt_file, rt_lock
        self._sv_file_lock = sv_file, sv_lock
        self._key = None
        self._writer = None
        self._keep_top_k = keep_top_k
        self._evalb = evalb
        self.log(datetime.now())

    def new_trial_recorder(self, specs_update_fn, trial):
        _, instance_dir = self._instance_dir
        specs          = load_yaml(*self._sv_file_lock, wait_lock = False)
        results        = specs.pop('results')
        trial_name     = specs_update_fn(specs, trial)
        best_model     = max(results, key = lambda x: results[x])
        child_recorder = Recorder(create_join(instance_dir, 'trials'), self._module, specs, trial_name, 1, self._evalb)
        _, child_dir   = child_recorder._instance_dir
        self.log(f'New trial {child_dir} from best model {best_model}')
        copy_with_prefix_and_rename(join(instance_dir, 'models', best_model), child_dir, 'checkpoint')
        return child_recorder

    def summary_trials(self): # should only be a super_recorder
        if self._sv_unlock is None: # inactive recorder should only leave to the active one
            return False
        _, instance_dir = self._instance_dir
        children = load_yaml(*_rt_file_lock(instance_dir))
        best_child = max(children, key = lambda cid: children[cid]['key'])
        for fname in listdir(join(instance_dir, 'trials')):
            if '.' in fname:
                thatsit = fname.split('.')[0] == best_child
            else:
                thatsit = fname == best_child
            if thatsit:
                child_specs = load_yaml(*_sv_file_lock(join(instance_dir, 'trials', fname)))
                child_results = child_specs['results']
                best_model = max(child_results, key = lambda x: child_results[x])
                best_fpath = join(instance_dir, 'trials', fname, 'models', best_model)
                
                specs = load_yaml(*self._sv_file_lock, wait_lock = False)
                results = specs['results']
                results[best_model] = child_results[best_model]
                copy_with_prefix_and_rename(best_model, self._model_dir, best_model)
                
                weakest_model = min(results, key = lambda x: results[x])
                remove(join(self._model_dir, weakest_model))
                results.pop(weakest_model)

                self.log(' Replace the worst model', weakest_model, 'with the best model from trial', best_child, best_model)
                save_yaml(specs, *self._sv_file_lock, wait_lock = False)
                return True
        return False

    def detach(self):
        if self._sv_unlock is not None:
            self._sv_unlock()

    def delete_all(self):
        instance, instance_dir = self._instance_dir
        rt = load_yaml(*self._rt_file_lock)
        rt.pop(instance)
        save_yaml(rt, *self._rt_file_lock)
        rm_rf(instance_dir, stderr)

    def delete_most(self):
        instance, instance_dir = self._instance_dir
        remove(join(instance_dir, 'checkpoint'))
        with open(join(instance_dir, 'experiment.log'), 'a+') as fw:
            for fname in listdir(instance_dir):
                fpath = join(instance_dir, fname)
                if isdir(fpath):
                    rm_rf(fpath, fw)

    def log(self, *args, **kwargs):
        _, instance_dir = self._instance_dir
        with open(join(instance_dir, 'experiment.log'), 'a+') as fw:
            kwargs['flush'] = True
            kwargs['file']  = fw
            print(*args, **kwargs)

    def init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from utils.shell_io import byte_style
            Recorder.msg(byte_style('(tensorboard is not installed; not tracking training statistics)', '3'))
            SummaryWriter = None
        if SummaryWriter is not None:
            self._writer = SummaryWriter(self.create_join('train'))

    def tensorboard(self, step, template, **kwargs):
        if self._writer is None:
            return
        for key, value in kwargs.items():
            if value is None: continue
            self._writer.add_scalar(template % key, value, step)

    def tensorboard_histogram(self, step, key, vector):
        if self._writer is None:
            return
        self._writer.add_histogram(key, vector, step)

    @staticmethod
    def msg(*args, **kwargs):
        print(*args, **kwargs, file = stderr, flush = True)

    def task_specs(self): # TODO if not training set trainset & develset to {}
        from utils.param_ops import HParams
        specs = load_yaml(*self._sv_file_lock, wait_lock = False)
        _, model_type, train_type = self._module.get_configs()
        model_config = get_obj_from_config(model_type, specs['model'])
        train_config = get_obj_from_config(train_type, specs['train'])
        train_config = HParams(train_config)
        return specs['data'], model_config, train_config, specs['results']

    def create_join(self, *args):
        _, instance_dir = self._instance_dir
        return create_join(instance_dir, *args)

    def initial_or_restore(self, model, optimizer = None, restore_from_best_validation = False):
        model_fname = None
        if not restore_from_best_validation and isfile(self._ckpt_fname):
            # if not set_vocab(vis_path, r_pu_su[0].py_vocabs, vocab_size):
            # recorder.set_resume_cleaner(lambda mj, mn: clean_epoch(vis_path, mj)) # no mn
            # self._path = vis_path
            # self._init = None
            # # self._pool = []

            # def list_func(self, *token):
            # if self._init is None or self._init == token:
            # if self._init is None:
            # clean_tree_heads(self._path)
            model_fname = self._ckpt_fname

        elif isdir(self._model_dir) or restore_from_best_validation:
            resutls = load_yaml(*self._sv_file_lock, wait_lock = False)['results']
            if resutls:
                best_model = max(resutls, key = lambda x: resutls[x])
                model_fname = join(self._model_dir, best_model)

        if model_fname is None:
            epoch = global_step = 0
            fine_validation = False
            md = dict(model.named_parameters())
            self.log(dict_print(zip_nt_params(md), v_to_str = lambda tensor: '*'.join(str(s) for s in tensor.shape)))
            total = 0
            for t in md.values():
                x = 1
                for s in t.shape:
                    x *= s
                total += x
            self.log('Total:', total)
        else:
            checkpoint = torch.load(model_fname)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                model_old_dict = checkpoint['model_state_dict']
                model_new_dict = model.state_dict()
                new_keys = tuple(model_new_dict)
                from utils.shell_io import byte_style
                for old_key in tuple(model_old_dict):
                    if old_key in new_keys:
                        continue
                    new_candidates = {}
                    old_segs = old_key.split('.')
                    old_segs.reverse()
                    for new_key in new_keys:
                        new_segs = new_key.split('.')
                        new_segs.reverse()
                        if new_segs[0] != old_segs[0]:
                            continue
                        match_depth = 0
                        for ns, os in zip(new_segs, old_segs):
                            if ns == os:
                                match_depth += 1
                        if match_depth > 1:
                            new_candidates[new_key] = match_depth
                    new_candidates = sorted(new_candidates, key = new_candidates.get, reverse = True)
                    if len(new_candidates) == 1:
                        new_key = new_candidates[0]
                        more = len(new_key) - len(old_key)
                        prompt = byte_style('Rename ', '1') # red
                        if more > 0:
                            prompt += ' ' * more
                            prompt += old_key
                            prompt += byte_style('\n    as ', '2') # green
                        else:
                            more = 0 - more
                            prompt += old_key
                            prompt += byte_style('\n    as ', '2') # green
                            prompt += ' ' * more
                        prompt += new_key
                        print(prompt)
                    else:
                        prompt = f'Change {old_key} into:\n'
                        for i, k in enumerate(new_candidates):
                            prompt += f'{i}) {k}\n'
                        new_key = input(prompt)
                        if new_key == 'q':
                            exit()
                        new_key = int(new_key)
                        assert new_key in range(len(new_candidates))
                        new_key = new_candidates[new_key]
                    change_key(model_old_dict, old_key, new_key)
                model.load_state_dict(checkpoint['model_state_dict'])
                decision = input(f'Save change to {model_fname}? [Y]')
                if decision == 'Y':
                    torch.save(checkpoint, model_fname)

            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch, fine_validation, global_step = checkpoint['status']
            self._key = checkpoint['key']
            
            self.log(f"Model restored from", model_fname)
            Recorder.msg(f'Model Restored at {epoch:.2f}, key score {self._key:.2f}')
            if restore_from_best_validation:
                return epoch, global_step
            epoch = int(epoch)
        return epoch, fine_validation, global_step

    def check_betterment(self, epoch, falling, global_step, model, optimizer, key):
        if isnan(key):
            key = float('-inf')
        specs = load_yaml(*self._sv_file_lock, wait_lock = False)
        betterment = (self._key is None or self._key < key)
        in_top_k = any(old_key < key for old_key in specs['results'].values())
        fine_validation = falling and not betterment
        torch.save({'model_state_dict':         model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'status': (epoch, fine_validation, global_step),
                    'key': key}, self._ckpt_fname)
        if betterment or in_top_k:
            if betterment:
                self._key = key
            model_fname = timestamp(epoch)
            copy_with_prefix_and_rename(self._ckpt_fname, self._model_dir, model_fname)
            specs['results'][model_fname] = key
            results = specs['results']
            if len(results) > self._keep_top_k:
                weakest_model = min(results, key = lambda x: results[x])
                remove(join(self._model_dir, weakest_model))
                results.pop(weakest_model)
                self.log(' Replace worst model', weakest_model, 'with a', 'new best' if betterment else 'better', 'model', model_fname)
            else:
                self.log(' A new', 'best' if betterment else 'better', 'model', model_fname)
            save_yaml(specs, *self._sv_file_lock, wait_lock = False)
        else:
            self.log()
        return betterment

    def register_test_scores(self, scores):
        instance, _ = self._instance_dir
        rt = load_yaml(*self._rt_file_lock)
        rt[instance] = scores
        save_yaml(rt, *self._rt_file_lock)

    @staticmethod
    def experiments_status(task_path):
        rt_file = join(task_path, _rt_file)
        rt_lock = join(task_path, _rt_lock)
        (instance_status, unlock), modifed = load_yaml(rt_file, rt_lock, True), False
        status = dict(locking = [], unlocked = [], other = [], tested = [])
        folders = listdir(task_path)

        name_len = 0
        instance_folders = []
        for fx in folders:
            if '.' in fx:
                sep = fx.index('.')
                instance = fx[:sep]
                exp_name = fx[sep+1:]
            else:
                instance = fx
                exp_name = None
            instance_path = join(task_path, fx)
            if isdir(instance_path):
                if instance in instance_status:
                    name_len = max(name_len, len(instance))
                    if isfile(join(instance_path, _sv_lock)):
                        status['locking'].append(instance_path) # avoid ongoing experiments
                    else:
                        instance_folders.append((instance, exp_name, fx, instance_path))
                else:
                    status['other'].append(instance_path)

        rename_list = []
        instance_folders.sort(key = lambda x: int(x[0]))
        for _cnt, (instance, exp_name, folder, fpath) in enumerate(instance_folders):
            _instance = str(_cnt)
            ap_zeros  = name_len - len(_instance)
            _instance = '0' * ap_zeros + _instance
            modify = instance != _instance
            if modify:
                new_folder = f'{_instance}.{exp_name}' if exp_name else _instance
                new_fpath = join(task_path, new_folder)
                change_key(instance_status, instance, _instance)
                rename_list.append((fpath, new_fpath))
                fpath = new_fpath + '\t<- ' + folder
                instance = _instance
                modifed = True
            key = instance_status[instance].get('key')
            if key:
                status['tested'].append(f'({key:.2f})    {fpath}')
            else:
                status['unlocked'].append(f'(?)            {fpath}')

        unlock()
        if modifed:
            save_yaml(instance_status, rt_file, rt_lock)
            for fpath, new_fpath in rename_list:
                rename(fpath, new_fpath)
        return status

    @property
    def evalb(self):
        return self._evalb

    @property
    def key_score(self):
        return self._key

from utils.param_ops import zip_nt_params, iter_zipped_nt_params
def get_obj_from_config(types, configs):
    # import pdb; pdb.set_trace()
    model_params = {}
    for k, vi, vj in iter_zipped_nt_params(types, configs):
        # if vi.is_valid(vj):
        #     model_params[k] = vj
        # else:
        model_params[k] = vi[vj]
    return zip_nt_params(model_params)

def timestamp(main, prefix = 'M'):
    if isinstance(main, str):
        return float(main[1:])
    return f'{prefix}{main:.2f}'