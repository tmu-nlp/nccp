from utils.types import M_TRAIN, M_DEVEL, M_TEST
from numpy.random import choice
from tqdm import tqdm
from time import time
from datetime import timedelta
from torch import nn, no_grad
from utils.recorder import Recorder, timestamp
from utils.emoji import get_train_validation_pair
import torch

class Operator:
    '''An (abstract) Operator operate a customized nn.Module for training, validation and testing.
    To operator, it feeds the model with multi-tasking batch from the customized get_datasets function,
    uses the environmental Recorder to record the results of the model, and i2vs to help a Vis to visualize them.'''
    def __init__(self, model, get_datasets, recorder, i2vs):
        assert isinstance(model, nn.Module)
        assert callable(get_datasets)
        assert isinstance(recorder, Recorder)
        assert 'token' in i2vs._nested
        self._model = model
        self._get_datasets = get_datasets
        self._recorder = recorder
        self._i2vs = i2vs
        self._optimizer = None
        self._train_materials = None
        self._validate_materials = None
        self._test_materials = (_, ds_names, _) = self._get_materials(M_TEST)
        self._ds_icons = {ds_name: icon for ds_name, icon in zip(ds_names, '⚀⚁⚂⚃⚄⚅')}
        self._optuna_mode = None
        self._epoch_start = time()

    def _get_materials(self, mode):
        ds_specs = self._get_datasets(mode)
        ds_specs = ((dn,) + ds for dn, ds in ds_specs.items())
        ds_names, ds_freqs, ds_iters = zip(*ds_specs)
        ds_total = sum(ds_freqs)
        return ds_total, ds_names, ds_iters

    def train_initials(self, with_train_set):
        if self._train_materials is None: # single run
            train_icon, devel_icon = get_train_validation_pair()
            self._validate_materials = self._get_materials(M_DEVEL), devel_icon
            self._train_materials    = self._get_datasets(M_TRAIN) if with_train_set else None, train_icon
        else: # from optuna
            train_cnf, train_icon = self._train_materials
            self._train_materials = self._get_datasets(M_TRAIN, train_cnf), train_icon

        (epoch, fine_validation, global_step) = self._recorder.initial_or_restore(self._model)
        self._optimizer = self._build_optimizer(epoch)
        self._global_step = global_step
        return epoch, fine_validation

    def train_step(self, epoch_cnt, wander_ratio, update_every_n_batch = 1):
        ds_specs, train_icon = self._train_materials
        ds_freqs = {dn: ds.size       for dn, ds in ds_specs.items()}
        ds_iters = {dn: iter(ds.iter) for dn, ds in ds_specs.items()}
        with tqdm(total = sum(ds_freqs.values()), desc = train_icon) as qbar:
            while sum(ds_freqs.values()):
                # prepare datasets for joint tasks
                total = sum(ds_freqs.values())
                ds_names, ds_probs = zip(*((dn, df/total) for dn, df in ds_freqs.items()))
                ds_name = choice(ds_names, p = ds_probs)
                ds_icon = self._ds_icons[ds_name] if len(ds_names) > 1 else ''
                batch = next(ds_iters[ds_name])

                self._schedule(epoch_cnt + qbar.n / qbar.total, wander_ratio)
                with torch.autograd.set_detect_anomaly(True):
                    num_samples, seq_len = self._step(M_TRAIN, ds_name, batch) # neural core
                if self._global_step % update_every_n_batch == update_every_n_batch - 1:
                    self._optimizer.step()
                    self._optimizer.zero_grad()

                # display
                qbar.update(num_samples)
                from_start = timedelta(seconds = int(time() - self._epoch_start))
                qbar.desc = f'[{epoch_cnt}] {100*wander_ratio:.0f}% {train_icon} {from_start} {ds_icon}:{num_samples}×{seq_len}'
                ds_freqs[ds_name] -= num_samples
                self._global_step += 1

                updated_wander_ratio = yield qbar.n / qbar.total
                if updated_wander_ratio is not None:
                    wander_ratio = updated_wander_ratio
            qbar.desc = f'[{epoch_cnt}] {100*wander_ratio:.0f}% {train_icon}'
        # next epoch

    def validate_betterment(self, epoch, falling):
        (ds_total, ds_names, ds_iters), devel_icon = self._validate_materials
        scores, ds_logg, from_start = self.validation_or_test(M_DEVEL, ds_total, ds_names, ds_iters, devel_icon, epoch)
        self._recorder.log(timestamp(epoch, 'Validation ') + f' - {ds_logg} ({from_start} from start)', end = '.')
        if self._optuna_mode is not None:
            super_recorder, trial, trial_step = self._optuna_mode
            trial.report(scores['key'], trial_step)
            if trial.should_prune():
                import optuna
                self._recorder.register_test_scores(dict(key = self._recorder.key_score, step = trial_step))
                super_recorder.log(f'  (got pruned at the {trial_step}-th step)')
                raise optuna.exceptions.TrialPruned()
            self._optuna_mode = super_recorder, trial, trial_step + 1
        return self._recorder.check_betterment(epoch, falling, self._global_step, self._model, self._optimizer, scores['key'])

    def test_model(self, epoch = None):
        ds_total, ds_names, ds_iters = self._test_materials
        final_test = epoch is None
        if final_test:
            if self._optuna_mode is not None:
                super_recorder, trial, trial_step = self._optuna_mode
                super_recorder.log(f'  (finished after {trial_step + 1} steps)')
                self._recorder.register_test_scores(dict(key = self._recorder.key_score, step = trial_step))
                return # dead end: optuna_mode should not nest
            prefix = 'Test ' # final label
            epoch, self._global_step = self._recorder.initial_or_restore(self._model, restore_from_best_validation = True)
        else:
            prefix = '   ⌙→ Test ' # match length of validation
        scores, ds_logg, from_start = self.validation_or_test(M_TEST, ds_total, ds_names, ds_iters, '🔮', epoch, final_test)
        scores['epoch'] = epoch
        self._recorder.log(timestamp(epoch, prefix) + f' - {ds_logg} ({from_start} from start).')
        if final_test:
            if hasattr(self._model, 'message'):
                message = self._model.message
                if message:
                    self._recorder.log(message)
            return dict(scores)

    def validation_or_test(self, mode, ds_total, ds_names, ds_iters, icon, epoch, final_test = False):
        ds_desc   = []
        ds_logg   = []
        ds_scores = {}
        self._model.eval() # stack
        epoch_stamp = timestamp(epoch, '')
        with tqdm(total = ds_total, desc = f'#{icon}{epoch_stamp}') as qbar:
            for ds_name, ds_iter in zip(ds_names, ds_iters):
                self._before_validation(ds_name, f'{epoch:08.2f}', mode == M_TEST, final_test)
                start, cnt = time(), 0
                for batch_id, batch in enumerate(ds_iter):
                    with no_grad():
                        num_samples, seq_len = self._step(mode, ds_name, batch, batch_id = batch_id)
                    cnt += num_samples
                    qbar.desc = f'#{icon}{epoch_stamp} {num_samples}×{seq_len}'
                    qbar.update(num_samples)
                scores, desc, logg = self._after_validation(ds_name, cnt, time() - start) # evalb time is excluded
                ds_desc  .append(desc)
                ds_logg  .append(logg)
                ds_scores[ds_name] = scores
            from_start = timedelta(seconds = int(time() - self._epoch_start))
            qbar.total = None
            qbar.desc = f'[{epoch_stamp}] {icon} ' + ' '.join(ds_desc)
            ds_logg = '\n'.join(ds_logg)
        self._model.train() # restore
        scores = self.combine_scores_and_decide_key(epoch, ds_scores)
        return scores, ds_logg, from_start

    def setup_optuna_mode(self, spec_update_fn, trial):
        if self._optuna_mode is None:
            super_recorder = self._recorder
        else:
            self._recorder.detach() # previous trail
            super_recorder = self._optuna_mode[0]
        self._optuna_mode = super_recorder, trial, 0
        self._recorder = super_recorder.new_trial_recorder(spec_update_fn, trial)
        return super_recorder.key_score

    def restore_recorder(self):
        assert self._optuna_mode
        self._recorder.detach() # previous trail
        self._recorder = self._optuna_mode[0]
        self._optuna_mode = None
        self._recorder.summary_trials()

    def optuna_model(self, train_params):
        '''Set objective function with argument trial.
        It should include a starting checkpoint from the best model,
        a training process with multiple trials: each with checkpoint-i
        and models folers. Only two trial folders is perserved:
        current and last_trial. Each folder shares the framework with this
        operator and recorder relationship, so as it will reuse the train_ops.
        The child operator and recorder will be a little bit different.
        import optuna

        def obj_fn(trial):
            def spec_update_fn(specs):
                x = trial.suggest_int('x', 2, 20)
                y = int(trial.suggest_float('y', 1, 32, log=True))
                specs['.....x'] = x
                specs['.....y'] = y
                return f'x={x},y={y}'

            child_recorder = self._recorder.trial(spec_update_fn) # a new working folder with checkpoint from models/Mbest
            base_devel_score = self.setup_optuna_mode(child_recorder, trial)
            train(train_params, self) # change train_params
            return child_recorder.key_score

        study = optuna.create_study(direction = 'maximize')
        study.optimize(obj_fn, n_trials = 100)
        self.restore_recorder()

        trial = study.best_trial
        print('Accuracy: {}'.format(trial.value))
        print("Best hyperparameters: {}".format(trial.params))
        optuna.visualization.plot_optimization_history(study)
        optuna.visualization.plot_slice(study)
        optuna.visualization.plot_contour(study, params=['n_estimators', 'max_depth'])
        '''

    def _schedule(self, epoch, wander_ratio):
        pass

    def _step(self, mode, ds_name, batch, flush = True, batch_id = None):
        raise NotImplementedError()

    def _build_optimizer(self, start_epoch):
        raise NotImplementedError()

    def _before_validation(self, ds_name, epoch, use_test_set, final_test):
        raise NotImplementedError()

    def _after_validation(self, ds_name, count, seconds):
        raise NotImplementedError()

    @property
    def recorder(self):
        return self._recorder

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def i2vs(self):
        return self._i2vs

    @property
    def global_step(self):
        return self._global_step
    
    @staticmethod
    def combine_scores_and_decide_key(epoch, ds_scores):
        raise NotImplementedError('Should manually decide the key')

    @staticmethod
    def msg(*args, **kw_args):
        Recorder.msg(*args, **kw_args)


class CsvWriter:
    def __init__(self, fpath):
        self._file_headers = open(fpath, 'a+'), None

    def write(self, outputs):
        fw, headers = self._file_headers
        if headers is None:
            headers = tuple(outputs.keys())
            self._file_header = fw, headers
            fw.write(','.join(headers) + '\n')
        fw.write(','.join(str(outputs[h]) for h in headers) + '\n')