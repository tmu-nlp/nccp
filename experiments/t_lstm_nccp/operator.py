import torch
from torch import nn
from utils.operator import Operator
from data.delta import get_rgt, get_dir, s_index
from utils.param_ops import get_sole_key
from time import time
from utils.math_ops import is_bin_times
from utils.types import M_TRAIN, BaseType, frac_open_0, true_type, false_type, tune_epoch_type, frac_06
from models.utils import PCA, fraction, hinge_score
from models.loss import binary_cross_entropy, hinge_loss
from experiments.helper import WarmOptimHelper
from utils.shell_io import byte_style

train_type = dict(loss_weight = dict(tag    = BaseType(0.2, validator = frac_open_0),
                                     label  = BaseType(0.3, validator = frac_open_0),
                                     orient = BaseType(0.5, validator = frac_open_0)),
                  learning_rate = BaseType(0.001, validator = frac_open_0),
                  label_freq_as_loss_weight = false_type,
                  multiprocessing_decode = true_type,
                  orient_hinge_loss = true_type,
                  tune_pre_trained = dict(from_nth_epoch = tune_epoch_type,
                                          lr_factor = frac_06))

class PennOperator(Operator):
    def __init__(self, model, get_datasets, recorder, i2vs, evalb, train_config):
        super().__init__(model, get_datasets, recorder, i2vs)
        self._evalb = evalb
        self._sigmoid = nn.Sigmoid()
        self._mode_length_bins = None, None
        self._initial_run = True, True
        self._train_config = train_config
        self._tune_pre_trained = False
        self._mp_decode = False
        self._dm = None

    def _build_optimizer(self, start_epoch):
        # self._loss_weights_of_tag_label_orient = 0.3, 0.1, 0.6 betas = (0.9, 0.98), weight_decay = 0.01, eps = 1e-6
        self._mp_decode = self._train_config.multiprocessing_decode
        self._schedule_lr = hp = WarmOptimHelper.adam(self._model, self._train_config.learning_rate)
        if start_epoch > 0:
            fpath = self.recorder.create_join('penn_devel')
            PennOperator.clean_and_report(fpath, start_epoch)
        self.recorder.init_tensorboard()
        optim = hp.optimizer
        optim.zero_grad()
        return optim

    def _schedule(self, epoch, wander_ratio):
        tune = self._train_config.tune_pre_trained.from_nth_epoch
        self._tune_pre_trained = tune = tune is not None and tune < epoch
        lr_factor = self._train_config.tune_pre_trained.lr_factor if tune else 1
        learning_rate = self._schedule_lr(epoch, wander_ratio, lr_factor)
        self.recorder.tensorboard(self.global_step, 'Batch/%s', Learning_Rate = learning_rate, Epoch = epoch)

    def _step(self, mode, ds_name, batch, batch_id = None):

        # assert ds_name == C_ABSTRACT
        gold_orients = get_rgt(batch['xtype'])
        if mode == M_TRAIN:
            batch['supervised_orient'] = gold_orients
            #(batch['offset'], batch['length'])

        batch_time = time()
        (batch_size, batch_len, static, top3_label_logits,
         layers_of_base, _, existences, orient_logits, tag_logits, label_logits,
         trapezoid_info) = self._model(batch['token'], self._tune_pre_trained, **batch)
        batch_time = time() - batch_time

        orient_logits.squeeze_(dim = 2)
        existences   .squeeze_(dim = 2)
        if self._train_config.orient_hinge_loss:
            orients = orient_logits > 0
        else:
            orient_logits = self._sigmoid(orient_logits)
            orients = orient_logits > 0.5

        if mode == M_TRAIN:
            tags    = self._model.get_decision(tag_logits  )
            labels  = self._model.get_decision(label_logits)
            bottom_existence = existences[:, -batch_len:]
            orient_weight = get_dir(batch['xtype'])
            tag_mis       = (tags    != batch['tag'])
            label_mis     = (labels  != batch['label'])
            orient_match  = (orients == gold_orients) & orient_weight
            tag_weight    = (   tag_mis | bottom_existence)
            label_weight  = ( label_mis | existences)

            if self._train_config.label_freq_as_loss_weight:
                label_mask = self._train_config.label_log_freq_inv[batch['label']]
            else:
                label_mask = None

            if trapezoid_info is None:
                height_mask = s_index(batch_len - batch['length'])[:, None, None]
            else:
                height_mask = batch['mask_length'] # ?? negative effect ???

            tag_loss, label_loss = self._model.get_losses(batch, tag_logits, top3_label_logits, label_logits, height_mask, label_mask)

            if self._train_config.orient_hinge_loss:
                orient_loss = hinge_loss(orient_logits, gold_orients, orient_weight)
            else:
                orient_loss = binary_cross_entropy(orient_logits, gold_orients, orient_weight)

            total_loss = self._train_config.loss_weight.tag * tag_loss
            total_loss = self._train_config.loss_weight.label * label_loss + total_loss
            total_loss = self._train_config.loss_weight.orient * orient_loss + total_loss
            total_loss.backward()
            
            if hasattr(self._model, 'tensorboard'):
                self._model.tensorboard(self.recorder, self.global_step)
            self.recorder.tensorboard(self.global_step, 'Accuracy/%s',
                                      Tag    = 1 - fraction(tag_mis,     tag_weight),
                                      Label  = 1 - fraction(label_mis, label_weight),
                                      Orient = fraction(orient_match, orient_weight))
            self.recorder.tensorboard(self.global_step, 'Loss/%s',
                                      Tag    = tag_loss,
                                      Label  = label_loss,
                                      Orient = orient_loss,
                                      Total  = total_loss)
            batch_kwargs = dict(Length = batch_len, SamplePerSec = batch_len / batch_time)
            if 'segment' in batch:
                batch_kwargs['Height'] = len(batch['segment'])
            self.recorder.tensorboard(self.global_step, 'Batch/%s', **batch_kwargs)
        else:
            vis, _, _, serial, c_vis = self._vis_mode
            if serial:
                pca = self._model.get_static_pca() if hasattr(self._model, 'get_static_pca') else None
                if pca is None:
                    pca = PCA(layers_of_base.reshape(-1, layers_of_base.shape[2]))
                mpc_token = pca(static)
                mpc_label = pca(layers_of_base)

                # if c_vis is not None:
                #     batch['supervised_orient'] = gold_orients
                #     (_, _, _, _, gold_base, _, _, _, _, _,
                #     _) = self._model(batch['token'], False, **batch)
                #     l_pca = pca(gold_base)
                #     label_embeddings = {}
                #     interesting_labels = 'NP PP VP S+VP S'.split()
                #     for lbl in interesting_labels:
                #         lid = self.i2vs.label.index(lbl)
                #         embs = l_pca[torch.where(batch['label'] == lid)]
                #         label_embeddings[lbl] = embs.type(torch.float16).cpu().numpy()
                #     c_vis.process(label_embeddings)
                b_head = tuple(batch[x].type(torch.uint8) if x in ('tag', 'label') else batch[x] for x in 'offset length token tag label'.split())
                b_head = b_head + (gold_orients,)

                tag_scores,   tags   = self._model.get_decision_with_value(tag_logits)
                label_scores, labels = self._model.get_decision_with_value(label_logits)
                tags = tags.type(torch.uint8)
                labels = labels.type(torch.uint8)
                if self._train_config.orient_hinge_loss: # otherwise with sigmoid
                    hinge_score(orient_logits, inplace = True)
                b_mpcs = (None if mpc_token is None else mpc_token.type(torch.float16), mpc_label.type(torch.float16))
                b_scores = (tag_scores.type(torch.float16), label_scores.type(torch.float16), orient_logits.type(torch.float16))
                b_data = (tags, labels, orients) + b_mpcs + b_scores
                if trapezoid_info is not None:
                    d_seg, d_seg_len = trapezoid_info
                    trapezoid_info = batch['segment'], batch['seg_length'], d_seg, d_seg_len.cpu().numpy()
            else:
                b_head = tuple(batch[x] for x in 'offset length token'.split())
                tags   = self._model.get_decision(tag_logits  ).type(torch.uint8)
                labels = self._model.get_decision(label_logits).type(torch.uint8)
                if self._mp_decode:
                    b_data = (tags, labels, orients)
                    if trapezoid_info is not None:
                        d_seg, d_seg_len = trapezoid_info
                        trapezoid_info = d_seg, d_seg_len.cpu().numpy()
                else:
                    b_head = b_head + (None, None, None)
                    b_data = (tags, labels, orients, None, None, None, None, None)
                    if trapezoid_info is not None:
                        d_seg, d_seg_len = trapezoid_info
                        trapezoid_info = batch['segment'], batch['seg_length'], d_seg, d_seg_len.cpu().numpy()

            b_size = (batch_id, batch_size, batch_len)
            tensors = b_size + b_head + b_data
            tensors = tuple(x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in tensors)
            vis.process(tensors, trapezoid_info)
        return batch_size, batch_len

    def _before_validation(self, ds_name, epoch, use_test_set = False, final_test = False):
        devel_bins, test_bins = self._mode_length_bins
        devel_init, test_init = self._initial_run
        epoch_major, epoch_minor = epoch.split('.')
        if use_test_set:
            if final_test:
                folder = ds_name + '_test'
                scores_of_bins = save_tensors = True
            else:
                folder = ds_name + '_test_with_devel'
                save_tensors = is_bin_times(int(epoch_major)) if int(epoch_minor) == 0 else False
                scores_of_bins = False
            length_bins = test_bins
            flush_heads = test_init
            self._initial_run = devel_init, False
        else:
            folder = ds_name + '_devel'
            length_bins = devel_bins
            save_tensors = is_bin_times(int(epoch_major)) if int(epoch_minor) == 0 else False
            scores_of_bins = False
            flush_heads = devel_init
            self._initial_run = False, test_init

        if hasattr(self._model, 'update_static_pca'):
            self._model.update_static_pca()
        work_dir = self.recorder.create_join(folder)
        serial = save_tensors or flush_heads or not self._mp_decode
        if serial:
            async_ = True
            vis = SerialVis(epoch,
                            work_dir,
                            self._evalb,
                            self.i2vs,
                            self.recorder.log,
                            save_tensors,
                            length_bins,
                            scores_of_bins,
                            flush_heads)
        else:
            async_ = False
            vis = ParallelVis(epoch, work_dir, self._evalb, self.i2vs, self.recorder.log, self._dm)
        vis = VisRunner(vis, async_ = async_) # wrapper
        vis.before()
        if final_test:
            c_vis = ScatterVis(epoch, work_dir)
            c_vis = VisRunner(c_vis, async_ = True)
            c_vis.before()
        else:
            c_vis = None
        self._vis_mode = vis, use_test_set, final_test, serial, c_vis

    def _after_validation(self, ds_name, count, seconds):
        vis, use_test_set, final_test, serial, c_vis = self._vis_mode
        scores, desc, logg, dm = vis.after()
        if dm and self._dm is None:
            self._dm = dm
        length_bins = vis.length_bins
        devel_bins, test_bins = self._mode_length_bins
        if length_bins is not None:
            if use_test_set:
                self._mode_length_bins = devel_bins, length_bins # change test
            else:
                self._mode_length_bins = length_bins, test_bins # change devel
        speed_outer = float(f'{count / seconds:.1f}')
        speed_inner = float(f'{count / vis.proc_time:.1f}') # unfolded with multiprocessing
        if vis.is_async:
            rate = vis.proc_time / seconds
        else:
            rate = vis.proc_time / (seconds - vis.proc_time)
        if serial:
            dmt = speed_dm = ''
        else:
            dmt = self._dm.duration
            speed_dm = f' ◇ {count / dmt:.1f}'
            dmt = f' ◇ {dmt:.3f}'
            desc += byte_style(speed_dm + 'sps.', '2')

        logg += f' @{speed_outer} ◇ {speed_inner}{speed_dm} sps. (sym:nn {rate:.2f}; {seconds:.3f}{dmt} sec.)'
        scores['speed'] = speed_outer
        if final_test:
            if self._dm:
                self._dm.close()
        else:
            self.recorder.tensorboard(self.global_step, 'TestSet/%s' if use_test_set else 'DevelSet/%s',
                                      F1 = scores.get('F1', 0), SamplePerSec = speed_outer)
        self._vis_mode = None
        if c_vis is not None:
            c_vis.after()
        return scores, desc, logg

    @staticmethod
    def combine_scores_and_decide_key(epoch, ds_scores):
        scores = ds_scores[get_sole_key(ds_scores)]
        scores['key'] = scores.get('F1', 0)
        return scores

    @staticmethod
    def clean_and_report(fpath, start_epoch):
        removed = remove_vis_data_from(fpath, start_epoch)
        if removed:
            if len(removed) == 1:
                content = removed[0]
            else:
                content = f'{len(removed)} files'
            Operator.msg(f' [{start_epoch:.2f}:] {content} removed in folder penn_devel.')

        fpath = fpath.replace('penn_devel', 'penn_test_with_devel')
        if isdir(fpath):
            removed = remove_vis_data_from(fpath, start_epoch)
            if removed:
                if len(removed) == 1:
                    content = removed[0]
                else:
                    content = f'{len(removed)} files'
                Operator.msg(f' [{start_epoch:.2f}:] {content} removed in folder penn_test_with_devel.')

    def optuna_model(self):
        pass


from utils.vis import BaseVis, VisRunner
from utils.file_io import join, isfile, listdir, remove, isdir
from utils.pickle_io import pickle_dump
from utils.param_ops import HParams
from utils.shell_io import parseval, rpt_summary
from visualization import ContinuousTensorVis
class SerialVis(BaseVis):
    def __init__(self, epoch, work_dir, evalb, i2vs, logger,
                 save_tensors   = True,
                 length_bins    = None,
                 scores_of_bins = False,
                 flush_heads    = False):
        super().__init__(epoch)
        self._evalb = evalb
        fname = join(work_dir, 'vocabs.pkl')
        # import pdb; pdb.set_trace()
        if flush_heads and isfile(fname):
            remove(fname)
        self._ctvis = ContinuousTensorVis(work_dir, i2vs)
        self._logger = logger
        htree = join(work_dir, 'head.tree')
        dtree = join(work_dir, f'data.{epoch}.tree')
        self._fnames = htree, dtree
        self._head_tree = None
        self._data_tree = None
        self._scores_of_bins = scores_of_bins
        self.register_property('save_tensors', save_tensors)
        self.register_property('length_bins',  length_bins)

    def __del__(self):
        if self._head_tree: self._head_tree.close()
        if self._data_tree: self._data_tree.close()

    def _before(self):
        htree, dtree = self._fnames
        if self._ctvis.is_anew: # TODO
            self._head_tree = open(htree, 'w')
            self.register_property('length_bins', set())
        self._data_tree = open(dtree, 'w')

    def _process(self, batch, trapezoid_info):
        # process batch to instances, compress
        # if head is in batch, save it
        # else check head.emm.pkl
        # make data.emmb.tree & concatenate to data.emm.tree
        # make data.emm.rpt
        (batch_id, _, size, h_offset, h_length, h_token, h_tag, h_label, h_right,
         d_tag, d_label, d_right, mpc_token, mpc_label,
         tag_score, label_score, split_score) = batch
        d_trapezoid_info = None
        if trapezoid_info:
            segment, seg_length, d_segment, d_seg_length = trapezoid_info
            trapezoid_info = segment, seg_length
            d_trapezoid_info = d_segment, d_seg_length

        # if h_tag is None:
        #     import pdb; pdb.set_trace()

        if self._head_tree:
            bins = self._ctvis.set_head(self._head_tree, h_offset, h_length, h_token, h_tag, h_label, h_right, trapezoid_info, batch_id, size, 10)
            self.length_bins |= bins

        if self.save_tensors:
            if self.length_bins is not None and self._scores_of_bins:
                bin_width = 10
            else:
                bin_width = None
            extended = size, bin_width, self._evalb
        else:
            extended = None

        self._ctvis.set_data(self._data_tree, self._logger, batch_id, self.epoch,
                             h_offset, h_length, h_token, d_tag, d_label, d_right,
                             mpc_token, mpc_label,
                             tag_score, label_score, split_score,
                             d_trapezoid_info,
                             extended) # TODO go async

    def _after(self):
        # call evalb to data.emm.rpt return the results, and time counted
        # provide key value in results
        if self._head_tree:
            self._head_tree.close()
        self._data_tree.close()
        proc = parseval(self._evalb, *self._fnames)
        report = proc.stdout.decode()
        scores = rpt_summary(report, False, True)
        errors = proc.stderr.decode().split('\n')
        assert errors.pop() == ''
        num_errors = len(errors)
        if num_errors:
            self._logger(f'  {num_errors} errors from evalb')
            if num_errors < 10:
                for e, error in enumerate(errors):
                    self._logger(f'    {e}. ' + error)
                fname = f'data.{self.epoch}.rpt'
                with open(self._ctvis.join(fname), 'w') as fw:
                    fw.write(report)
                self._logger(f'  Go check {fname} for details.')

        self._head_tree = self._data_tree = None

        if self.length_bins is not None and self._scores_of_bins:
            with open(self._ctvis.join(f'{self.epoch}.scores'), 'w') as fw:
                fw.write('wbin,num,lp,lr,f1,ta\n')
                for wbin in self.length_bins:
                    fhead = self._ctvis.join(f'head.bin_{wbin}.tree')
                    fdata = self._ctvis.join(f'data.bin_{wbin}.tree')
                    proc = parseval(self._evalb, fhead, fdata)
                    smy = rpt_summary(proc.stdout.decode(), False, True)
                    fw.write(f"{wbin},{smy['N']},{smy['LP']},{smy['LR']},{smy['F1']},{smy['TA']}\n")
                    remove(fhead)
                    remove(fdata)

        desc = f'Evalb({scores["LP"]:.2f}/{scores["LR"]:.2f}/'
        key_score = f'{scores["F1"]:.2f}'
        desc_for_screen = desc + byte_style(key_score, underlined = True) + ')'
        desc_for_logger = f'N: {scores["N"]} {desc}{key_score})'
        return scores, desc_for_screen, desc_for_logger, None

class ScatterVis(BaseVis):
    def __init__(self, epoch, work_dir, dim = 10):
        super().__init__(epoch)
        self._work_dir = work_dir
        self._fname = dim


    def _before(self):
        line = 'label,' + ','.join(f'pc{i}' if i else 'mag' for i in range(self._fname)) + '\n'
        fname = join(self._work_dir, f'pca.{self.epoch}.csv')
        self._fname = fname
        with open(fname, 'w') as fw:
            fw.write(line)

    def _process(self, label_embeddings):
        with open(self._fname, 'a+') as fw:
            for label, embeddings in label_embeddings.items():
                for emb in embeddings:
                    fw.write(label + ',' + ','.join(f'{e:.3f}' for e in emb) + '\n')

    def _after(self):
        pass


from data.triangle import TriangularDM
from data.trapezoid import TrapezoidalDM
from utils.types import num_threads
class ParallelVis(BaseVis):
    def __init__(self, epoch, work_dir, evalb, i2vs, logger, dm):
        super().__init__(epoch)
        self._join = lambda fname: join(work_dir, fname)
        self._fdata = self._join(f'data.{self.epoch}.tree')
        self._args = evalb, i2vs, logger, dm

    def _before(self):
        _, _, _, dm = self._args
        if dm: dm.timeit()

    def _process(self, batch, d_trapezoid_info):
        (_, batch_size, _, h_offset, h_length, h_token,
         d_tag, d_label, d_right) = batch
        evalb, i2vs, logger, dm = self._args
        
        if d_trapezoid_info:
            if dm is None:
                dm = TrapezoidalDM(batch_size, i2vs, num_threads)
                self._args = evalb, i2vs, logger, dm
            d_segment, d_seg_length = d_trapezoid_info
            dm.batch(d_segment, h_offset, h_length, h_token, d_tag, d_label, d_right, d_seg_length)
        else:
            if dm is None:
                dm = TriangularDM(batch_size, i2vs, num_threads)
                self._args = evalb, i2vs, logger, dm
            dm.batch(h_offset, h_length, h_token, d_tag, d_label, d_right)
    
    def _after(self):
        evalb, i2vs, logger, dm = self._args
        fhead = self._join(f'head.tree')
        fdata = self._fdata

        tree_text = dm.batched()
        if tree_text: # none mean text concat without a memory travel
            with open(fdata, 'w') as fw:
                fw.write(tree_text)

        proc = parseval(evalb, fhead, fdata)
        report = proc.stdout.decode()
        scores = rpt_summary(report, False, True)
        errors = proc.stderr.decode().split('\n')
        assert errors.pop() == ''
        num_errors = len(errors)
        if num_errors:
            logger(f'  {num_errors} errors from evalb')
            if num_errors < 10:
                for e, error in enumerate(errors):
                    logger(f'    {e}. ' + error)
                fname = f'data.{self.epoch}.rpt'
                with open(self._join(fname), 'w') as fw:
                    fw.write(report)
                logger(f'  Go check {fname} for details.')

        desc = f'Evalb({scores["LP"]:.2f}/{scores["LR"]:.2f}/'
        key_score = f'{scores["F1"]:.2f}'
        desc_for_screen = desc + byte_style(key_score, underlined = True) + ')'
        desc_for_logger = f'N: {scores["N"]} {desc}{key_score})'
        return scores, desc_for_screen, desc_for_logger, dm

    @property
    def save_tensors(self):
        return False

    @property
    def length_bins(self):
        return None

# an example of Unmatched Length from evalb
# head
# (S (S (VP (VBG CLUBBING) (NP (DT A) (NN FAN)))) (VP (VBD was) (RB n't) (NP (NP (DT the) (NNP Baltimore) (NNP Orioles) (POS ')) (NN fault))) (. .))
# (S (NP (NP (JJ CLUBBING) (NNP A)) ('' FAN)) (VP (VBD was) (PP (RB n't) (NP     (DT the) (NNP Baltimore) (NNS Orioles) (POS ') (NN fault)))) (. .))
# data

def remove_vis_data_from(fpath, start_epoch):
    removed = []
    for fname in listdir(fpath):
        if fname.startswith('data.'):
            if fname.endswith('.tree'): # batch | epoch
                if '.bin_' in fname:
                    batch_or_epoch = fname[5:fname.find('.bin_')] # data.[].bin_xx.tree
                else:
                    batch_or_epoch = fname[5:-5] # data.[].tree
                if '.' in batch_or_epoch and float(batch_or_epoch) >= start_epoch:
                    remove(join(fpath, fname))
                    removed.append(fname)
            elif fname.endswith('.pkl') or fname.endswith('.rpt'): # batch_epoch
                epoch = fname[5:-4]
                if '_' in epoch:
                    epoch = epoch[epoch.index('_') + 1:]
                if float(epoch) >= start_epoch:
                    remove(join(fpath, fname))
                    removed.append(fname)
    return removed
