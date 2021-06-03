from data.penn import PennReader
from data.penn_types import C_ABSTRACT, C_KTB, nccp_data_config, select_and_split_corpus
from utils.types import M_TRAIN, M_DEVEL, M_TEST
from utils.param_ops import HParams, get_sole_key
from utils.shell_io import byte_style
from data.backend import CharTextHelper

from experiments.t_lstm_nccp.model import ContinuousRnnTree, model_type
from experiments.t_lstm_nccp.operator import PennOperator, train_type

get_any_penn = lambda ptb = None, ctb = None, ktb = None: ptb or ctb or ktb
def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: nccp_data_config}, model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    penn = HParams(get_any_penn(**data_config), fallback_to_none = True)
    train_cnf     = penn.binarization._nested
    non_train_cnf = {max(train_cnf, key = lambda x: train_cnf[x]): 1}
    corp_name = get_sole_key(data_config)

    if penn.trapezoid_height:
        specs = select_and_split_corpus(corp_name,
                                        penn.source_path,
                                        penn.data_splits.train_set,
                                        penn.data_splits.devel_set,
                                        penn.data_splits.test_set)
        data_splits = {k:v for k,v in zip((M_TRAIN, M_DEVEL, M_TEST), specs[-1])}
        trapezoid_specs = specs[:-1] + (data_splits, penn.trapezoid_height, get_sole_key(data_config) == C_KTB)
        prompt = f'Use trapezoidal data (stratifying height: {penn.trapezoid_height})', '2'
    else:
        trapezoid_specs = None
        prompt = f'Use triangular data (stratifying height: +inf)', '3'
    print(byte_style(*prompt))

    model = HParams(model_config)
    reader = PennReader(penn.data_path,
                        penn.vocab_size,
                        True, # load_label
                        penn.unify_sub,
                        penn.with_ftags,
                        penn.nil_as_pads,
                        trapezoid_specs,
                        CharTextHelper if model.use.char_rnn else None)
    
    def get_datasets(mode):
        datasets = {}
        if mode == M_TRAIN:
            datasets[corp_name] = reader.batch(M_TRAIN, penn.batch_size, penn.bucket_len, train_cnf,
                                                max_len = penn.max_len, sort_by_length = penn.sort_by_length)
        else:
            datasets[corp_name] = reader.batch(mode, penn.batch_size << 1, 0, non_train_cnf)
        return datasets

    task_params = ['num_tags', 'num_labels', 'paddings']
    if model.use.word_emb:
        task_params += ['initial_weights', 'num_tokens']
    if model.use.char_rnn:
        task_params.append('num_chars')
    task_params = {pname: reader.get_to_model(pname) for pname in task_params}

    model = ContinuousRnnTree(**model_config, **task_params)
    model.to(reader.device)
    train_config.create(label_log_freq_inv = reader.frequency('label', log_inv = True))
    return PennOperator(model, get_datasets, recorder, reader.i2vs, recorder.evalb, train_config)
        
# def get_datasets_for_tagging(ptb = None, ctb = None, ktb = None):
#     if not (ptb or ctb or ktb):
#         return dict(penn = none_type)

#     datasets = {}
#     penn = ptb or ctb or ktb
#     reader = PennReader(penn['data_path'], False)
    
#     if M_TRAIN in mode_keys:
#         datasets[M_TRAIN] = reader.batch(M_TRAIN, 100, 20, max_len = 100)
#     if M_DEVEL in mode_keys:
#         datasets[M_DEVEL]  = reader.batch(M_DEVEL, 60, 20, max_len = 100)
#     if M_TEST in mode_keys:
#         datasets[M_TEST]  = reader.batch(M_TEST, 60, 20, max_len = 100)
#     return datasets, reader

