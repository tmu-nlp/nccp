from data.penn import MultiReader
from data.penn_types import C_ABSTRACT, C_KTB, accp_data_config, select_and_split_corpus
from utils.types import M_TRAIN, M_DEVEL, M_TEST
from utils.param_ops import HParams, get_sole_key
from data.backend import CharTextHelper

from experiments.t_lstm_accp.model import MultiRnnTree, model_type
from experiments.t_lstm_accp.operator import MultiOperator, train_type

get_any_penn = lambda ptb = None, ctb = None, ktb = None: ptb or ctb or ktb
def get_configs(recorder = None):
    if recorder is None:
        return {C_ABSTRACT: accp_data_config}, model_type, train_type
    
    data_config, model_config, train_config, _ = recorder.task_specs()
    penn = HParams(get_any_penn(**data_config), fallback_to_none = True)
    corp_name = get_sole_key(data_config)

    (corpus_reader, get_fnames, _,
     data_splits) = select_and_split_corpus(corp_name,
                                            penn.source_path,
                                            penn.data_splits.train_set,
                                            penn.data_splits.devel_set,
                                            penn.data_splits.test_set)

    model = HParams(model_config)
    reader = MultiReader(penn.data_path,
                         penn.balanced > 0,
                         penn.unify_sub,
                         corpus_reader,
                         get_fnames,
                         data_splits,
                         penn.vocab_size,
                         C_KTB in data_config,
                         CharTextHelper if model.use.char_rnn else None)
    
    def get_datasets(mode):
        datasets = {}
        if mode == M_TRAIN:
            datasets[corp_name] = reader.batch(M_TRAIN, penn.batch_size, penn.bucket_len,
                                               balanced = penn.balanced,
                                               max_len = penn.max_len,
                                               sort_by_length = penn.sort_by_length)
        else:
            datasets[corp_name] = reader.batch(mode, penn.batch_size << 1, 0)
        return datasets

    task_params = ['num_tags', 'num_labels', 'paddings']
    if model.use.word_emb:
        task_params += ['initial_weights', 'num_tokens']
    if model.use.char_rnn:
        task_params.append('num_chars')
    task_params = {pname: reader.get_to_model(pname) for pname in task_params}

    model = MultiRnnTree(**model_config, **task_params)
    model.to(reader.device)
    train_config.create(label_log_freq_inv = reader.frequency('label', log_inv = True))
    return MultiOperator(model, get_datasets, recorder, reader.i2vs, recorder.evalb, train_config)
