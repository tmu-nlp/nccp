from utils.str_ops import strange_to

def check_select(select):
    if ':' in select: # 3/ptb:annotate
        i = select.index(':')
        exp_name = select[i+1:]
        select = select[:i]
    else:
        exp_name = None

    if '/' in select:
        i = select.index('/')
        select, corp_name = select.split('/')
    else:
        corp_name = None
    return select, corp_name, exp_name

def check_instances_operation(instance):
    op_code = instance and instance[0].isalpha() and instance[0]
    exp_ids = instance[1:] if op_code else instance
    exp_ids = strange_to(exp_ids, str) if exp_ids else [exp_ids]
    return op_code, exp_ids

def check_train(train_str):
    # fv=4:30:4,max=100,!
    train = dict(test_with_validation = False,
                 fine_validation_at_nth_wander = 10,
                 stop_at_nth_wander = 100,
                 fine_validation_each_nth_epoch = 4,
                 update_every_n_batch = 1,
                 optuna_trials = 0,
                 max_epoch = 1000)
    assert ' ' not in train_str
    for group in train_str.split(','):
        if group.startswith('fine='):
            group = [int(x) if x else 0 for x in group[5:].split(':')]
            assert 1 <= len(group) <= 3
            if group[0]:
                train['fine_validation_at_nth_wander'] = group[0]
            if len(group) > 1 and group[1]:
                train['stop_at_nth_wander'] = group[1]
            if len(group) > 2 and group[2]:
                train['fine_validation_each_nth_epoch'] = group[2]

        elif group.startswith('max='):
            train['max_epoch'] = int(group[4:])

        elif group.startswith('update='):
            train['update_every_n_batch'] = int(group[7:])
    
        elif group == '!':
            train['test_with_validation'] = True

        elif group.startswith('optuna='):
            train['optuna_trials'] = int(group[7:])

        elif group:
            raise ValueError('Unknown training param:' + group)
    return train