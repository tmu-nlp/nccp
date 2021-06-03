import yaml
from os import remove
from os.path import isfile
from datetime import datetime
from utils.shell_io import byte_style

def _wait(lfile):
    if isfile(lfile):
        msg = 'Wait for '
        msg += byte_style(f"'{lfile}'", '3')
        msg += '\n  (Locked at '
        with open(lfile) as fr:
            for line in fr:
                msg += line.split('.')[0]
        msg += ') Unlock? [Y or any key to exit] '
        if input(msg) != 'Y':
            # print('\nexited')
            exit()
        remove(lfile)

def _block(lfile):
    with open(lfile, 'w') as fw:
        fw.write(str(datetime.now()))

def _unblock(lfile):
    if isfile(lfile):
        remove(lfile)
        return True
    return False

def save_yaml(status, mfile, lfile, wait_lock = True):
    if lfile:
        if wait_lock:
            _wait(lfile)
            _block(lfile)

        finished = False
        do_exit = None
        while not finished:
            try:
                with open(mfile, 'w') as fw:
                    fw.write(f'# {datetime.now()}\n')
                    yaml.dump(status, fw, default_flow_style = False)
                finished = True
            except KeyboardInterrupt as e:
                do_exit = e
                print('suppress', e, 'for saving', mfile)
        # release
        if wait_lock:
            _unblock(lfile)
            if do_exit is not None:
                raise do_exit
    else:
        with open(mfile, 'a+') as fw:
            fw.write(f'# {datetime.now()}\n')
            yaml.dump(status, fw, default_flow_style = False)
    return True

def load_yaml(mfile, lfile, block = False, wait_lock = True):
    if isfile(mfile):
        if wait_lock:
            _wait(lfile)
            if block:
                _block(lfile)
    else:
        save_yaml({}, mfile, lfile)
    with open(mfile, 'r') as fr:
        status = yaml.load(fr, Loader = yaml.FullLoader)
    if not status:
        status = {}
    if wait_lock and block:
        return status, lambda : _unblock(lfile)
    return status