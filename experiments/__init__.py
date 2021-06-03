from itertools import product
from utils.param_ops import filter_flatten
from os import listdir
from os.path import isdir, join

types = tuple(f for f in listdir('experiments') if f.startswith('t_') and isdir(join('experiments', f)))