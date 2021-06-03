from os import listdir

types = tuple(f[:-3] for f in listdir('data') if f.endswith('_types.py'))