import pickle
# import compress_pickle

def pickle_load(fname):
    with open(fname, 'rb') as fw:
        return pickle.load(fw)
        # return compress_pickle.load(fw)

def pickle_dump(fname, obj):
    with open(fname, 'wb') as fw:
        pickle.dump(obj, fw)
        # else:
        #     compress_pickle.dump(obj, fw)