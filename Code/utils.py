import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import json

def set_rngs(seed=None):
    if seed is None:
        seed = 1234
    else:
        seed = seed
    np.random.seed(seed)

def read_csv_data(fname):
	return pd.read_csv(fname,dtype=str)

def read_dir_files(dir):
	return [f for f in listdir(dir) if isfile(join(dir, f))]

def read_dir(dir):
    return [f for f in listdir(dir)]

def write_to_json(data,outfname):
	with open(outfname,'w') as outfile:
		json.dump(data, outfile,indent=4)

def read_from_json(infname):
	with open(infname,'r') as infile:
		data = json.load(infile)
		return data

def read_file_to_list(fname):
	with open(fname,"r") as f:
		data = []
		for l in f.readlines():
			if l[-1]=='\n':
				data.append(l[:-1])
			else:
				data.append(l)
		return data

def write_list_to_file(fname,data_list):
	file = open(fname,"w")
	for data in data_list:
		file.write(data+"\n")
	file.close()

def shuffle_array(array):
	set_rngs()
	return np.random.shuffle(array)

def generate_minibatch_idx(dataset_size, minibatch_size):
    # generate idx for minibatches SGD
    # output [m1, m2, m3, ..., mk] where mk is a list of indices
    assert dataset_size >= minibatch_size
    n_minibatches = dataset_size / minibatch_size
    leftover = dataset_size % minibatch_size
    idx = range(dataset_size)
    if leftover == 0:
        minibatch_idx = np.split(np.asarray(idx), n_minibatches)
    else:
        print('uneven minibatch chunking, overall %d, last one %d'%(minibatch_size, leftover))
        minibatch_idx = np.split(np.asarray(idx)[:-leftover], n_minibatches)
        minibatch_idx = minibatch_idx + [np.asarray(idx[-leftover:])]
    minibatch_idx = [idx_.tolist() for idx_ in minibatch_idx]
    return minibatch_idx