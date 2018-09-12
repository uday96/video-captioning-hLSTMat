import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import json, copy
import config
from collections import OrderedDict
import tensorflow as tf
import pickle

def get_rngs(seed=None):
    if seed is None:
        seed = 1234
    else:
        seed = seed
    rng_numpy = np.random.RandomState(seed)
    return rng_numpy

rng_numpy = get_rngs()

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

def write_to_pickle(data,outfname):
    with open(outfname,'wb') as outfile:
        pickle.dump(data, outfile)

def read_from_pickle(infname):
    with open(infname,'rb') as infile:
        data = pickle.load(infile)
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

def flatten_list_of_list(l):
    # l is a list of list
    return [item for sublist in l for item in sublist]

def shuffle_array(array):
	return rng_numpy.shuffle(array)

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

def load_default_params():
	return copy.deepcopy(config.params)

def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def ortho_weight(ndim):
    """
    Random orthogonal weights, we take
    the right matrix in the SVD.

    Remember in SVD, u has the same # rows as W
    and v has the same # of cols as W. So we
    are ensuring that the rows are 
    orthogonal. 
    """
    W = rng_numpy.randn(ndim, ndim)
    u, _, _ = np.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * rng_numpy.randn(nin, nout)
    return W.astype('float32')

def tanh(x):
    return tf.tanh(x)

def rectifier(x):
    return np.maximum(0., x)

def linear(x):
    return x

def _p(pp, name):
    """
    Make prefix-appended name
    """
    return '%s_%s'%(pp, name)

def init_tfparams(params):
    """
    Initialize Tensorflow variables according to the initial parameters
    """
    tfparams = OrderedDict()
    for kk, pp in params.iteritems():
        tfparams[kk] = tf.Variable(params[kk], name=kk)
    return tfparams

# https://github.com/tensorflow/tensorflow/issues/216
def batch_matmul(A, B, transpose_a=False, transpose_b=False):
    '''Batch support for matrix matrix product.

    Args:
        A: General matrix of size (A_Batch, M, X).
        B: General matrix of size (B_Batch, X, N).
        transpose_a: Whether A is transposed (A_Batch, X, M).
        transpose_b: Whether B is transposed (B_Batch, N, X).

    Returns:
        The result of multiplying A with B (A_Batch, B_Batch, M, N).
        Works more efficiently if B_Batch is empty.
    '''
    Andim = len(A.shape)
    Bndim = len(B.shape)
    if Andim == Bndim:
        return tf.matmul(A, B, transpose_a=transpose_a,
                         transpose_b=transpose_b)  # faster than tensordot
    with tf.name_scope('matmul'):
        a_index = Andim - (2 if transpose_a else 1)
        b_index = Bndim - (1 if transpose_b else 2)
        AB = tf.tensordot(A, B, axes=[a_index, b_index])
        if Bndim > 2:  # only if B is batched, rearrange the axes
            A_Batch = np.arange(Andim - 2)
            M = len(A_Batch)
            B_Batch = (M + 1) + np.arange(Bndim - 2)
            N = (M + 1) + len(B_Batch)
            perm = np.concatenate((A_Batch, B_Batch, [M, N]))
            AB = tf.transpose(AB, perm)
    return AB