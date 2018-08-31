import numpy as np
import tensorflow as tf
from layers import Layers
from collections import OrderedDict
import copy
import sys
from data_engine import prepare_data
import utils

class Model(object):

    def __init__(self):
        self.layers = Layers()

    def init_params(self, options):
        # all parameters
        params = OrderedDict()
        # tfparams = OrderedDict()
        # embedding
        params['Wemb'] = utils.norm_weight(options['vocab_size'], options['word_dim'])
        # tfparams['Wemb'] = tf.placeholder(tf.float32, shape=(options['vocab_size'], options['word_dim']), name="Wemb")
        # LSTM initial states
        params = self.layers.get_layer('ff')[0](options, params, prefix='ff_state', nin=options['ctx_dim'], nout=options['lstm_dim'])
        params = self.layers.get_layer('ff')[0](options, params, prefix='ff_memory', nin=options['ctx_dim'], nout=options['lstm_dim'])
        # decoder: LSTM
        params = self.layers.get_layer('lstm_cond')[0](options, params, prefix='bo_lstm',
                                                                nin=options['word_dim'], dim=options['lstm_dim'],dimctx=options['ctx_dim'])
        params = self.layers.get_layer('lstm')[0](params, nin=options['lstm_dim'], dim=options['lstm_dim'], prefix='to_lstm')
        # readout
        params = self.layers.get_layer('ff')[0](options, params, prefix='ff_logit_bo', nin=options['lstm_dim'], nout=options['word_dim'])
        if options['ctx2out']:
            params = self.layers.get_layer('ff')[0](options, params, prefix='ff_logit_ctx', nin=options['ctx_dim'], nout=options['word_dim'])
            params = self.layers.get_layer('ff')[0](options, params, prefix='ff_logit_to', nin=options['lstm_dim'], nout=options['word_dim'])
        # MLP
        params = self.layers.get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['word_dim'], nout=options['vocab_size'])
        return params

    def build_model(self, tfparams, options, x, mask, ctx, ctx_mask):
        use_noise = np.float32(0.)
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]
        # get word embeddings
        inputs = tf.nn.embedding_lookup(tfparams['Wemb'], x)    # (num_steps,64,512)
        # count num_frames==28
        counts = tf.expand_dims(tf.reduce_sum(ctx_mask, 1), 1)  # (64,1)
        ctx_ = ctx
        ctx0 = ctx_     # (64,28,2048)
        ctx_mean = tf.reduce_sum(ctx0, 1) / counts  #mean pooling of {vi}   # (64,2048)
        # initial state/cell
        init_state = self.layers.get_layer('ff')[1](tfparams, ctx_mean, options, prefix='ff_state', activ='tanh')   # (64,512)
        init_memory = self.layers.get_layer('ff')[1](tfparams, ctx_mean, options, prefix='ff_memory', activ='tanh') # (64,512)
        # hstltm = self.layers.build_hlstm(['bo_lstm','to_lstm'], inputs, n_timesteps, init_state, init_memory)
        bo_lstm = self.layers.get_layer('lstm_cond')[1](tfparams, inputs, options,
                                                        prefix='bo_lstm',
                                                        mask=mask, context=ctx0,
                                                        one_step=False,
                                                        init_state=init_state,
                                                        init_memory=init_memory,
                                                        use_noise=use_noise)
        return init_state, init_memory, bo_lstm