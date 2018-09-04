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
        use_noise = tf.Variable(False, dtype=tf.bool)
        x_shape = tf.shape(x)
        n_timesteps = x_shape[0]
        n_samples = x_shape[1]
        # get word embeddings
        inputs = tf.nn.embedding_lookup(tfparams['Wemb'], x)    # (num_steps,64,512)
        # count num_frames==28
        counts = tf.expand_dims(tf.reduce_sum(ctx_mask, 1), 1)  # (64,1)
        ctx_ = ctx
        ctx0 = ctx_     # (64,28,2048)
        ctx_mean = tf.reduce_sum(ctx0, axis=1) / counts  #mean pooling of {vi}   # (64,2048)
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
        to_lstm = self.layers.get_layer('lstm')[1](tfparams, bo_lstm[0],
                                                   mask=mask,
                                                   one_step=False,
                                                   prefix='to_lstm')
        bo_lstm_h = bo_lstm[0]  # (t,64,512)
        to_lstm_h = to_lstm[0]  # (t,64,512)
        alphas = bo_lstm[2]
        ctxs = bo_lstm[3]
        betas = bo_lstm[4]  # (t,m)
        if options['use_dropout']:
            bo_lstm_h = self.layers.dropout_layer(bo_lstm_h, use_noise)
            to_lstm_h = self.layers.dropout_layer(to_lstm_h, use_noise)
        # compute word probabilities
        logit = self.layers.get_layer('ff')[1](tfparams, bo_lstm_h, options, prefix='ff_logit_bo', activ='linear')  # (t,64,512)*(512,512) = (t,64,512)
        if options['prev2out']:
            logit += inputs
        if options['ctx2out']:
            to_lstm_h *= (1-betas[:, :, None])  # (t,64,512)*(t,64,1)
            ctxs_beta = self.layers.get_layer('ff')[1](tfparams, ctxs, options, prefix='ff_logit_ctx', activ='linear')  # (t,64,2048)*(2048,512) = (t,64,512)
            ctxs_beta += self.layers.get_layer('ff')[1](tfparams, to_lstm_h, options, prefix='ff_logit_to', activ='linear') # (t,64,512)+((t,64,512)*(512,512)) = (t,64,512)
            logit += ctxs_beta
        logit = utils.tanh(logit)   # (t,64,512)
        if options['use_dropout']:
            logit = self.layers.dropout_layer(logit, use_noise)
        # (t,m,n_words)
        logit = self.layers.get_layer('ff')[1](tfparams, logit, options, prefix='ff_logit', activ='linear') # (t,64,512)*(512,vocab_size) = (t,64,vocab_size)
        logit_shape = tf.shape(logit)
        # (t*m, n_words)
        probs = tf.nn.softmax(tf.reshape(logit,[logit_shape[0] * logit_shape[1], logit_shape[2]]))    # (t*64, vocab_size)
        # cost
        x_flat = tf.reshape(x,[x_shape[0] * x_shape[1]])  # (t*m,)
        x_flat_shape = tf.shape(x_flat)
        gather_indices = tf.stack([tf.range(x_flat_shape[0]), x_flat], axis=1)  # (t*m,2)
        cost = -tf.log(tf.gather_nd(probs,gather_indices) + 1e-8) # (t*m,) : pick probs of each word in each timestep 
        cost = tf.reshape(cost,[x_shape[0], x_shape[1]])    # (t,m)
        cost = tf.reduce_sum((cost * mask), axis=0) # (m,) : sum across all timesteps for each element in batch
        extra = [probs, alphas, betas]
        return use_noise, cost, extra