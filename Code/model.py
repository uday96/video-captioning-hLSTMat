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
        # embedding
        params['Wemb'] = utils.norm_weight(options['vocab_size'], options['word_dim'])
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
        use_noise = tf.Variable(False, dtype=tf.bool, trainable=False, name="use_noise")
        x_shape = tf.shape(x)
        n_timesteps = x_shape[0]
        n_samples = x_shape[1]
        # get word embeddings
        emb = tf.nn.embedding_lookup(tfparams['Wemb'], x, name="inputs_emb_lookup")    # (num_steps,64,512)
        emb_shape = tf.shape(emb)
        indices = tf.expand_dims(tf.range(1,emb_shape[0]), axis=1)
        emb_shifted = tf.scatter_nd(indices, emb[:-1], emb_shape)
        emb = emb_shifted

        # count num_frames==28
        with tf.name_scope("ctx_mean"):
            with tf.name_scope("counts"):
                counts = tf.expand_dims(tf.reduce_sum(ctx_mask, axis=-1, name="reduce_sum_ctx_mask"), 1)  # (64,1)
            ctx_ = ctx
            ctx0 = ctx_     # (64,28,2048)
            ctx_mean = tf.reduce_sum(ctx0, axis=1, name="reduce_sum_ctx") / counts  #mean pooling of {vi}   # (64,2048)

        # initial state/cell
        with tf.name_scope("init_state"):
            init_state = self.layers.get_layer('ff')[1](tfparams, ctx_mean, options, prefix='ff_state', activ='tanh')   # (64,512)

        with tf.name_scope("init_memory"):
            init_memory = self.layers.get_layer('ff')[1](tfparams, ctx_mean, options, prefix='ff_memory', activ='tanh') # (64,512)

        # hstltm = self.layers.build_hlstm(['bo_lstm','to_lstm'], inputs, n_timesteps, init_state, init_memory)
        with tf.name_scope("bo_lstm"):
            bo_lstm = self.layers.get_layer('lstm_cond')[1](tfparams, emb, options,
                                                            prefix='bo_lstm',
                                                            mask=mask, context=ctx0,
                                                            one_step=False,
                                                            init_state=init_state,
                                                            init_memory=init_memory,
                                                            use_noise=use_noise)
        with tf.name_scope("to_lstm"):
            to_lstm = self.layers.get_layer('lstm')[1](tfparams, bo_lstm[0],
                                                       mask=mask,
                                                       one_step=False,
                                                       prefix='to_lstm')
        bo_lstm_h = bo_lstm[0]  # (t,64,512)
        to_lstm_h = to_lstm[0]  # (t,64,512)
        alphas = bo_lstm[2] # (t,64,28)
        ctxs = bo_lstm[3]   # (t,64,2048)
        betas = bo_lstm[4]  # (t,64,)
        if options['use_dropout']:
            bo_lstm_h = self.layers.dropout_layer(bo_lstm_h, use_noise)
            to_lstm_h = self.layers.dropout_layer(to_lstm_h, use_noise)
        # compute word probabilities
        logit = self.layers.get_layer('ff')[1](tfparams, bo_lstm_h, options, prefix='ff_logit_bo', activ='linear')  # (t,64,512)*(512,512) = (t,64,512)
        if options['prev2out']:
            logit += emb
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

    def build_sampler(self, tfparams, options, use_noise, ctx0, ctx_mask, x,
                    bo_init_state_sampler, to_init_state_sampler, bo_init_memory_sampler, to_init_memory_sampler, mode=None):
        # ctx: # frames x ctx_dim
        ctx_ = ctx0
        counts = tf.reduce_sum(ctx_mask, axis=-1)   # scalar

        ctx = ctx_
        ctx_mean = tf.reduce_sum(ctx, axis=0) / counts  # (2048,)
        ctx = tf.expand_dims(ctx, 0)  # (1,28,2048)

        # initial state/cell
        bo_init_state = self.layers.get_layer('ff')[1](tfparams, ctx_mean, options, prefix='ff_state', activ='tanh')    # (512,)
        bo_init_memory = self.layers.get_layer('ff')[1](tfparams, ctx_mean, options, prefix='ff_memory', activ='tanh')  # (512,)
        to_init_state = tf.zeros(shape=(options['lstm_dim'],), dtype=tf.float32)  # DOUBT : constant or not? # (512,)
        to_init_memory = tf.zeros(shape=(options['lstm_dim'],), dtype=tf.float32)    # (512,)
        init_state = [bo_init_state, to_init_state]
        init_memory = [bo_init_memory, to_init_memory]

        print 'building f_init...',
        f_init = [ctx0] + init_state + init_memory
        print 'done'

        init_state = [bo_init_state_sampler, to_init_state_sampler]
        init_memory = [bo_init_memory_sampler, to_init_memory_sampler]

        # # if it's the first word, embedding should be all zero
        emb = tf.cond(tf.reduce_any(x[:, None] < 0),
                        lambda: tf.zeros(shape=(1,tfparams['Wemb'].shape[1]), dtype=tf.float32),
                        lambda: tf.nn.embedding_lookup(tfparams['Wemb'], x))    # (m,512)

        bo_lstm = self.layers.get_layer('lstm_cond')[1](tfparams, emb, options,
                                                        prefix='bo_lstm',
                                                        mask=None, context=ctx,
                                                        one_step=True,
                                                        init_state=init_state[0],
                                                        init_memory=init_memory[0],
                                                        use_noise=use_noise,
                                                        mode=mode)
        to_lstm = self.layers.get_layer('lstm')[1](tfparams, bo_lstm[0],
                                                   mask=None,
                                                   one_step=True,
                                                   init_state=init_state[1],
                                                   init_memory=init_memory[1],
                                                   prefix='to_lstm'
                                                   )
        next_state = [bo_lstm[0], to_lstm[0]]
        next_memory = [bo_lstm[1], to_lstm[0]]

        bo_lstm_h = bo_lstm[0]  # (1,512)
        to_lstm_h = to_lstm[0]  # (1,512)
        alphas = bo_lstm[2] # (1,28)
        ctxs = bo_lstm[3]   # (1,2048)
        betas = bo_lstm[4]  # (1,)
        if options['use_dropout']:
            bo_lstm_h = self.layers.dropout_layer(bo_lstm_h, use_noise)
            to_lstm_h = self.layers.dropout_layer(to_lstm_h, use_noise)
        # compute word probabilities
        logit = self.layers.get_layer('ff')[1](tfparams, bo_lstm_h, options, prefix='ff_logit_bo', activ='linear')  # (1,512)*(512,512) = (1,512)
        if options['prev2out']:
            logit += emb
        if options['ctx2out']:
            to_lstm_h *= (1-betas[:, None])  # (1,512)*(1,1) = (1,512)
            ctxs_beta = self.layers.get_layer('ff')[1](tfparams, ctxs, options, prefix='ff_logit_ctx', activ='linear')  # (1,2048)*(2048,512) = (1,512)
            ctxs_beta += self.layers.get_layer('ff')[1](tfparams, to_lstm_h, options, prefix='ff_logit_to', activ='linear') # (1,512)+((1,512)*(512,512)) = (1,512)
            logit += ctxs_beta
        logit = utils.tanh(logit)   # (1,512)
        if options['use_dropout']:
            logit = self.layers.dropout_layer(logit, use_noise)
        # (1,n_words)
        logit = self.layers.get_layer('ff')[1](tfparams, logit, options, prefix='ff_logit', activ='linear') # (1,512)*(512,vocab_size) = (1,vocab_size)
        next_probs = tf.nn.softmax(logit)
        # next_sample = trng.multinomial(pvals=next_probs).argmax(1)    # INCOMPLETE , DOUBT : why is multinomial needed?
        next_sample = tf.multinomial(next_probs,1) # draw samples with given probabilities (1,1)
        next_sample_shape = tf.shape(next_sample)
        next_sample = tf.reshape(next_sample,[next_sample_shape[0]])
        # next word probability
        print 'building f_next...',
        f_next = [next_probs, next_sample] + next_state + next_memory
        print 'done'
        return f_init, f_next

    def gen_sample(self, sess, tfparams, f_init, f_next, ctx0, ctx_mask, options,
                   k=1, maxlen=30, stochastic=False, restrict_voc=False):
        '''
        ctx0: (28,2048) (f, dim_ctx)
        ctx_mask: (28,) (f, )

        restrict_voc: set the probability of outofvoc words with 0, renormalize
        '''
        if k > 1:
            assert not stochastic, 'Beam search does not support stochastic sampling'

        sample = []
        sample_score = []
        if stochastic:
            sample_score = 0

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype('float32')
        hyp_states = []
        hyp_memories = []

        # [(28,2048),(512,),(512,),(512,),(512,)]
        rval = sess.run(f_init, feed_dict={
                    "ctx_sampler:0": ctx0,
                    "ctx_mask_sampler:0": ctx_mask
                })
        ctx0 = rval[0]

        next_state = []
        next_memory = []
        n_layers_lstm = 2

        for lidx in xrange(n_layers_lstm):
            next_state.append(rval[1 + lidx])
            next_state[-1] = next_state[-1].reshape([live_k, next_state[-1].shape[0]])
        for lidx in xrange(n_layers_lstm):
            next_memory.append(rval[1 + n_layers_lstm + lidx])
            next_memory[-1] = next_memory[-1].reshape([live_k, next_memory[-1].shape[0]])
        next_w = -1 * np.ones((1,)).astype('int32')
        for ii in xrange(maxlen):
            # return [(1, vocab_size), (1,), (1, 512), (1, 512), (1, 512), (1, 512)]
            # next_w: vector (1,)
            # ctx: matrix   (28, 2048)
            # ctx_mask: vector  (28,)
            # next_state: [matrix] [(1, 512), (1, 512)]
            # next_memory: [matrix] [(1, 512), (1, 512)]
            rval = sess.run(f_next, feed_dict={
                        "x_sampler:0": next_w,
                        "ctx_sampler:0": ctx0,
                        "ctx_mask_sampler:0": ctx_mask,
                        'bo_init_state_sampler:0': next_state[0],
                        'to_init_state_sampler:0': next_state[1],
                        'bo_init_memory_sampler:0': next_memory[0],
                        'to_init_memory_sampler:0': next_memory[1]
                    })
            next_p = rval[0]
            if restrict_voc:
                raise NotImplementedError()
            next_w = rval[1]  # already argmax sorted
            next_state = []
            for lidx in xrange(n_layers_lstm):
                next_state.append(rval[2 + lidx])
            next_memory = []
            for lidx in xrange(n_layers_lstm):
                next_memory.append(rval[2 + n_layers_lstm + lidx])
            if stochastic:
                sample.append(next_w[0])  # take the most likely one
                sample_score += next_p[0, next_w[0]]
                if next_w[0] == 0:
                    break
            else:
                # the first run is (1,vocab_size)
                cand_scores = hyp_scores[:, None] - np.log(next_p)
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:(k - dead_k)]


                voc_size = next_p.shape[1]
                trans_indices = ranks_flat / voc_size  # index of row
                word_indices = ranks_flat % voc_size  # index of col
                costs = cand_flat[ranks_flat]

                new_hyp_samples = []
                new_hyp_scores = np.zeros(k - dead_k).astype('float32')
                new_hyp_states = []
                for lidx in xrange(n_layers_lstm):
                    new_hyp_states.append([])
                new_hyp_memories = []
                for lidx in xrange(n_layers_lstm):
                    new_hyp_memories.append([])

                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    for lidx in xrange(n_layers_lstm):
                        new_hyp_states[lidx].append(copy.copy(next_state[lidx][ti]))
                    for lidx in xrange(n_layers_lstm):
                        new_hyp_memories[lidx].append(copy.copy(next_memory[lidx][ti]))

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []
                for lidx in xrange(n_layers_lstm):
                    hyp_states.append([])
                hyp_memories = []
                for lidx in xrange(n_layers_lstm):
                    hyp_memories.append([])

                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == 0:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        for lidx in xrange(n_layers_lstm):
                            hyp_states[lidx].append(new_hyp_states[lidx][idx])
                        for lidx in xrange(n_layers_lstm):
                            hyp_memories[lidx].append(new_hyp_memories[lidx][idx])
                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break

                next_w = np.array([w[-1] for w in hyp_samples])
                next_state = []
                for lidx in xrange(n_layers_lstm):
                    next_state.append(np.array(hyp_states[lidx]))
                next_memory = []
                for lidx in xrange(n_layers_lstm):
                    next_memory.append(np.array(hyp_memories[lidx]))

        if not stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in xrange(live_k):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])

        return sample, sample_score, next_state, next_memory

    def pred_probs(self, sess, engine, whichset, f_log_probs, verbose=True):
        probs = []
        n_done = 0
        NLL = []
        L = []
        if whichset == 'train':
            tags = engine.train_data_ids
            iterator = engine.kf_train
        elif whichset == 'val':
            tags = engine.val_data_ids
            iterator = engine.kf_val
        elif whichset == 'test':
            tags = engine.test_data_ids
            iterator = engine.kf_test
        else:
            raise NotImplementedError()
        n_samples = np.sum([len(index) for index in iterator])
        for index in iterator:
            tag = [tags[i] for i in index]
            x, mask, ctx, ctx_mask = prepare_data(engine, tag, mode=whichset)
            
            pred_probs = sess.run(f_log_probs, feed_dict={
                                        "word_seq_x:0": x,
                                        "word_seq_mask:0": mask,
                                        "ctx:0": ctx,
                                        "ctx_mask:0": ctx_mask})
            
            L.append(mask.sum(0).tolist())
            NLL.append((-1 * pred_probs).tolist())
            probs.append(pred_probs.tolist())
            n_done += len(tag)
            if verbose:
                sys.stdout.write('\rComputing LL on %d/%d examples' % (
                    n_done, n_samples))
                sys.stdout.flush()
        print ""
        probs = utils.flatten_list_of_list(probs)
        NLL = utils.flatten_list_of_list(NLL)
        L = utils.flatten_list_of_list(L)
        perp = 2 ** (np.sum(NLL) / np.sum(L) / np.log(2))
        return -1 * np.mean(probs), perp

    def sample_execute(self, sess, engine, options, tfparams, f_init, f_next, x, ctx, ctx_mask):
        stochastic = False
        # x = (t,64)
        # ctx = (64,28,2048)
        # ctx_mask = (64,28)
        for jj in xrange(np.minimum(10, x.shape[1])):
            sample, score, _, _ = self.gen_sample(sess, tfparams, f_init, f_next, ctx[jj], ctx_mask[jj],
                                                  options, k=5, maxlen=30, stochastic=stochastic)
            if not stochastic:
                best_one = np.argmin(score)
                sample = sample[best_one]
            else:
                sample = sample
            print 'Truth ', jj, ': ',
            for vv in x[:, jj]:
                if vv == 0:
                    break
                if vv in engine.reverse_vocab:
                    print engine.reverse_vocab[vv],
                else:
                    print 'UNK',
            print ""
            for kk, ss in enumerate([sample]):
                print 'Sample (', jj, ') ', ': ',
                for vv in ss:
                    if vv == 0:
                        break
                    if vv in engine.reverse_vocab:
                        print engine.reverse_vocab[vv],
                    else:
                        print 'UNK',
            print ""