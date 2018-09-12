import tensorflow as tf
import numpy as np
from utils import _p, norm_weight, ortho_weight, tanh, linear, batch_matmul

class Layers(object):

    def __init__(self):
        # layers: 'name': ('parameter initializer', 'feedforward')
        self.layers = {
            'ff': ('self.param_init_fflayer', 'self.fflayer'),
            'lstm': ('self.param_init_lstm', 'self.lstm_layer'),
            'lstm_cond': ('self.param_init_lstm_cond', 'self.lstm_cond_layer'),
            }

    def get_layer(self, name):
        """
        Part of the reason the init is very slow is because,
        the layer's constructor is called even when it isn't needed
        """
        fns = self.layers[name]
        return (eval(fns[0]), eval(fns[1]))

    # dropout
    def dropout_layer(self, state_before, use_noise):
        proj = tf.cond(use_noise,
                        lambda: tf.nn.dropout(state_before, keep_prob=0.5)*0.5, # DOUBT
                        lambda: state_before * 0.5)
        return proj

    def param_init_fflayer(self, options, params, prefix='ff', nin=None, nout=None):
        if nin == None:
            nin = options['ctx_dim']
        if nout == None:
            nout = options['lstm_dim']
        params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01)
        params[_p(prefix, 'b')] = np.zeros((nout,)).astype('float32')
        return params

    def fflayer(self, tfparams, state_below, options, 
                prefix='rconv', activ='lambda x: tf.tanh(x)', **kwargs):
        return eval(activ)(batch_matmul(state_below, tfparams[_p(prefix, 'W')]) + tfparams[_p(prefix, 'b')])

    # LSTM layer
    def param_init_lstm(self, params, nin, dim, prefix='lstm'):
        assert prefix is not None
        # Stack the weight matricies for faster dot prods
        W = np.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        params[_p(prefix, 'W')] = W     # to_lstm_W:(512,2048)
        U = np.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix, 'U')] = U     # to_lstm_U:(512,2048)
        params[_p(prefix, 'b')] = np.zeros((4 * dim,)).astype('float32')    # to_lstm_b:(2048,)
        return params

    # This function implements the lstm fprop
    def lstm_layer(self, tfparams, state_below, mask=None, init_state=None, init_memory=None,
                   one_step=False, prefix='lstm', **kwargs):
        # state_below (t, m, dim_word), or (m, dim_word) in sampling

        if one_step:
            if init_memory is None:
                raise ValueError('previous memory must be provided')
            if init_state is None:
                raise ValueError('previous state must be provided')
        
        dim = tfparams[_p(prefix, 'U')].shape[0]
        state_below_shape = tf.shape(state_below)
        if state_below.shape.ndims == 3:
            n_samples = state_below_shape[1]
        else:   
            n_samples = 1

        # mask
        if mask is None:
            mask = tf.fill([state_below_shape[0]], np.float32(1.))
        if init_state is None:
            # init_state = tf.constant(0., shape=(n_samples, dim), dtype=tf.float32)  # DOUBT ? getting same ans for tf.variable and tf.constant
            init_state = tf.fill([n_samples, dim], np.float32(0.))
        if init_memory is None:
            # init_memory = tf.constant(0., shape=(n_samples, dim), dtype=tf.float32)
            init_memory = tf.fill([n_samples, dim], np.float32(0.))

        def _slice(_x, n, dim):
            if _x.shape.ndims == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            elif _x.shape.ndims == 2:
                return _x[:, n * dim:(n + 1) * dim]
            return _x[n * dim:(n + 1) * dim]

        U = tfparams[_p(prefix, 'U')]
        b = tfparams[_p(prefix, 'b')]

        def step(prev, elems):
            m_, x_ = elems
            h_, c_ = prev
            preact = tf.matmul(h_, U)   # (64,512)*(512,2048) = (64,2048) or (1,2048) in sampling
            preact = preact + x_
            i = tf.sigmoid(_slice(preact, 0, dim))  # (64,512)
            f = tf.sigmoid(_slice(preact, 1, dim))  # (64,512)
            o = tf.sigmoid(_slice(preact, 2, dim))  # (64,512)
            c = tf.tanh(_slice(preact, 3, dim)) # (64,512)
            c = f * c_ + i * c
            h = o * tf.tanh(c)
            if m_.shape.ndims == 0:
                # when using this for minibatchsize=1
                h = m_ * h + (1. - m_) * h_
                c = m_ * c + (1. - m_) * c_
            else:
                h = m_[:, None] * h + (1. - m_)[:, None] * h_
                c = m_[:, None] * c + (1. - m_)[:, None] * c_
            return [h, c]

        state_below = batch_matmul(state_below, tfparams[_p(prefix, 'W')]) + b  # (19,64,512)*(512,2048)+(2048,) = (19,64,2048) or (m,2048) in sampling

        if one_step:
            rval = step(elems=[mask, state_below], prev=[init_state, init_memory])
        else:
            rval = tf.scan(step, 
                    (mask,state_below),
                    initializer=[init_state,init_memory],
                    name=_p(prefix, '_layers'))
        return rval

    # Conditional LSTM layer with Attention
    def param_init_lstm_cond(self, options, params,
                             prefix='lstm_cond', nin=None, dim=None, dimctx=None):  #nin=512 dim=512 dimctx=2048
        if nin == None:
            nin = options['word_dim']
        if dim == None:
            dim = options['lstm_dim']
        if dimctx == None:
            dimctx = options['ctx_dim']
        # input to LSTM
        W = np.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        params[_p(prefix, 'W')] = W     # bo_lstm_W:(512,2048)
        # LSTM to LSTM
        U = np.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix, 'U')] = U     # bo_lstm_U:(512,2048)
        # bias to LSTM
        params[_p(prefix, 'b')] = np.zeros((4 * dim,)).astype('float32')      # bo_lstm_b:(2048,)
        # attention: context -> hidden
        Wc_att = norm_weight(dimctx, ortho=False)
        params[_p(prefix, 'Wc_att')] = Wc_att    # bo_lstm_Wc_att:(2048,2048)
        # attention: LSTM -> hidden
        Wd_att = norm_weight(dim, dimctx)
        params[_p(prefix, 'Wd_att')] = Wd_att   # bo_lstm_Wd_att:(512,2048)
        # attention: hidden bias
        b_att = np.zeros((dimctx,)).astype('float32')
        params[_p(prefix, 'b_att')] = b_att     # bo_lstm_b_att:(2048,)
        # attention:
        U_att = norm_weight(dimctx, 1)
        params[_p(prefix, 'U_att')] = U_att      # bo_lstm_U_att:(2048,1)
        c_att = np.zeros((1,)).astype('float32')
        params[_p(prefix, 'c_att')] = c_att  # bo_lstm_c_att:(1,)
        if options['selector']:
            # attention: selector
            W_sel = norm_weight(dim, 1)
            params[_p(prefix, 'W_sel')] = W_sel     # bo_lstm_W_sel:(512,1)
            b_sel = np.float32(0.)
            params[_p(prefix, 'b_sel')] = b_sel     # bo_lstm_b_sel: 0
        return params

    def lstm_cond_layer(self, tfparams, state_below, options, prefix='lstm',
                        mask=None, context=None, one_step=False,
                        init_memory=None, init_state=None,
                        trng=None, use_noise=None, mode=None,
                        **kwargs):
        # state_below (t, m, dim_word), or (m, dim_word) in sampling
        # mask (t, m)
        # context (m, f, dim_ctx), or (1, f, dim_ctx) in sampling
        # init_memory, init_state (m , dim)
        # t = time steps
        # m = batch size

        if context is None:
                raise ValueError('Context must be provided')

        if one_step:
            if init_memory is None:
                raise ValueError('previous memory must be provided')
            if init_state is None:
                raise ValueError('previous state must be provided')

        state_below_shape = tf.shape(state_below)
        if state_below.shape.ndims == 3:
            n_samples = state_below_shape[1]
        else:
            n_samples = 1
        dim = tfparams[_p(prefix, 'U')].shape[0]

        if mask is None:
            mask = tf.fill([state_below_shape[0]], np.float32(1.))   # (m,) in sampling DOUBT ? (m, 1) or (m, )
        if init_state is None:
            init_state = tf.constant(0., shape=(n_samples, dim), dtype=tf.float32)  # DOUBT ? getting same ans for tf.variable and tf.constant
        if init_memory is None:
            init_memory = tf.constant(0., shape=(n_samples, dim), dtype=tf.float32)

        # projected context
        pctx_ = batch_matmul(context, tfparams[_p(prefix, 'Wc_att')]) + tfparams[_p(prefix, 'b_att')]    # (64,28,2048)*(2048,2048)+(2048,) = (64,28,2048) or (1,28,2048) in sampling
        if one_step:
            # tensor.dot will remove broadcasting dim
            # raise NotImplementedError()
            pass    # INCOMPLETE
        # projected x
        state_below = batch_matmul(state_below, tfparams[_p(prefix, 'W')]) + tfparams[_p(prefix, 'b')]    # (19,64,512)*(512,2048)+(2048) = (19,64,2048) or (m,2048) in sampling
        Wd_att = tfparams[_p(prefix, 'Wd_att')]  # (512,2048)
        U_att = tfparams[_p(prefix, 'U_att')]    # (2048,1)
        c_att = tfparams[_p(prefix, 'c_att')] # (1,)
        if options['selector']:
            W_sel = tfparams[_p(prefix, 'W_sel')]
            b_sel = tfparams[_p(prefix, 'b_sel')]
        else:
            # W_sel = tensor.alloc(0., 1)
            # b_sel = tensor.alloc(0., 1)
            raise NotImplementedError()
        U = tfparams[_p(prefix, 'U')]    # (512,2048)

        pctx_shape = tf.shape(pctx_)
        # init_alpha = tf.constant(0.,shape=(n_samples, pctx_shape[1]), dtype=tf.float32)
        init_alpha = tf.fill([n_samples, pctx_shape[1]], np.float32(0.))
        # init_ctx = tf.constant(0.,shape=(n_samples, U.shape[1]), dtype=tf.float32)
        init_ctx = tf.fill([n_samples, U.shape[1]], np.float32(0.))
        # init_beta = tf.constant(0.,shape=(n_samples, ), dtype=tf.float32)
        init_beta = tf.fill([n_samples,], np.float32(0.))

        def _slice(_x, n, dim):
            if _x.shape.ndims == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def step(prev, elems):
            # gather previous internal state and output state
            if options['use_dropout']:
                m_, x_, dp_ = elems
            else:
                m_, x_ = elems
            h_, c_, _, _, _ = prev
            preact = tf.matmul(h_, U)   # (64,512)*(512,2048) = (64,2048) or (m,2048) in sampling
            preact = preact + x_
            i = _slice(preact, 0, dim)  # (64,512)  (0-511) or (m,512) in sampling
            f = _slice(preact, 1, dim)  # (64,512)  (512,1023)  or (m,512) in sampling
            o = _slice(preact, 2, dim)  # (64,512)  (1024-1535) or (m,512) in sampling
            if options['use_dropout']:
                i = i * _slice(dp_, 0, dim)
                f = f * _slice(dp_, 1, dim)
                o = o * _slice(dp_, 2, dim)
            i = tf.sigmoid(i)
            f = tf.sigmoid(f)
            o = tf.sigmoid(o)
            c = tf.tanh(_slice(preact, 3, dim))  # (64,512)  (1024-1535)    or (m,512) in sampling
            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_   # (m,1)*(m,512) + (m,1)*(m,512) = (m,512) in sampling
            h = o * tf.tanh(c)  # (m,512)*(m,512) = (m,512) in sampling
            h = m_[:, None] * h + (1. - m_)[:, None] * h_
            # attention
            pstate_ = tf.matmul(h, Wd_att) # shape = (64,512)*(512,2048) = (64,2048) or (m,2048) in sampling
            pctx_t = pctx_ + pstate_[:, None, :] # shape = (64,28,2048)+(64,?,2048) = (64,28,2048)  # DOUBT pctx_ += ??
                #   (1,28,2048) + (m,?,2048) = (m,28,2048)
            pctx_t = tanh(pctx_t)
            alpha = batch_matmul(pctx_t, U_att) + c_att    # (64,28,2048)*(2048,1) + (1,) = (64,28,1) or (m,28,1) in sampling
            alpha_pre = alpha
            alpha_shape = tf.shape(alpha)
            alpha = tf.nn.softmax(tf.reshape(alpha,[alpha_shape[0], alpha_shape[1]]))  # softmax (64,28) or (m,28) in sampling
            ctx_ = tf.reduce_sum((context * alpha[:, :, None]), 1)  # (m, ctx_dim)     # (64*28*2048)*(64,28,1).sum(1) = (64,2048) or (m,2048) in sampling
            if options['selector']:
                sel_ = tf.sigmoid(tf.matmul(h_, W_sel) + b_sel)   # (64,512)*(512,1)+(scalar) = (64,1) or (m,1) in sampling
                sel_shape = tf.shape(sel_)
                sel_ = tf.reshape(sel_,[sel_shape[0]])    # (64,) or (m,) in sampling
                ctx_ = sel_[:, None] * ctx_     # (64,1)*(64,2048) = (64,2048) or (m,2048) in sampling
            rval = [h, c, alpha, ctx_, sel_]
            return rval

        if options['use_dropout']:
            dp_shape = tf.shape(state_below)
            if one_step:
                dp_mask = tf.cond(use_noise,
                                lambda: tf.nn.dropout(tf.fill([dp_shape[0], 3 * dim], np.float32(0.5)), keep_prob=0.5),
                                lambda: tf.fill([dp_shape[0], 3 * dim], np.float32(0.5)))
            else:
                dp_mask = tf.cond(use_noise,
                                lambda: tf.nn.dropout(tf.fill([dp_shape[0], dp_shape[1], 3 * dim], np.float32(0.5)), keep_prob=0.5),
                                lambda: tf.fill([dp_shape[0], dp_shape[1], 3 * dim], np.float32(0.5)))

        if one_step:
            if options['use_dropout']:
                rval = step(elems=[mask, state_below, dp_mask], prev=[init_state, init_memory, init_alpha, init_ctx, init_beta])
            else:
                rval = step(elems=[mask, state_below], prev=[init_state, init_memory, init_alpha, init_ctx, init_beta])
        else:
            seqs = [mask, state_below]
            if options['use_dropout']:
                seqs.append(dp_mask)
            rval = tf.scan(step, 
                            seqs,
                            initializer=[init_state, init_memory, init_alpha, init_ctx, init_beta],
                            name=_p(prefix, '_layers'))
        return rval

    # def create_get_lstm_cell(prefix, is_training=True, rnn_mode="basic"):
    #     if rnn_mode == "basic":
    #       return tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=0.0, state_is_tuple=True, reuse=not is_training, name=prefix)
    #     if rnn_mode == "block":
    #       return tf.contrib.rnn.LSTMBlockCell(lstm_size, forget_bias=0.0, name=prefix)
    #     raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

    # def make_cell(prefix, is_training=True, rnn_mode="basic"):
    #     cell = _get_lstm_cell(is_training, rnn_mode)
    #     # if is_training and config.keep_prob < 1:
    #     #   cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    #     return cell

    # # Incomplete
    # def build_hlstm(prefix, inputs, n_timesteps, init_state, init_memory, is_training=True, rnn_mode="basic"):
    #     """Build the inference graph using canonical LSTM cells."""
    #     # Slightly better results can be obtained with forget gate biases
    #     # initialized to 1 but the hyperparameters of the model would need to be
    #     # different than reported in the paper.
    #     cell = tf.contrib.rnn.MultiRNNCell([make_cell(lstm_name, is_training, rnn_mode) for lstm_name in prefix], state_is_tuple=True)
    #     # Simplified version of tf.nn.static_rnn().
    #     # This builds an unrolled LSTM for tutorial purposes only.
    #     # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
    #     #
    #     # The alternative version of the code below is:
    #     #
    #     # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    #     # outputs, state = tf.nn.static_rnn(cell, inputs, initial_state=self._initial_state)
    #     outputs = []
    #     with tf.variable_scope("RNN"):
    #       for time_step in range(n_timesteps):
    #         if time_step > 0: tf.get_variable_scope().reuse_variables()
    #         (cell_output, state) = cell(inputs[time_step, :, :], state)
    #         outputs.append(cell_output)
    #     output = tf.reshape(tf.concat(outputs, 1), [-1, lstm_size])
    #     return output, state